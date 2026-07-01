"""4B subject-축 실험 manifest 빌더 (cliff backbone).

중간선 설계: 난이도(level) 구성은 cliff_P와 **완전 동일**(per-(level,stage) 개수 고정),
subject만 다르게 배치해 'subject 기하가 난이도 너머 효과인가'를 난이도-매칭으로 검정.

- cliff_subjgeo (treatment): level 안에서 문제를 subject 기하 1D좌표로 정렬 → co-move(reinforcing).
    subject 좌표는 sim_matrices_pooled3025_levsubj.npz 코사인행렬의 leading MDS축,
    부호는 level과 양의 상관이 되게 고정(초기 stage=이산/저좌표, 후기=연속/고좌표).
- cliff_subjrand_s{0,1} (control): 같은 개수, level 안에서 subject 무작위(cond5 규율). seed별 재추첨.

세 arm 모두 per-(level,stage) 개수 동일 → subject 조성만 차이. 결정론적.
"""
import pandas as pd, numpy as np, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
REPO="/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum"
SCR="/tmp/claude-5021/-scratch-lami2026-personal-jimin-2782/d07f46df-c094-41ed-be6d-811de1613bd7/scratchpad"

rows=pd.read_parquet(f"{REPO}/training/outputs/join_setA_rows.parquet")
clean=set(pd.read_csv(f"{SCR}/clean_train_ids.csv")['problem_id'].astype(str))
uni=rows[(rows['in_setA']==True)&(rows['problem_id'].astype(str).isin(clean))].copy()
uni['problem_id']=uni['problem_id'].astype(str)
N=len(uni); print(f"universe={N}")

# ---- subject 기하 1D 좌표 (leading MDS, level과 양의 상관으로 부호 고정) ----
d=np.load(f"{REPO}/reasoning_pivot/activation/analysis/sim_matrices_pooled3025_levsubj.npz",
          allow_pickle=True)
S=d['THINKING_centered_subject_S'].astype(float)
sorder=[str(x) for x in d['THINKING_centered_subject_order']]
present=sorted(uni['subject'].dropna().unique().tolist())
si=[sorder.index(s) for s in present]
Ssub=S[np.ix_(si,si)]
Dm=1.0-Ssub; n=len(si); J=np.eye(n)-np.ones((n,n))/n
B=-0.5*J@(Dm**2)@J; w,V=np.linalg.eigh(B)
coord=V[:,-1]*np.sqrt(max(w[-1],0.0))
ml=uni.groupby('subject')['level'].mean(); mlv=np.array([ml[s] for s in present])
if np.corrcoef(coord,mlv)[0,1]<0: coord=-coord
SUBJ_COORD={s:float(c) for s,c in zip(present,coord)}
print("subject 스윕순서(초기→후기):", " → ".join(sorted(present,key=lambda s:SUBJ_COORD[s])))

uni['_sc']=uni['subject'].map(SUBJ_COORD)

# ---- cliff_P와 동일한 창/dwell/개수 로직 ----
CTX={0:1024,1:1536,2:2048,3:2560,4:3072,5:4096}
CTX6=[CTX[k] for k in range(6)]
WIN={0:[1,2,3],1:[2,3,4],2:[3,4,5],3:[4,5,6],4:[5,6,7],5:[6,7,8]}
DW ={0:13,1:23,2:25,3:16,4:11,5:11}

def level_stage_counts():
    """cliff_P와 동일한 per-(level,stage) 개수 (rng 무관, 결정론적)."""
    counts={}
    for L in range(1,9):
        nL=int((uni['level']==L).sum())
        elig=[k for k in WIN if L in WIN[k]]
        wv=np.array([DW[k] for k in elig],float); wv/=wv.sum()
        bnds=np.floor(np.cumsum(wv)*nL).astype(int)
        start=0; c={}
        for j,(k,b) in enumerate(zip(elig,bnds)):
            end = nL if j==len(elig)-1 else b   # 마지막 창은 나머지 흡수
            c[k]=end-start; start=end
        counts[L]=(elig,c)
    return counts

CNT=level_stage_counts()

def write(arm, construction, stages):
    man={"arm":arm,"construction":construction,"universe_N":N,"n_stages":6,
         "context_per_stage":CTX6,
         "stages":[{"stage_index":k,"n":len(s),"context_len":CTX6[k],"problem_ids":s}
                   for k,s in enumerate(stages)]}
    json.dump(man,open(f"{HERE}/stages_{arm}.json",'w'))
    tot=sum(len(s) for s in stages); uniq=len(set(x for s in stages for x in s))
    print(f"  [{arm}] 6 stage, 총 {tot} (고유 {uniq}) | stage크기 {[len(s) for s in stages]}")
    return stages

def build(arm, construction, order_fn):
    """order_fn(level_df) -> problem_id 리스트 (level 내 배치 순서). 개수는 CNT로 슬라이스."""
    stages=[[] for _ in range(6)]
    for L in range(1,9):
        elig,c=CNT[L]
        pool=order_fn(uni[uni['level']==L])
        start=0
        for k in elig:
            stages[k].extend(pool[start:start+c[k]]); start+=c[k]
        assert start==len(pool), f"L{L}: {start} != {len(pool)}"
    return write(arm,construction,stages)

# treatment: subject 좌표 오름차순(동률은 problem_id로 결정론적)
def order_geo(df):
    return df.sort_values(['_sc','problem_id'])['problem_id'].tolist()

# control: level 내 subject 무작위 (seed별)
def order_rand(seed):
    def f(df):
        rng=np.random.default_rng(seed)
        ids=df['problem_id'].tolist()
        return [ids[i] for i in rng.permutation(len(ids))]
    return f

print("\n=== 빌드 ===")
geo=build("cliff_subjgeo","cliff_backbone_subject_geo_comove",order_geo)
r0 =build("cliff_subjrand_s0","cliff_backbone_subject_random_seed0",order_rand(0))
r1 =build("cliff_subjrand_s1","cliff_backbone_subject_random_seed1",order_rand(1))

# ---- 검증: 세 arm의 per-(level,stage) 개수 동일한가 + subject 조성 스윕 확인 ----
def per_stage_level_counts(stages):
    id2lvl=dict(zip(uni['problem_id'],uni['level']))
    return [tuple(sorted((L,sum(1 for x in s if id2lvl[x]==L)) for L in range(1,9))) for s in stages]
assert per_stage_level_counts(geo)==per_stage_level_counts(r0)==per_stage_level_counts(r1), "level 개수 불일치!"
print("\n[검증] treatment/control 세 arm의 per-(level,stage) 개수 동일 ✓")

id2sc=dict(zip(uni['problem_id'],uni['_sc']))
print("\nstage별 평균 subject좌표 (co-move면 단조증가 기대):")
print("  geo :", [round(np.mean([id2sc[x] for x in s]),3) for s in geo])
print("  rand:", [round(np.mean([id2sc[x] for x in s]),3) for s in r0])
print("\n전부 생성 완료 ->", HERE)
