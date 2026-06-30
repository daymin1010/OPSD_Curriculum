"""4B 커리큘럼 manifest 빌더 — 7 arm. clean universe(in_setA & 오염제거) 기준.
출력: stages_<arm>.json  (stages[*].problem_ids + stage별 context_len).
결정론적(고정 seed). 활성 재계산 없음."""
import pandas as pd, numpy as np, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
REPO="/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum"
SCR="/tmp/claude-5021/-scratch-lami2026-personal-jimin-2782/d07f46df-c094-41ed-be6d-811de1613bd7/scratchpad"

rows=pd.read_parquet(f"{REPO}/training/outputs/join_setA_rows.parquet")
clean=set(pd.read_csv(f"{SCR}/clean_train_ids.csv")['problem_id'].astype(str))
uni=rows[(rows['in_setA']==True)&(rows['problem_id'].astype(str).isin(clean))].copy()
uni['problem_id']=uni['problem_id'].astype(str)
N=len(uni); print(f"universe={N}")

# 레벨/과목 → 문제 id 리스트
def ids_where(mask): return uni.loc[mask,'problem_id'].tolist()
lvl_ids={L: ids_where(uni['level']==L) for L in range(1,9)}
CLU={'C_alg':['Algebra','Intermediate Algebra','Precalculus'],'C_geo':['Geometry'],
     'C_disc':['Counting & Probability','Number Theory','Prealgebra']}
CTX={0:1024,1:1536,2:2048,3:2560,4:3072,5:4096}  # cliff stage별 context
def write(arm, construction, stages, ctx_list):
    man={"arm":arm,"construction":construction,"universe_N":N,"n_stages":len(stages),
         "context_per_stage":ctx_list,
         "stages":[{"stage_index":k,"n":len(s),"context_len":ctx_list[k],"problem_ids":s}
                   for k,s in enumerate(stages)]}
    p=f"{HERE}/stages_{arm}.json"; json.dump(man,open(p,'w'))
    tot=sum(len(s) for s in stages)
    print(f"  [{arm}] {len(stages)} stage, 총배정 {tot} (고유 {len(set(x for s in stages for x in s))}), ctx {ctx_list}")
    return man

rng=np.random.default_rng(0)
WIN={0:[1,2,3],1:[2,3,4],2:[3,4,5],3:[4,5,6],4:[5,6,7],5:[6,7,8]}  # cliff 슬라이딩 창
DW ={0:13,1:23,2:25,3:16,4:11,5:11}                                # cliff dwell(분배 비율)
CTX6=[CTX[k] for k in range(6)]

# ---------- 1. shuffle (무커리큘럼) : 전체 랜덤 → 6 등분 ----------
allids=uni['problem_id'].tolist(); perm=list(rng.permutation(allids))
cut=[perm[k*N//6:(k+1)*N//6] for k in range(6)]
write("shuffle","full_random_no_curriculum",cut,CTX6)

# ---------- 2. diff : level 정렬 → 6 등질량(tight 밴드) ----------
s=uni.sort_values(['level']).copy(); s['j']=rng.random(len(s)); s=s.sort_values(['level','j'])
order=s['problem_id'].tolist()
cut=[order[k*N//6:(k+1)*N//6] for k in range(6)]
write("diff","level_sorted_equalmass_tight",cut,CTX6)

# ---------- 3. cliff-P : 레벨을 창에 dwell비율로 *분배* (partition) ----------
stages=[[] for _ in range(6)]
for L in range(1,9):
    elig=[k for k in WIN if L in WIN[k]]; w=np.array([DW[k] for k in elig],float); w/=w.sum()
    pool=list(rng.permutation(lvl_ids[L])); n=len(pool)
    bnds=np.floor(np.cumsum(w)*n).astype(int)
    start=0
    for k,b in zip(elig,bnds):
        stages[k].extend(pool[start:b]); start=b
    stages[elig[-1]].extend(pool[start:])  # 나머지
write("cliff_P","sliding_window_partition_distribute",stages,CTX6)

# ---------- 6. subj-V1 : 과목 blocked(대수→기하→이산), 블록내 쉬움(L1-4)→어려움(L5-8) ----------
def subj_ids(subs,lo,hi):
    m=uni['subject'].isin(subs)&(uni['level']>=lo)&(uni['level']<=hi);
    return list(rng.permutation(ids_where(m)))
blocks=[]
for cn,subs in [('C_alg',CLU['C_alg']),('C_geo',CLU['C_geo']),('C_disc',CLU['C_disc'])]:
    blocks.append(subj_ids(subs,1,4)); blocks.append(subj_ids(subs,5,8))
# context: 과목블록은 난이도 톱니라 블록 최난도 기준 → 쉬움블록 2048, 어려움블록 4096
ctx_subj=[2048,4096,2048,4096,2048,4096]
write("subj_V1","subject_blocked_seq_alg_geo_disc",blocks,ctx_subj)

# ---------- 7. subj-shuf : 같은 6 블록, 블록 순서 무작위 ----------
idx=list(rng.permutation(range(6))); blocks_s=[blocks[i] for i in idx]; ctx_s=[ctx_subj[i] for i in idx]
write("subj_shuf",f"subject_blocked_random_order_{idx}",blocks_s,ctx_s)

print("\n전부 생성 완료 ->", HERE)
