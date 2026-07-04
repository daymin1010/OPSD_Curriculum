"""subjgeo 완만 버전 빌더 — 난이도(level×stage 개수)는 cliff_subjgeo와 완전 동일,
subject 정렬 진폭 alpha만 축소(geo정렬↔무작위 blend). alpha=1 원본, 0 무작위."""
import pandas as pd, numpy as np, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
REPO="/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum"
rows=pd.read_parquet(f"{REPO}/training/outputs/join_setA_rows.parquet"); rows['problem_id']=rows['problem_id'].astype(str)
geo=json.load(open(f"{HERE}/stages_cliff_subjgeo.json"))
stage_ids=[[str(x) for x in st["problem_ids"]] for st in geo["stages"]]
uni=rows[rows['problem_id'].isin(set(x for s in stage_ids for x in s))].copy()
id2lvl=dict(zip(uni.problem_id,uni.level))
CTX6=[1024,1536,2048,2560,3072,4096]; N=len(uni)
d=np.load(f"{REPO}/reasoning_pivot/activation/analysis/sim_matrices_pooled3025_levsubj.npz",allow_pickle=True)
S=d['THINKING_centered_subject_S'].astype(float); sorder=[str(x) for x in d['THINKING_centered_subject_order']]
present=sorted(uni['subject'].dropna().astype(str).unique().tolist()); si=[sorder.index(s) for s in present]
Ssub=S[np.ix_(si,si)]; Dm=1.0-Ssub; n=len(si); J=np.eye(n)-np.ones((n,n))/n
B=-0.5*J@(Dm**2)@J; w,V=np.linalg.eigh(B); coord=V[:,-1]*np.sqrt(max(w[-1],0.0))
ml=uni.groupby('subject')['level'].mean(); mlv=np.array([ml[s] for s in present])
if np.corrcoef(coord,mlv)[0,1]<0: coord=-coord
SC={s:float(c) for s,c in zip(present,coord)}; uni['_sc']=uni['subject'].astype(str).map(SC)
CNT={}
for L in range(1,9):
    elig=[k for k in range(6) if any(id2lvl[i]==L for i in stage_ids[k])]
    CNT[L]=(elig,{k:sum(1 for i in stage_ids[k] if id2lvl[i]==L) for k in elig})
def order_blend(alpha,seed):
    def f(df):
        rng=np.random.default_rng(seed); sc=df['_sc'].values
        r=sc.argsort().argsort().astype(float); u=r/max(len(r)-1,1)
        key=alpha*u+(1-alpha)*rng.random(len(r)); ids=df['problem_id'].values
        return [ids[i] for i in np.argsort(key,kind='stable')]
    return f
def build(arm,alpha,seed=7):
    stages=[[] for _ in range(6)]
    for L in range(1,9):
        elig,c=CNT[L]; pool=order_blend(alpha,seed+L)(uni[uni['level']==L]); s=0
        for k in elig: stages[k]+=pool[s:s+c[k]]; s+=c[k]
    man={"arm":arm,"construction":f"cliff_backbone_subject_geo_comove_alpha{alpha}","universe_N":N,
         "n_stages":6,"context_per_stage":CTX6,
         "stages":[{"stage_index":k,"n":len(s),"context_len":CTX6[k],"problem_ids":s} for k,s in enumerate(stages)]}
    json.dump(man,open(f"{HERE}/stages_{arm}.json",'w'))
    return stages
def plc(stages):
    return [tuple(sorted((L,sum(1 for x in s if id2lvl[x]==L)) for L in range(1,9))) for s in stages]
ref=plc(stage_ids)
for arm,a in [("cliff_subjgeo_a50",0.5),("cliff_subjgeo_a30",0.3)]:
    st=build(arm,a); tot=sum(len(s) for s in st); uq=len(set(x for s in st for x in s))
    ok = plc(st)==ref
    print(f"[{arm}] tot={tot} uniq={uq} stage크기={[len(s) for s in st]} | 난이도 cliff_subjgeo와 동일: {ok}")
