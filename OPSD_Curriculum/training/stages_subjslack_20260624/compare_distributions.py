#!/usr/bin/env python3
"""
compare_distributions.py — distributional comparison of the three stage
constructions (diff / ours-subjslack / tiered-old) on the identical N=28,771
universe. Produces the statistics in METHOD §10 (Spearman vs diff, stage-move
fraction, per-stage level variance & monotonicity, per-stage subject-composition
TV, within-level subject-deviation T with permutation p, stage x subject
Cramer's V). CPU-only, reads the published manifests.
"""
import json, numpy as np
from scipy.stats import spearmanr, chi2_contingency

SUB = "stages_subjslack_20260624/"
TIE = "stages_tiered_20260622/"

def load(path):
    d = json.load(open(path)); rows = {}
    for st in d["stages"]:
        for it in st["items"]:
            rows[str(it["problem_id"])] = (st["stage_index"], it["subject"], int(it["level"]))
    return rows

diff = load(SUB+"stages_cond2_diff.json")
ours = load(SUB+"stages_cond3_ours_subjslack.json")
tier = load(TIE+"stages_cond3_ours_C2.json")
ids = sorted(set(diff)&set(ours)&set(tier))
print("common universe N =", len(ids))
SUBJECTS = sorted({diff[p][1] for p in ids}); LEVELS = sorted({diff[p][2] for p in ids}); K=5
def arr(m): return np.array([m[p][0] for p in ids])
sd, so, st_ = arr(diff), arr(ours), arr(tier)
lv = np.array([diff[p][2] for p in ids]); sb = np.array([diff[p][1] for p in ids])

print("\n(1) Spearman vs diff / stage-move fraction")
for n,a in [("ours",so),("tiered",st_)]:
    r,p=spearmanr(sd,a); print(f"  {n:7s} rho={r:+.3f} moved={np.mean(sd!=a)*100:.1f}% mad={np.mean(np.abs(sd-a)):.3f}")
print("\n(2) difficulty backbone")
for n,a in [("diff",sd),("ours",so),("tiered",st_)]:
    m=[lv[a==k].mean() for k in range(K)]; v=[lv[a==k].var() for k in range(K)]
    print(f"  {n:7s} mean-lvl={[round(x,2) for x in m]} meanVar={np.mean(v):.3f} minDelta={np.diff(m).min():+.2f}")
def Tstat(a):
    T=0.0
    for L in LEVELS:
        idx=np.where(lv==L)[0]; stg=a[idx]; sub=sb[idx]; nL=len(idx)
        for k in range(K):
            nk=(stg==k).sum()
            for s in SUBJECTS: T+=abs(((stg==k)&(sub==s)).sum()-nk*(sub==s).mean())
    return T
rng=np.random.default_rng(0)
def perm_p(a,nperm=500):
    obs=Tstat(a); null=np.empty(nperm)
    for j in range(nperm):
        perm=a.copy()
        for L in LEVELS:
            idx=np.where(lv==L)[0]; perm[idx]=a[idx][rng.permutation(len(idx))]
        null[j]=Tstat(perm)
    return obs,null.mean(),(np.sum(null>=obs)+1)/(nperm+1)
print("\n(3) within-level subject-deviation T")
for n,a in [("diff",sd),("ours",so),("tiered",st_)]:
    o,nm,p=perm_p(a); print(f"  {n:7s} T={o:.0f} ratio={o/nm:.1f}x p={p:.4f}")
print("\n(4) stage x subject Cramer's V")
for n,a in [("diff",sd),("ours",so),("tiered",st_)]:
    ct=np.array([[((a==k)&(sb==s)).sum() for s in SUBJECTS] for k in range(K)])
    chi2,p,_,_=chi2_contingency(ct); V=np.sqrt(chi2/(ct.sum()*min(K-1,len(SUBJECTS)-1)))
    print(f"  {n:7s} chi2={chi2:.0f} V={V:.3f}")
