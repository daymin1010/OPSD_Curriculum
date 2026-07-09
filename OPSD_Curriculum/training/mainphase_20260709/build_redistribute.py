#!/usr/bin/env python3
"""build_redistribute.py — Main-phase curriculum builder (2026-07-09).

subject 축(Eq 5-6) + benchmark-aligned 방향(Eq 7: κ=ℓ−λz, 이산=벤치 과목을 하드로)
+ 난이도-질량 재가중(Eq 9: hard 꼬리 L6-8 k배 복제, 나머지 s배 subsample, 총량 M 고정).

출력: stages_benchsubj_k{K}.json  (K=1,2,3)
  - hard 문제는 problem_ids에 k번 등장(중복) → 트레이너 allow_duplicate_pids=True로 학습에 반영.
  - 총 문제수 ≈ M(고정) → T≈900스텝(컴퓨트 main과 동일).

전부 결정론적(seed 고정). CPU only.
"""
import json, sys
import numpy as np, pandas as pd

R = "/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum"
HERE = f"{R}/training/mainphase_20260709"
PARQUET = f"{R}/training/outputs/join_setA_rows.parquet"
NPZ = f"{R}/reasoning_pivot/activation/analysis/sim_matrices_pooled3025_levsubj.npz"

LAMBDA = 1.0          # subject 섭동 강도 (Eq 7)
DELTA = 0.3           # ε ~ U(-δ, δ)
N_STAGES = 5
HARD = {6, 7, 8}      # hard 꼬리 H (Eq 9)
SUBJ_SIGN = -1.0      # benchmark-aligned: κ = ℓ + SIGN*λz  (SIGN=-1 → 이산 subject를 하드로)
SEED = 7


def subject_z(uni):
    """Eq 5-6: subject prototype similarity → classical MDS leading axis z(s)."""
    d = np.load(NPZ, allow_pickle=True)
    S = d['THINKING_centered_subject_S'].astype(float)
    order = [str(x) for x in d['THINKING_centered_subject_order']]
    present = sorted(uni['subject'].dropna().astype(str).unique())
    si = [order.index(s) for s in present]
    Ssub = S[np.ix_(si, si)]
    Dm = 1.0 - Ssub
    n = len(si)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    z = V[:, -1] * np.sqrt(max(w[-1], 0.0))
    ml = uni.groupby('subject')['level'].mean()
    if np.corrcoef(z, np.array([ml[s] for s in present]))[0, 1] < 0:
        z = -z                              # Eq 6: signed so corr(z, level) > 0
    return dict(zip(present, z))


def build(k: int):
    rows = pd.read_parquet(PARQUET)
    rows['problem_id'] = rows['problem_id'].astype(str)
    uni = rows[rows['in_setA'] == True].copy()
    M = len(uni)
    Z = subject_z(uni)

    nH = int(uni['level'].isin(HARD).sum())
    s = (M - k * nH) / (M - nH)             # Eq 9 normalization (총량 M 고정)
    assert 0 < s <= 1, f"k={k} gives s={s} out of range"

    hard = uni[uni['level'].isin(HARD)]
    nonhard = uni[~uni['level'].isin(HARD)].sample(frac=s, random_state=SEED)
    pool = pd.concat([nonhard] + [hard] * k, ignore_index=True)   # 복제 = 중복 problem_id

    # Eq 7: κ = ℓ + SIGN*λ*z + ε   (SIGN=-1 → benchmark-aligned)
    rng = np.random.default_rng(SEED)
    kappa = (pool['level'].values
             + SUBJ_SIGN * LAMBDA * pool['subject'].astype(str).map(Z).values
             + rng.uniform(-DELTA, DELTA, len(pool)))
    order = np.argsort(kappa, kind='stable')      # Eq 8: rank(κ)
    Np = len(pool)
    sz = Np // N_STAGES
    stages_idx = [order[i * sz:(i + 1) * sz] if i < N_STAGES - 1 else order[i * sz:]
                  for i in range(N_STAGES)]

    pid = pool['problem_id'].values
    stages = []
    for si, idx in enumerate(stages_idx):
        ids = [str(pid[j]) for j in idx]          # 중복 pid 그대로 유지
        stages.append({"stage_index": si, "n": len(ids), "problem_ids": ids})

    man = {
        "arm": f"benchsubj_k{k}",
        "construction": (f"benchmark_aligned_subject(kappa=l{'+' if SUBJ_SIGN>0 else '-'}"
                         f"{LAMBDA}z+eps) + difficulty_emphasis(H={sorted(HARD)},k={k},s={s:.4f})"),
        "universe_N": M, "pool_N": Np, "n_stages": N_STAGES,
        "hard_multiplicity_k": k, "nonhard_subsample_s": round(s, 4),
        "stages": stages,
    }
    out = f"{HERE}/stages_benchsubj_k{k}.json"
    json.dump(man, open(out, 'w'))

    # 요약
    tot = sum(len(st["problem_ids"]) for st in stages)
    dup = tot - len(set(x for st in stages for x in st["problem_ids"]))
    print(f"[k={k}] s={s:.4f} pool_N={Np} (총슬롯 {tot}, 중복 {dup}) "
          f"stage크기 {[len(st['problem_ids']) for st in stages]} → {out}")
    return man


if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1:]] or [1, 2, 3]
    for k in ks:
        build(k)
