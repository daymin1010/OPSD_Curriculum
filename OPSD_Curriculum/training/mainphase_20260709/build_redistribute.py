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


def build(k: int, sign: float = SUBJ_SIGN, subset: int = 0, shuffle: bool = False, maxlevel: int = 0, lam: float = LAMBDA):
    """subset>0: 유니버스를 level×subject 층화로 subset개로 축소 후 동일 파이프라인.
    (데이터 스케일링 스윕용: T=subset/32로 스텝도 비례 축소됨을 유의.)
    shuffle=True: 완전 랜덤 대조(κ=uniform, level·subject 무시). 4B main_shuffle과 동일 구성. k=1 권장.
    maxlevel>0: 레벨 ≤ maxlevel 문제만 사용(easy-only 커리큘럼; 교수님 피드백 — 4B rationalization
    한계 가설. maxlevel=5면 H={6,7,8}가 공집합 → k 무의미, k=1 권장)."""
    tag = "shuffle" if shuffle else ("diffonly" if lam == 0 else ("benchsubj" if sign < 0 else "contsubj"))   # sign<0 = discrete-late(벤치정렬), >0 = continuous-late(old main); shuffle = 완전 랜덤; lam=0 = 난이도-only(cliff형)
    rows = pd.read_parquet(PARQUET)
    rows['problem_id'] = rows['problem_id'].astype(str)
    uni = rows[rows['in_setA'] == True].copy()
    if maxlevel:
        uni = uni[uni['level'] <= maxlevel].copy()
        tag = f"{tag}_L{maxlevel}"
    if subset and subset < len(uni):
        frac = subset / len(uni)
        uni = (uni.groupby(['level', 'subject'], group_keys=False)
                  .apply(lambda g: g.sample(frac=frac, random_state=SEED)))
        tag = f"{tag}_n{subset//1000}k"
    M = len(uni)
    Z = subject_z(uni)

    nH = int(uni['level'].isin(HARD).sum())
    s = (M - k * nH) / (M - nH)             # Eq 9 normalization (총량 M 고정)
    assert 0 < s <= 1, f"k={k} gives s={s} out of range"

    hard = uni[uni['level'].isin(HARD)]
    nonhard = uni[~uni['level'].isin(HARD)].sample(frac=s, random_state=SEED)
    pool = pd.concat([nonhard] + [hard] * k, ignore_index=True)   # 복제 = 중복 problem_id

    # Eq 7: κ = ℓ + sign*λ*z + ε   (sign=-1 → benchmark-aligned/discrete-late)
    rng = np.random.default_rng(SEED)
    if shuffle:
        # 완전 랜덤 대조군: level·subject 모두 무시 → 각 스테이지가 전체 분포의 균일 표본
        # (4B main_shuffle과 동일 구성: easy→hard 진행 없음, 스테이지별 lvl_mean≈전체평균). k=1 권장.
        kappa = rng.uniform(0.0, 1.0, len(pool))
    else:
        kappa = (pool['level'].values
                 + sign * lam * pool['subject'].astype(str).map(Z).values
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

    arm_name = tag if shuffle else f"{tag}_k{k}"   # shuffle도 maxlevel/subset 접미사(tag) 유지
    construction = ("random_shuffle(kappa=uniform; level·subject 모두 무시, 완전 랜덤 대조 = 4B main_shuffle 재현)"
                    if shuffle else
                    (f"{'discrete_late(benchmark_aligned)' if sign<0 else 'continuous_late(old_main)'}"
                     f"(kappa=l{'+' if sign>0 else '-'}{LAMBDA}z+eps) "
                     f"+ difficulty_emphasis(H={sorted(HARD)},k={k},s={s:.4f})"))
    man = {
        "arm": arm_name,
        "construction": construction,
        "universe_N": M, "pool_N": Np, "n_stages": N_STAGES,
        "subj_sign": sign, "hard_multiplicity_k": k, "nonhard_subsample_s": round(s, 4),
        "shuffle": shuffle,
        "stages": stages,
    }
    out = f"{HERE}/stages_{arm_name}.json"
    json.dump(man, open(out, 'w'))

    # 요약
    tot = sum(len(st["problem_ids"]) for st in stages)
    dup = tot - len(set(x for st in stages for x in st["problem_ids"]))
    print(f"[{arm_name}] s={s:.4f} pool_N={Np} (총슬롯 {tot}, 중복 {dup}) "
          f"stage크기 {[len(st['problem_ids']) for st in stages]} → {out}")
    return man


if __name__ == "__main__":
    # usage: build_redistribute.py [cont|bench] [subset=N] <k...>   (기본 bench, 기본 k=1 2 3)
    #   ex) build_redistribute.py cont 2               → stages_contsubj_k2.json
    #   ex) build_redistribute.py subset=15000 2       → stages_benchsubj_n15k_k2.json
    #   ex) build_redistribute.py shuffle             → stages_shuffle.json (완전 랜덤 대조, k=1)
    #   ex) build_redistribute.py maxlevel=5 1        → stages_benchsubj_L5_k1.json (easy-only)
    #   ex) build_redistribute.py maxlevel=5 lambda=0 1  → stages_diffonly_L5_k1.json (난이도-only, cliff형)
    sign = SUBJ_SIGN
    subset = 0
    shuffle = False
    maxlevel = 0
    lam = LAMBDA
    ks = []
    for a in sys.argv[1:]:
        if a in ("cont", "contsubj"): sign = +1.0
        elif a in ("bench", "benchsubj"): sign = -1.0
        elif a in ("shuffle", "shuf"): shuffle = True
        elif a.startswith("subset="): subset = int(a.split("=")[1])
        elif a.startswith("maxlevel="): maxlevel = int(a.split("=")[1])
        elif a.startswith("lambda="): lam = float(a.split("=")[1])
        else: ks.append(int(a))
    if shuffle:
        build(1, sign, subset, shuffle=True, maxlevel=maxlevel)   # 완전 랜덤 대조 (k=1)
    else:
        for k in (ks or [1, 2, 3]):
            build(k, sign, subset, maxlevel=maxlevel, lam=lam)
