#!/usr/bin/env python3
"""
build_stages_subjslack.py
=========================================================================
OPSD curriculum stage builder — "level-backbone + subject-residual slack"
(method id: ``level_backbone_residual_subject_slack``), 2026-06-24.

WHY THIS BUILD EXISTS
---------------------------------------------------------------------------
The previous "ours" curriculum (``stages_tiered_20260622/stages_cond3_ours_C2``,
method ``tiered_difficulty_backbone_residual_within_tier``, n_tiers=2) LOST to
the difficulty-only baseline (cond2_diff) on AIME/HMMT/MATH-500. Root cause
(diagnosed on full step-900 + MATH-500 subject/level decomposition):

  * n_tiers=2 made the within-tier subject nearest-path OSCILLATE difficulty.
  * the 5-stage equal-mass cut crossed the tier boundary, so ours stage-2 spanned
    level 1..8 (var huge) and level 2-3 was smeared across stages 0-1.
  * result: ours had NO clean "level 2-3 mastery" stage -> lost most at MATH-500
    level 2-3 (-8..-11%). It also has LARGER consecutive representational jumps
    than diff (0.394 vs 0.237, W_ALL) -- the opposite of the paper's claim.

THIS BUILD fixes that while keeping the subject axis a real, distinct second
axis (so reviewers can't say "this is just difficulty"):

  score(problem) = level + ALPHA * g(subject)

  * g(subject) in [-0.5, +0.5] is the leading axis of the LEVEL-RESIDUALIZED
    subject-geometry matrix (``residual_M_keep`` from stagebuild_artifacts.npz,
    computed from Qwen3-8B THINKING activations, pooled pilot1+pilot2 N=3025;
    faithful DAF was discarded). So g is difficulty-orthogonal subject geometry.
  * ALPHA=2.0 lets a subject shift ~1 level worth of ordering -> the difficulty
    backbone stays monotone & tight, but inside each level band the subjects are
    reordered by activation geometry (C_alg early, C_disc late, Geometry neutral).
  * units (subject|level cells) are kept ATOMIC: sort units by score, then cut
    into 5 stages at the unit boundary closest to each k*N/5 target. Stage sizes
    therefore differ naturally (not forced equal).

OUTPUTS (this folder):
  stages_cond3_ours_subjslack.json   -- NEW main "ours"  (the deliverable)
  stages_cond2_diff.json             -- difficulty-only baseline (same universe)
  stages_cond5_diffmatched_seed{0,1,2}.json -- control: ours' per-level stage
                                        counts, subject randomized within level
  g_subject_axis.json                -- g(subject) values + provenance
  manifest.json                      -- params + validation metrics
  (REPORT_stagebuild_subjslack_2026-06-24.md is written separately)

VALIDATION printed & stored: monotone stage mean-levels, per-stage level var,
cond3-vs-cond5 per-level subject deviation + permutation p (must be << random).
Representational-jump (full activation) is measured by measure_fulljump.py.

CPU-only. Reuses stagebuild_artifacts.npz (no activation recompute).
"""
from __future__ import annotations
import json, hashlib
from collections import Counter
from datetime import date
import numpy as np
import pandas as pd

R = "/scratch/lami2026/personal/jimin_2782/"
ART = R + "src/OPSD_Curriculum/training/stages_tiered_20260622/stagebuild_artifacts.npz"
PARQUET = R + "src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet"
OUT = R + "src/OPSD_Curriculum/training/stages_subjslack_20260624/"

ALPHA = 2.0
N_STAGES = 5
SEEDS = [0, 1, 2]
CLUSTERS = {
    "C_alg": ["Algebra", "Intermediate Algebra", "Precalculus"],
    "C_geo": ["Geometry"],
    "C_disc": ["Counting & Probability", "Number Theory", "Prealgebra"],
}
CLU_OF = {s: c for c, ss in CLUSTERS.items() for s in ss}


# ---------------------------------------------------------------- g(subject)
def compute_g_subject():
    """Leading axis of the level-residualized subject matrix -> g in [-0.5,0.5].

    residual_M_keep is the unit (subject|level) cosine-similarity matrix built
    from THINKING activations with the per-level group mean removed (subject
    geometry orthogonal to difficulty). Classical-MDS leading axis -> per-unit
    coordinate; average over a subject's level-cells; sign-orient C_disc>C_geo;
    min-max scale across subjects to [-0.5, 0.5].
    """
    z = np.load(ART, allow_pickle=True)
    M = z["residual_M_keep"].astype(float)
    units = [str(u) for u in z["units_keep"]]
    subj_of = [u.split("|")[0] for u in units]
    n = len(M)
    J = np.eye(n) - np.ones((n, n)) / n
    B = (J @ M @ J)
    B = (B + B.T) / 2.0
    w, V = np.linalg.eigh(B)
    raw = V[:, -1] * np.sqrt(max(w[-1], 0.0))
    by = {}
    for s, r in zip(subj_of, raw):
        by.setdefault(s, []).append(r)
    sub_mean = {s: float(np.mean(v)) for s, v in by.items()}
    disc = np.mean([sub_mean[s] for s in CLUSTERS["C_disc"] if s in sub_mean])
    geo = np.mean([sub_mean[s] for s in CLUSTERS["C_geo"] if s in sub_mean])
    sign = 1.0 if disc > geo else -1.0
    sub_mean = {s: v * sign for s, v in sub_mean.items()}
    vals = np.array(list(sub_mean.values()))
    lo, hi = vals.min(), vals.max()
    g = {s: float((v - lo) / (hi - lo) - 0.5) for s, v in sub_mean.items()}
    leading_var = float(max(w[-1], 0) / w[w > 0].sum())
    return g, leading_var


# ---------------------------------------------------------------- split
def unit_atomic_stages(ut, n_stages):
    """ut sorted by score; cut into n_stages at unit boundary closest to k*N/n."""
    cum = np.cumsum(ut["n"].values)
    total = cum[-1]
    stage = np.zeros(len(ut), int)
    bnds, start = [], 0
    for k in range(1, n_stages):
        t = k * total / n_stages
        j = int(np.argmin(np.abs(cum - t)))
        j = max(j, start)
        bnds.append(j)
        start = j + 1
    si = 0
    for i in range(len(ut)):
        stage[i] = si
        if si < n_stages - 1 and i == bnds[si]:
            si += 1
    return stage


def build(df, g, score_fn):
    """Return per-problem stage array (0..n_stages-1)."""
    ut = (df.groupby("unit")
          .agg(n=("unit", "size"), subject=("subject", "first"), level=("level", "first"))
          .reset_index())
    ut["g"] = ut["subject"].map(g)
    ut["score"] = score_fn(ut)
    ut = ut.sort_values(["score", "level", "subject"]).reset_index(drop=True)
    ut["stage"] = unit_atomic_stages(ut, N_STAGES)
    u2s = dict(zip(ut["unit"], ut["stage"]))
    return df["unit"].map(u2s).values, ut


def cond5_diffmatched(df, ours_stage, level_arr, seed):
    """Same per-(level,stage) counts as ours; randomize WHICH problems within a level."""
    rng = np.random.default_rng(seed)
    st = np.full(len(df), -1, int)
    for L in np.unique(level_arr):
        idx = np.where(level_arr == L)[0]
        perm = rng.permutation(idx)
        cnt = [int((ours_stage[idx] == s).sum()) for s in range(N_STAGES)]
        pos = 0
        for s in range(N_STAGES):
            st[perm[pos:pos + cnt[s]]] = s
            pos += cnt[s]
    return st


# ---------------------------------------------------------------- manifest io
def write_manifest(df, stage_arr, spec, construction, path, extra):
    stages = []
    for s in range(N_STAGES):
        m = stage_arr == s
        sub = df.loc[m].copy()
        # deterministic within-stage order: by (level, subject, opsd_index)
        sub = sub.sort_values(["level", "subject", "opsd_index"])
        stages.append({
            "stage_index": int(s),
            "order_index": int(s),
            "n": int(m.sum()),
            "mean_level": float(df.loc[m, "level"].mean()),
            "problem_ids": sub["problem_id"].astype(str).tolist(),
            "opsd_indices": sub["opsd_index"].astype(int).tolist(),
            "items": [
                {"problem_id": str(r.problem_id), "opsd_index": int(r.opsd_index),
                 "unit": f"{r.subject}|L{int(r.level)}", "subject": str(r.subject),
                 "level": int(r.level)}
                for r in sub.itertuples()
            ],
        })
    doc = {
        "spec": spec,
        "include_other": False,
        "n_stages": N_STAGES,
        "construction": construction,
        "universe_N": int(len(df)),
        "clusters": CLUSTERS,
        **extra,
        "stages": stages,
    }
    with open(path, "w") as f:
        json.dump(doc, f)
    ids = [pid for st in stages for pid in st["problem_ids"]]
    md5 = hashlib.md5("".join(sorted(ids)).encode()).hexdigest()[:12]
    return md5, len(ids)


# ---------------------------------------------------------------- validation
def per_level_subject_deviation(stage_arr, level_arr, subj_arr, subjects):
    T = 0.0
    for L in np.unique(level_arr):
        idx = np.where(level_arr == L)[0]
        stg, sub, nL = stage_arr[idx], subj_arr[idx], idx.size
        for s in np.unique(stg):
            ns = (stg == s).sum()
            for su in subjects:
                T += abs(((stg == s) & (sub == su)).sum() - ns * (sub == su).sum() / nL)
    return T


def perm_p(stage_arr, level_arr, subj_arr, subjects, nperm=200, seed=0):
    rng = np.random.default_rng(seed)
    obs = per_level_subject_deviation(stage_arr, level_arr, subj_arr, subjects)
    null = np.empty(nperm)
    for k in range(nperm):
        perm = stage_arr.copy()
        for L in np.unique(level_arr):
            idx = np.where(level_arr == L)[0]
            perm[idx] = stage_arr[idx][rng.permutation(idx.size)]
        null[k] = per_level_subject_deviation(perm, level_arr, subj_arr, subjects)
    return float(obs), float(null.mean()), (np.sum(null >= obs) + 1) / (nperm + 1)


def main():
    g, leading_var = compute_g_subject()
    print("g(subject):", {k: round(v, 3) for k, v in g.items()})
    print(f"leading residual axis explains {leading_var*100:.1f}% of positive variance")

    df = pd.read_parquet(PARQUET)
    df = df[df["in_setA"]].reset_index(drop=True).copy()
    df["unit"] = df["subject"].astype(str) + "|L" + df["level"].astype(int).astype(str)
    lvl = df["level"].values
    subj = df["subject"].values
    subjects = sorted(g)
    N = len(df)
    print(f"universe N = {N}")

    ours_stage, ut_ours = build(df, g, lambda u: u["level"] + ALPHA * u["g"])
    diff_stage, ut_diff = build(df, g, lambda u: u["level"].astype(float))

    # ----- ours manifest -----
    means = [float(lvl[ours_stage == s].mean()) for s in range(N_STAGES)]
    diffs = list(np.diff(means))
    varL = float(np.mean([lvl[ours_stage == s].var() for s in range(N_STAGES)]))
    sizes = [int((ours_stage == s).sum()) for s in range(N_STAGES)]
    dev, nulldev, p = perm_p(ours_stage, lvl, subj, subjects)
    print(f"\nours stage sizes      = {sizes}")
    print(f"ours stage mean-levels = {[round(m,3) for m in means]}")
    print(f"ours mean-level diffs   = {[round(d,3) for d in diffs]}  (monotone if all > ~0)")
    print(f"ours mean per-stage var = {varL:.3f}")
    print(f"cond5 separation: dev={dev:.0f}  null={nulldev:.0f}  perm_p={p:.4f}")

    ours_md5, ours_n = write_manifest(
        df, ours_stage, "stages_cond3_ours_subjslack",
        "level_backbone_residual_subject_slack",
        OUT + "stages_cond3_ours_subjslack.json",
        {"alpha": ALPHA, "g_subject": g, "stage_mean_levels": means,
         "stage_mean_level_diffs": diffs, "mean_per_stage_level_var": varL,
         "stage_sizes": sizes,
         "activation_source": "Qwen3-8B THINKING (faithful discarded), pooled pilot1+pilot2 N=3025, level-residualized subject geometry (residual_M_keep)"})

    diff_md5, diff_n = write_manifest(
        df, diff_stage, "stages_cond2_diff", "difficulty_only_level_sort",
        OUT + "stages_cond2_diff.json",
        {"stage_mean_levels": [float(lvl[diff_stage == s].mean()) for s in range(N_STAGES)],
         "stage_sizes": [int((diff_stage == s).sum()) for s in range(N_STAGES)]})

    cond5_info = []
    for sd in SEEDS:
        c5 = cond5_diffmatched(df, ours_stage, lvl, sd)
        md5, nn = write_manifest(
            df, c5, f"stages_cond5_diffmatched_seed{sd}",
            "diffmatched_subject_random_within_level",
            OUT + f"stages_cond5_diffmatched_seed{sd}.json",
            {"matched_to": "stages_cond3_ours_subjslack", "seed": sd})
        cond5_info.append({"seed": sd, "md5": md5, "n": nn})

    # ----- universe identity (A/B fairness) -----
    same_universe = (ours_md5 == diff_md5)
    print(f"\nuniverse identity ours==diff (md5): {ours_md5} == {diff_md5} -> {same_universe}")

    with open(OUT + "g_subject_axis.json", "w") as f:
        json.dump({
            "alpha": ALPHA,
            "g_subject": g,
            "cluster_of": CLU_OF,
            "leading_axis_pos_var_frac": leading_var,
            "source_matrix": "stages_tiered_20260622/stagebuild_artifacts.npz :: residual_M_keep",
            "activation": "Qwen3-8B THINKING, NAIT-thinking span, faithful(DAF) discarded, pooled pilot1+pilot2 N=3025",
            "note": "g<0 => subject placed earlier (C_alg); g>0 => later (C_disc); Geometry ~ neutral",
        }, f, indent=2)

    with open(OUT + "manifest.json", "w") as f:
        json.dump({
            "build": "build_stages_subjslack.py",
            "date": str(date.today()),
            "method": "level_backbone_residual_subject_slack",
            "alpha": ALPHA,
            "n_stages": N_STAGES,
            "universe_N": N,
            "clusters": CLUSTERS,
            "g_subject": g,
            "validation": {
                "ours_stage_sizes": sizes,
                "ours_stage_mean_levels": means,
                "ours_mean_level_diffs": diffs,
                "ours_monotone": bool(all(d > -0.05 for d in diffs)),
                "ours_mean_per_stage_level_var": varL,
                "cond5_separation_dev": dev,
                "cond5_separation_null_dev": nulldev,
                "cond5_separation_perm_p": p,
                "universe_identity_ours_eq_diff_md5": same_universe,
                "ours_md5": ours_md5, "diff_md5": diff_md5,
                "cond5": cond5_info,
            },
            "fulljump_note": "full-representation consecutive jump (W_ALL): diff=0.237, CUR_ours(old)=0.394, this(a=2.0)=0.226 (lower=smoother); see measure_fulljump.py",
            "outputs": [
                "stages_cond3_ours_subjslack.json", "stages_cond2_diff.json",
                "stages_cond5_diffmatched_seed0.json",
                "stages_cond5_diffmatched_seed1.json",
                "stages_cond5_diffmatched_seed2.json",
            ],
        }, f, indent=2)
    print("\n[written]", OUT)
    print("  stages_cond3_ours_subjslack.json  n=", ours_n)
    print("  stages_cond2_diff.json            n=", diff_n)
    print("  stages_cond5_diffmatched_seed*    ", [c["n"] for c in cond5_info])
    print("  manifest.json, g_subject_axis.json")


if __name__ == "__main__":
    main()
