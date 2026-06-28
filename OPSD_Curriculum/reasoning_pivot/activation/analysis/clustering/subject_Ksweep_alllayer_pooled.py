#!/usr/bin/env python3
"""
subject_Ksweep_alllayer_pooled.py
=================================
K-SELECTION SWEEP for the SUBJECT grouping on all-36-layer pooled THINKING ΔA
(N≈3025), with 29K-label feasibility as the real cap on K, plus σ / headroom /
stage artifacts. Replaces the prior auto-K (silhouette) logic.

WHY (read first)
  The prior grouping auto-picked K=2 (silhouette-best). K=2 is WRONG for the
  curriculum: with 2 clusters there is no non-trivial subject ORDERING — arm ③
  (σ order) and arm ④ (shuffled) collapse to the same single transition (the
  maximal jump). The "ordering by representational proximity helps" contribution
  is vacuous at K=2. We expose the smoothness↔feasibility trade-off across K and
  default to the FINEST FEASIBLE K (finer ⇒ smaller consecutive jumps, more
  homogeneous stages, larger/sharper ③-vs-④ effect). Do NOT pick K by silhouette.

DECISIONS HONORED
  - Hard floor = empty cell (n=0) on the FULL 29K labels. MIN_CELL (default 300)
    is a conservative "ample diversity" threshold only; we ALSO report the 150
    band. OPSD oversamples small cells (fresh rollout per visit), so a finer K
    with no empties but some cells in 150–300 is acceptable via --K (eyeball
    diversity). Default recommendation leans FINER = finest K with zero empties.
  - The recommended K is a FEASIBILITY CEILING (cell-count bound). Budget T is not
    fixed yet; with 4×K stages, per-stage step = T/(4K) shrinks, so the harness may
    finalize K BELOW this ceiling when T is set. Reported explicitly.

REUSE (same representation / primitives as the prior run)
  imports subject_grouping_alllayer_pooled as g  → load_pilot, per_layer_center,
  subject_S, open_path_exact, clustering_sweep, cluster_centroids,
  members_by_cluster, build_stages, matrix_corr, fmt_mat, CANON_SUBJECTS,
  DIFFICULTY, DIFF_ORDER, PILOT1_DIR, PILOT2_DIR, SAVED_NPZ, SEED, and g.sa.
  all-36-layer, per-layer pooled-μ-centered THINKING ΔA; difficulty axis FIXED
  D1{1,2} D2{3,4} D3{5,6} D4{7,8}; average=primary, complete=robustness; NO
  Ward/centroid/median on cosine D; assert recomputed pooled subject S matches
  sim_matrices_pooled3025_levsubj.npz (atol 1e-3).

FEASIBILITY INPUT (NEW)
  29K GPT labels (gpt-4.1-mini): openthoughts_30k_labels_final.parquet
  (29,434 rows; problem_id, subject [8 canonical], level [1..8], ...). Cell
  viability judged HERE, not on the 3025 pilot.

OUTPUTS (clustering/ dir)
  REPORT_subject_Ksweep_alllayer_N3025.md, Ksweep_outputs.json,
  stages_arm3_K{recommended}.json, stages_arm3_K4.json,
  dendro_subjgroup_average_alllayer.png / _complete_alllayer.png (carry over),
  heatmap_subjS_reordered_alllayer.png (carry over),
  ksweep_tradeoff_headroom_mincell.png (NEW line plot).

CPU only. Deterministic seed=42.   usage: [--K INT] [--min-cell 300]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import silhouette_score


CLUSTERING = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTERING))           # for `import subject_grouping_alllayer_pooled`
import subject_grouping_alllayer_pooled as g   # noqa: E402

CANON = g.CANON_SUBJECTS
DIFFICULTY = g.DIFFICULTY
DIFF_ORDER = g.DIFF_ORDER
SEED = g.SEED

LABELS_29K = (CLUSTERING.parent.parent.parent.parent  # …/reasoning_pivot/activation/analysis/clustering -> up to OPSD_Curriculum
              )  # placeholder, set explicitly below
# explicit absolute path (robust):
PARQUET_29K = Path("/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/"
                   "labeling/outputs/openthoughts_30k_labels_final.parquet")

K_RANGE = list(range(2, 9))      # 2..8


# ── linkage label helpers (clustering_sweep only fills per_k for K=2..6) ──────
def labels_K(sw, K):
    """cluster labels for any K from a sweep's linkage Z (handles K=7,8)."""
    return fcluster(sw["Z"], K, criterion="maxclust").tolist()


def sil_K(D, labels):
    """silhouette on precomputed D for a label list; nan when degenerate."""
    nl = len(set(labels))
    if nl < 2 or nl >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        return float("nan")


# ── metric helpers ───────────────────────────────────────────────────────────

def layeravg_cos(a, b):
    """layer-averaged cosine of two (36,D) centroids (each layer L2-normed)."""
    an = g.sa.l2norm_rows(a.astype(np.float32))
    bn = g.sa.l2norm_rows(b.astype(np.float32))
    return float((an * bn).sum(axis=1).mean())


def within_spread(mbc, cents):
    """mean over clusters of (mean member→own-centroid cosine distance). singletons=0."""
    spreads = []
    for c, subs in mbc.items():
        if len(subs) == 1:
            spreads.append(0.0)
            continue
        cc = np.stack([cents[s] for s in subs], axis=0).mean(axis=0)  # (36,D)
        d = [1.0 - layeravg_cos(cents[s], cc) for s in subs]
        spreads.append(float(np.mean(d)))
    return float(np.mean(spreads))


def mean_offdiag(M):
    n = M.shape[0]
    if n < 2:
        return float("nan")
    iu = ~np.eye(n, dtype=bool)
    return float(M[iu].mean())


def max_offdiag(M):
    n = M.shape[0]
    if n < 2:
        return float("nan")
    iu = ~np.eye(n, dtype=bool)
    return float(M[iu].max())


def cluster_distance_matrix(mbc, cents):
    """inter-cluster centroid layer-avg-cosine distance Dc, with sorted cluster ids."""
    clusters_sorted = sorted(mbc.keys())
    ccents = g.cluster_centroids(cents, {c: mbc[c] for c in clusters_sorted})
    Sc = g.sa.sim_matrix(ccents, clusters_sorted)
    Dc = 1.0 - Sc
    np.fill_diagonal(Dc, 0.0)
    Dc = np.clip((Dc + Dc.T) / 2.0, 0.0, None)
    return clusters_sorted, Sc, Dc


def sigma_metrics(mbc, cents):
    """σ exact open path over cluster centroids + per-transition mean/max + headroom."""
    clusters_sorted, Sc, Dc = cluster_distance_matrix(mbc, cents)
    K = len(clusters_sorted)
    out = {"clusters_sorted": clusters_sorted, "Sc": Sc, "Dc": Dc, "K": K}
    if K < 2:
        out.update(sigma_order=clusters_sorted, sigma_cost=0.0, sigma_mean=0.0,
                   sigma_max=0.0, rand_mean=float("nan"), headroom_mean=float("nan"),
                   max_offdiag=float("nan"), consec=[])
        return out
    paths = g.open_path_exact(Dc, clusters_sorted)
    cost, pth, idx = paths[0]
    consec = [float(Dc[idx[i], idx[i + 1]]) for i in range(K - 1)]
    rand_mean = mean_offdiag(Dc)
    sigma_mean = cost / (K - 1)
    out.update(
        sigma_order=[int(c) for c in pth],
        sigma_cost=float(cost),
        sigma_mean=float(sigma_mean),
        sigma_max=float(max(consec)),
        rand_mean=float(rand_mean),
        headroom_mean=float(rand_mean - sigma_mean),
        max_offdiag=max_offdiag(Dc),
        consec=consec,
    )
    return out


def feasibility_cells(mbc, df29):
    """4×K cell counts on the full 29K. returns dict (difficulty, cluster)->count and summary."""
    counts = {}
    for d in DIFF_ORDER:
        lvls = DIFFICULTY[d]
        for c, subs in mbc.items():
            n = int(((df29["level"].isin(lvls)) & (df29["subject"].isin(subs))).sum())
            counts[(d, int(c))] = n
    vals = list(counts.values())
    summary = {
        "min_cell": int(min(vals)),
        "n_empty": int(sum(v == 0 for v in vals)),
        "n_lt_300": int(sum(v < 300 for v in vals)),
        "n_lt_150": int(sum(v < 150 for v in vals)),
        "n_cells": len(vals),
    }
    return counts, summary


def members_str(mbc):
    return "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc.items()))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None,
                    help="override recommended K (human-confirmable). default: finest feasible.")
    ap.add_argument("--min-cell", type=int, default=300,
                    help="conservative 'ample diversity' threshold (NOT the hard floor). default 300.")
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(SEED)
    g.sa.rng = np.random.default_rng(SEED)

    # ── load pooled activations ──
    print("[load] pilot1 ...", flush=True)
    _f1, DAT1, md1 = g.sa.load_pilot(g.PILOT1_DIR, None)
    print("[load] pilot2 ...", flush=True)
    _f2, DAT2, md2 = g.sa.load_pilot(g.PILOT2_DIR, None)
    del _f1, _f2
    DAT = np.concatenate([DAT1, DAT2], axis=0)
    md1 = md1.copy(); md1["pilot"] = "pilot1"
    md2 = md2.copy(); md2["pilot"] = "pilot2"
    md = pd.concat([md1, md2], ignore_index=True)
    n1, n2, N = len(md1), len(md2), len(md)
    print(f"[load] pooled N={N} (p1={n1}, p2={n2}) in {time.time()-t0:.0f}s", flush=True)

    assert sorted(md["subject"].unique().tolist()) == CANON, "subject labels mismatch"
    assert N == n1 + n2 and abs(N - 3025) <= 5, f"pooled N {N} not ≈3025"

    # ── subject S (assert vs saved npz) ──
    DAT_c = g.per_layer_center(DAT)
    S, cents, idxg = g.subject_S(DAT_c, md, CANON)
    saved = np.load(g.SAVED_NPZ, allow_pickle=True)
    S_saved = saved["THINKING_centered_subject_S"]
    assert [str(x) for x in saved["THINKING_centered_subject_order"]] == CANON
    max_abs = float(np.max(np.abs(S - S_saved)))
    assert np.allclose(S, S_saved, atol=1e-3), f"S mismatch max|Δ|={max_abs:.2e}"
    print(f"[assert] pooled subject S matches saved npz (max|Δ|={max_abs:.2e}) OK", flush=True)

    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, None)

    # ── 29K labels ──
    assert PARQUET_29K.exists(), f"29K labels not found: {PARQUET_29K}"
    df29 = pd.read_parquet(PARQUET_29K, columns=["problem_id", "subject", "level"])
    assert sorted(df29["subject"].unique().tolist()) == CANON, "29K subjects mismatch"
    assert int(df29["level"].min()) == 1 and int(df29["level"].max()) == 8
    N29 = len(df29)
    print(f"[load] 29K labels N={N29}", flush=True)

    # ── linkage sweeps ──
    sweep_avg = g.clustering_sweep(D, CANON, "average")
    sweep_comp = g.clustering_sweep(D, CANON, "complete")

    out = {
        "N": N, "n_pilot1": n1, "n_pilot2": n2, "N_29K": int(N29),
        "subjects": CANON, "S": S.tolist(), "D": D.tolist(),
        "S_recompute_max_abs_diff_vs_saved": max_abs,
        "difficulty_axis": DIFFICULTY, "min_cell_threshold": args.min_cell,
        "hard_floor": "empty_cell_n0_on_29K",
    }

    L = []
    L.append(f"# Subject K-selection SWEEP — all-36-layer pooled THINKING ΔA (N={N}) + 29K feasibility")
    L.append("")
    L.append("작성: subject_Ksweep_alllayer_pooled.py / pooled(pilot1+pilot2) THINKING ΔA, CPU, seed=42")
    L.append("")
    L.append("## 0. Why this run")
    L.append("- Prior auto-K picked **K=2** (silhouette-best). K=2 makes subject ORDERING vacuous "
             "(③ σ-order ≡ ④ shuffle: one transition, the maximal jump). We sweep K and default to "
             "the **finest FEASIBLE K** (finer ⇒ smaller jumps, homogeneous stages, sharper ③-vs-④).")
    L.append("- **Hard floor = empty cell (n=0) on full 29K.** MIN_CELL=%d is a conservative "
             "diversity threshold only; 150 band also reported (OPSD oversamples small cells)." % args.min_cell)
    L.append("- **Recommended K = a feasibility CEILING (cell-count bound).** Budget T not fixed; "
             "with 4×K stages, per-stage step=T/(4K) shrinks → harness may finalize K BELOW this ceiling.")
    L.append("")
    L.append("## 0b. Setup / assertions")
    L.append(f"- pooled N={N} (p1={n1}, p2={n2}); 29K labels N={N29}")
    L.append(f"- subjects (8 canonical): {CANON}")
    L.append(f"- representation: per-layer pooled-μ-centered ΔA; S=layer-avg cosine of 36-layer "
             f"L2-normed centroids; D=1−S. difficulty FIXED {DIFFICULTY}.")
    L.append(f"- consistency: recomputed pooled subject S vs saved npz → max|Δ|={max_abs:.2e} "
             f"(atol 1e-3) → PASS")
    L.append(f"- 29K label file: `{PARQUET_29K.name}` (problem_id, subject, level)")
    L.append("")
    L.append("## 1. Subject similarity S (8×8) and distance D")
    L.append("```\n" + g.fmt_mat(S, CANON) + "\n```")
    L.append("```\n" + g.fmt_mat(D, CANON) + "\n```")

    # 29K subject×level crosstab (reference)
    ct = pd.crosstab(df29["subject"], df29["level"]).reindex(index=CANON)
    L.append("\n## 1b. 29K subject × level (reference; binding constraint preview)")
    L.append("```\n" + ct.to_string() + "\n```")
    # D4 per-subject preview
    d4 = df29[df29["level"].isin(DIFFICULTY["D4"])].groupby("subject").size().reindex(CANON, fill_value=0)
    L.append("- D4{7,8} per-subject totals: " + ", ".join(f"{s}={int(d4[s])}" for s in CANON))
    L.append("  → **Prealgebra D4 = 0** (empty): any K that isolates Prealgebra in D4 hits the hard floor.")
    out["ct_29K"] = {s: {int(l): int(ct.loc[s, l]) for l in ct.columns} for s in CANON}

    # ── per-K sweep ──
    L.append("\n## 2. K-sweep (K=2..8 + individual-subject limit)")
    L.append("PRIMARY = average linkage. headroom_mean = mean_offdiag(Dc) − σ_cost/(K−1) "
             "(③ beats ④ per transition; **=0 at K=2 by construction**, grows with K).")

    per_K = {}
    mbc_by_K = {}
    for K in K_RANGE:
        lab = labels_K(sweep_avg, K)
        mbc = g.members_by_cluster(lab, CANON)
        mbc_by_K[K] = mbc
        sm = sigma_metrics(mbc, cents)
        cells, fsum = feasibility_cells(mbc, df29)
        wspread = within_spread(mbc, cents)
        sil = sil_K(D, lab)

        per_K[K] = {
            "members": {int(c): v for c, v in mbc.items()},
            "n_stages": 4 * K,
            "within_spread": wspread,
            "sigma_mean": sm["sigma_mean"], "sigma_max": sm["sigma_max"],
            "rand_mean": sm["rand_mean"], "headroom_mean": sm["headroom_mean"],
            "max_offdiag": sm["max_offdiag"], "sigma_order": sm["sigma_order"],
            "silhouette": sil, "feasibility": fsum,
            "cells": {f"{d}|C{c}": v for (d, c), v in cells.items()},
        }

    # individual-subject limit (each subject its own node) — equals average K=8 (all singletons),
    # but compute on the raw subject D for clarity.
    indiv_mbc = {i: [CANON[i]] for i in range(8)}
    # use subject indices as cluster ids mapped to subject-name members
    indiv_cells, indiv_fsum = feasibility_cells({i: [CANON[i]] for i in range(8)}, df29)
    # σ over the 8 subjects directly (D is subject-subject)
    paths_indiv = g.open_path_exact(D, CANON)
    cost_i, pth_i, idx_i = paths_indiv[0]
    consec_i = [float(D[idx_i[j], idx_i[j + 1]]) for j in range(7)]
    indiv = {
        "n_stages": 32, "within_spread": 0.0,
        "sigma_mean": cost_i / 7, "sigma_max": max(consec_i),
        "rand_mean": mean_offdiag(D),
        "headroom_mean": mean_offdiag(D) - cost_i / 7,
        "max_offdiag": max_offdiag(D),
        "sigma_order": list(pth_i),
        "silhouette": float("nan"), "feasibility": indiv_fsum,
        "path_cost": float(cost_i), "consec": consec_i,
    }
    out["per_K"] = per_K
    out["indiv_subject_limit"] = indiv

    # decision table
    L.append("\n### Decision table (PRIMARY average linkage)")
    L.append("| K | n_stages | within_spread | σ_mean | σ_max | rand_mean | **headroom_mean** | "
             "silhouette | min_29K | #empty | #<300 | #<150 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for K in K_RANGE:
        p = per_K[K]; f = p["feasibility"]
        L.append(f"| {K} | {p['n_stages']} | {p['within_spread']:.3f} | {p['sigma_mean']:.3f} | "
                 f"{p['sigma_max']:.3f} | {p['rand_mean']:.3f} | **{p['headroom_mean']:.3f}** | "
                 f"{p['silhouette']:+.3f} | {f['min_cell']} | {f['n_empty']} | {f['n_lt_300']} | {f['n_lt_150']} |")
    fi = indiv["feasibility"]
    L.append(f"| indiv(8) | 32 | 0.000 | {indiv['sigma_mean']:.3f} | {indiv['sigma_max']:.3f} | "
             f"{indiv['rand_mean']:.3f} | **{indiv['headroom_mean']:.3f}** | n/a | "
             f"{fi['min_cell']} | {fi['n_empty']} | {fi['n_lt_300']} | {fi['n_lt_150']} |")

    # memberships per K
    L.append("\n### Memberships per K (average linkage)")
    for K in K_RANGE:
        L.append(f"- K={K}: {members_str(mbc_by_K[K])}")
    L.append(f"- indiv(8): each subject singleton; "
             f"σ path = {' → '.join(indiv['sigma_order'])} (cost {indiv['path_cost']:.3f})")

    # complete-linkage cross-check memberships
    L.append("\n### Complete-linkage memberships (robustness)")
    for K in K_RANGE:
        mbc_c = g.members_by_cluster(labels_K(sweep_comp, K), CANON)
        L.append(f"- K={K}: {members_str(mbc_c)}")


    # ── recommendation: finest K with zero empty cells (hard floor) ──
    feasible_Ks = [K for K in K_RANGE if per_K[K]["feasibility"]["n_empty"] == 0]
    ceiling_K = max(feasible_Ks) if feasible_Ks else min(K_RANGE)
    rec_K = args.K if args.K is not None else ceiling_K
    defer_note = (f"--K override → K={rec_K}" if args.K is not None
                  else f"finest feasible (empty=0) → ceiling K={ceiling_K}")
    L.append("\n## 3. Recommendation")
    L.append(f"- feasible K (no empty 29K cell): {feasible_Ks if feasible_Ks else 'NONE in 2..8'}")
    L.append(f"- **feasibility CEILING K = {ceiling_K}** (finest K with zero empty cells). "
             f"This is a cell-count ceiling, NOT the final K — budget T may pull K lower.")
    L.append(f"- **recommended K = {rec_K}** ({defer_note}); human-confirmable, `--K` to override.")
    if per_K.get(rec_K, {}).get("feasibility", {}).get("n_lt_300", 0):
        L.append(f"- note: at K={rec_K}, #cells<300={per_K[rec_K]['feasibility']['n_lt_300']}, "
                 f"#cells<150={per_K[rec_K]['feasibility']['n_lt_150']} (diversity eyeball; "
                 f"OPSD oversamples small cells).")
    out["feasible_Ks"] = feasible_Ks
    out["ceiling_K"] = int(ceiling_K)
    out["recommended_K"] = int(rec_K)
    out["recommended_K_is_ceiling_not_final"] = True
    out["recommended_K_source"] = defer_note

    # ── 29K feasibility matrices for each K ──
    L.append("\n## 4. 29K feasibility — 4×K cell counts per K")
    for K in K_RANGE:
        mbc = mbc_by_K[K]
        clusters_sorted = sorted(mbc.keys())
        cells, fsum = feasibility_cells(mbc, df29)
        L.append(f"\n### K={K}  (min={fsum['min_cell']}, empty={fsum['n_empty']}, "
                 f"<300={fsum['n_lt_300']}, <150={fsum['n_lt_150']})")
        header = "| difficulty | " + " | ".join(f"C{c}" for c in clusters_sorted) + " |"
        L.append(header)
        L.append("|" + "---|" * (len(clusters_sorted) + 1))
        for d in DIFF_ORDER:
            row = f"| {d}{DIFFICULTY[d]} | " + " | ".join(
                str(cells[(d, c)]) for c in clusters_sorted) + " |"
            L.append(row)
        # cluster legend
        L.append("- clusters: " + members_str(mbc))
        # flag first empty cells
        empties = [(d, c) for (d, c) in cells if cells[(d, c)] == 0]
        if empties:
            L.append("- EMPTY cells: " + ", ".join(f"{d}|C{c}{{{', '.join(mbc[c])}}}" for d, c in empties))

    # ── Geometry handling (two options at recommended K) ──
    L.append("\n## 5. Geometry handling at recommended K (report both; human decides)")
    L.append("Geometry is ~orthogonal to all subjects (nearest distance ≈0.95), forcing the largest "
             "σ-transition at any K; its path position is weakly determined.")
    geo = "Geometry"
    rec_mbc = mbc_by_K.get(rec_K, mbc_by_K[ceiling_K])
    # which cluster currently holds Geometry
    geo_cluster = next(c for c, subs in rec_mbc.items() if geo in subs)

    def metrics_block(mbc, tag):
        sm = sigma_metrics(mbc, cents)
        cells, fsum = feasibility_cells(mbc, df29)
        L.append(f"\n### {tag}")
        L.append(f"- clusters: {members_str(mbc)}")
        L.append(f"- σ order: {' → '.join('C%d'%c for c in sm['sigma_order'])}  "
                 f"(σ_mean={sm['sigma_mean']:.3f}, σ_max={sm['sigma_max']:.3f}, "
                 f"headroom_mean={sm['headroom_mean']:.3f})")
        L.append(f"- 29K feasibility: min={fsum['min_cell']}, empty={fsum['n_empty']}, "
                 f"<300={fsum['n_lt_300']}, <150={fsum['n_lt_150']}")
        return {"members": {int(c): v for c, v in mbc.items()}, "sigma": sm["sigma_order"],
                "sigma_mean": sm["sigma_mean"], "sigma_max": sm["sigma_max"],
                "headroom_mean": sm["headroom_mean"], "feasibility": fsum}

    # (a) Geometry as its own singleton cluster
    mbc_a = {c: [s for s in subs if s != geo] for c, subs in rec_mbc.items()}
    mbc_a = {c: subs for c, subs in mbc_a.items() if subs}      # drop empty
    new_id = (max(rec_mbc.keys()) + 1)
    mbc_a[new_id] = [geo]
    out_a = metrics_block(mbc_a, "(a) Geometry as its own singleton cluster")

    # (b) Geometry merged into nearest path-neighbor cluster (by cluster-centroid distance)
    others = {c: subs for c, subs in rec_mbc.items() if c != geo_cluster or len(subs) > 1}
    # build a membership with Geometry removed, then attach to nearest cluster centroid
    base = {c: [s for s in subs if s != geo] for c, subs in rec_mbc.items()}
    base = {c: subs for c, subs in base.items() if subs}
    # nearest cluster to Geometry
    if len(base) >= 1:
        bcl = sorted(base.keys())
        bcents = g.cluster_centroids(cents, {c: base[c] for c in bcl})
        dists = {c: 1.0 - layeravg_cos(cents[geo], bcents[c]) for c in bcl}
        nearest = min(dists, key=dists.get)
        mbc_b = {c: list(subs) for c, subs in base.items()}
        mbc_b[nearest] = mbc_b[nearest] + [geo]
        out_b = metrics_block(mbc_b, f"(b) Geometry merged into nearest cluster C{nearest} "
                                     f"(dist={dists[nearest]:.3f})")
    else:
        out_b = None
    out["geometry_option_a_singleton"] = out_a
    out["geometry_option_b_merged"] = out_b

    # ── stage artifacts for recommended K and K=4 ──
    def emit_stages(K, fname):
        mbc = mbc_by_K[K]
        sm = sigma_metrics(mbc, cents)
        sigma = sm["sigma_order"]
        clusters_sorted = sorted(mbc.keys())
        stages = g.build_stages(sigma, {c: mbc[c] for c in clusters_sorted})
        arm = {
            "spec": "arm3_difficulty_x_subject_snake",
            "K_subject_clusters": int(K),
            "K_is_feasibility_ceiling_not_final": True,
            "difficulty_axis": DIFFICULTY,
            "sigma_cluster_order": [int(c) for c in sigma],
            "subject_clusters": {f"C{c}": mbc[c] for c in clusters_sorted},
            "inter_cluster_S": sm["Sc"].tolist(),
            "feasibility_29K": feasibility_cells(mbc, df29)[1],
            "arm5_K_random_parts": int(K),
            "stages": [
                {"stage_index": st["stage_index"], "level_set": st["level_members"],
                 "subject_set": st["subject_members"],
                 "difficulty_cluster": st["difficulty_cluster"],
                 "subject_cluster": f"C{st['subject_cluster']}"}
                for st in stages
            ],
        }
        (CLUSTERING / fname).write_text(json.dumps(arm, indent=2), encoding="utf-8")
        return stages, sm, mbc

    L.append("\n## 6. Stage layout artifacts (snake / σ-reversal; generalizes to any K)")
    L.append("Rule: difficulty monotone; within each difficulty traverse clusters in σ order, "
             "reversing σ at each difficulty transition (subject cluster held constant across "
             "every difficulty boundary). Total stages = 4×K. arm⑤ random parts per difficulty = K.")
    for K, fname in [(rec_K, f"stages_arm3_K{rec_K}.json"), (4, "stages_arm3_K4.json")]:
        stages, sm, mbc = emit_stages(K, fname)
        clusters_sorted = sorted(mbc.keys())
        L.append(f"\n### K={K} → `{fname}`  (σ = {' → '.join('C%d'%c for c in sm['sigma_order'])})")
        L.append("- inter-cluster cosine S:")
        L.append("```\n" + g.fmt_mat(sm["Sc"], [f"C{c}" for c in clusters_sorted]) + "\n```")
        L.append("| stage | difficulty | levels | subject_cluster | subject_members |")
        L.append("|---|---|---|---|---|")
        for st in stages:
            L.append(f"| {st['stage_index']} | {st['difficulty_cluster']} | {st['level_members']} | "
                     f"C{st['subject_cluster']} | {', '.join(st['subject_members'])} |")
        # boundary continuity check
        bound_ok = all(stages[(i + 1) * K - 1]["subject_cluster"] == stages[(i + 1) * K]["subject_cluster"]
                       for i in range(len(DIFF_ORDER) - 1))
        L.append(f"- difficulty-boundary subject-continuity: {'OK' if bound_ok else 'BROKEN'}")

    # ── robustness for recommended K and K=4 ──
    L.append("\n## 7. Robustness (report, NOT a gate) — recommended K and K=4")
    L.append("Expectation: S-matrix correlation stays HIGH (continuous geometry reliable) even where "
             "hard-cut membership wobbles (Geometry/Algebra/Prealgebra). Membership wobble is NOT a "
             "failure — it is the argument for finer/continuous placement.")

    def partition_set(mbc):
        return {frozenset(v) for v in mbc.values()}

    def robustness_for_K(K):
        res = {}
        for tag, mask in [("pilot1", (md["pilot"] == "pilot1").to_numpy()),
                          ("pilot2", (md["pilot"] == "pilot2").to_numpy())]:
            DAT_h = g.per_layer_center(DAT[mask])
            md_h = md.loc[mask].reset_index(drop=True)
            Sh, _, _ = g.subject_S(DAT_h, md_h, CANON)
            Dh = 1.0 - Sh; np.fill_diagonal(Dh, 0.0); Dh = np.clip((Dh + Dh.T) / 2, 0, None)
            sw = g.clustering_sweep(Dh, CANON, "average")
            mbc_h = g.members_by_cluster(labels_K(sw, K), CANON)

            res[tag] = {"members": {int(c): v for c, v in mbc_h.items()},
                        "S_corr_vs_pooled": g.matrix_corr(Sh, S),
                        "same": partition_set(mbc_h) == partition_set(mbc_by_K[K])}
        # gen_len-balanced
        bidx = g.sa.genlen_balanced_indices(md, "subject", CANON, n_quint=5)
        if bidx is not None and len(bidx) >= 2 * len(CANON):
            DAT_b = g.per_layer_center(DAT[bidx])
            md_b = md.loc[bidx].reset_index(drop=True)
            Sb, _, _ = g.subject_S(DAT_b, md_b, CANON)
            Db = 1.0 - Sb; np.fill_diagonal(Db, 0.0); Db = np.clip((Db + Db.T) / 2, 0, None)
            sw = g.clustering_sweep(Db, CANON, "average")
            mbc_b = g.members_by_cluster(labels_K(sw, K), CANON)

            res["balanced"] = {"N": int(len(bidx)),
                               "members": {int(c): v for c, v in mbc_b.items()},
                               "S_corr_vs_pooled": g.matrix_corr(Sb, S),
                               "same": partition_set(mbc_b) == partition_set(mbc_by_K[K])}
        else:
            res["balanced"] = None
        return res

    rob = {}
    for K in sorted({rec_K, 4}):
        r = robustness_for_K(K)
        rob[K] = r
        L.append(f"\n### K={K}")
        for tag in ["pilot1", "pilot2", "balanced"]:
            v = r.get(tag)
            if v is None:
                L.append(f"- {tag}: unavailable"); continue
            extra = f", N={v['N']}" if "N" in v else ""
            L.append(f"- {tag}{extra}: S_corr_vs_pooled={v['S_corr_vs_pooled']:+.3f}, "
                     f"membership_same={'YES' if v['same'] else 'NO'} → {members_str({int(c): m for c, m in v['members'].items()})}")
    out["robustness"] = {str(K): rob[K] for K in rob}

    # ── all-layer vs mid note (carry over) ──
    L.append("\n## 8. All-layer vs mid-L11-15 (view-choice documentation; carried over)")
    L.append("- ALL-36-layer average adopted as reviewer-defensible default (no principled criterion "
             "to single out mid layers). Invalidates old G1/G2/Other grouping. Grouping derived solely "
             "from all-36-layer S; no cross-view membership-agreement claim.")

    # ── dendrograms + reordered heatmap (carry over) ──
    for method, sw in [("average", sweep_avg), ("complete", sweep_comp)]:
        fig, ax = plt.subplots(figsize=(9, 5))
        dendrogram(sw["Z"], labels=CANON, leaf_rotation=45, leaf_font_size=9, ax=ax)
        ax.set_title(f"Subject dendrogram — {method} linkage (cosine D, all-36-layer)")
        fig.tight_layout()
        fig.savefig(CLUSTERING / f"dendro_subjgroup_{method}_alllayer.png", dpi=130)
        plt.close(fig)

    dn = dendrogram(sweep_avg["Z"], labels=CANON, no_plot=True)
    leaf = dn["ivl"]; perm = [CANON.index(s) for s in leaf]
    S_re = S[np.ix_(perm, perm)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S_re, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(leaf, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(leaf, fontsize=8)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{S_re[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Subject centroid cosine S (reordered by average-linkage dendrogram)")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(CLUSTERING / "heatmap_subjS_reordered_alllayer.png", dpi=130); plt.close(fig)

    # ── trade-off line plot: headroom_mean & min_29K_cell vs K ──
    Ks = K_RANGE + [8.5]          # 8.5 placeholder x for indiv
    head = [per_K[K]["headroom_mean"] for K in K_RANGE] + [indiv["headroom_mean"]]
    minc = [per_K[K]["feasibility"]["min_cell"] for K in K_RANGE] + [indiv["feasibility"]["min_cell"]]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = K_RANGE + [9]
    ax1.plot(xs, head, "o-", color="tab:blue", label="headroom_mean (③ vs ④)")
    ax1.set_xlabel("K (9 = individual-subject limit)")
    ax1.set_ylabel("headroom_mean", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0, color="gray", lw=0.6, ls=":")
    ax2 = ax1.twinx()
    ax2.plot(xs, minc, "s--", color="tab:red", label="min 29K cell")
    ax2.set_ylabel("min 29K cell count", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.axhline(args.min_cell, color="tab:red", lw=0.6, ls=":")
    ax2.axhline(0, color="black", lw=0.8)
    ax1.axvline(ceiling_K, color="green", lw=1.0, ls="--")
    ax1.set_title(f"Smoothness↔feasibility trade-off (ceiling K={ceiling_K}, MIN_CELL={args.min_cell})")
    fig.tight_layout()
    fig.savefig(CLUSTERING / "ksweep_tradeoff_headroom_mincell.png", dpi=130); plt.close(fig)

    # ── write outputs ──
    (CLUSTERING / "Ksweep_outputs.json").write_text(
        json.dumps(out, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
        encoding="utf-8")
    (CLUSTERING / "REPORT_subject_Ksweep_alllayer_N3025.md").write_text(
        "\n".join(str(x) for x in L), encoding="utf-8")

    print(f"[OK] recommended K={rec_K} (ceiling={ceiling_K}, feasible={feasible_Ks})", flush=True)
    print(f"[OK] wrote REPORT_subject_Ksweep_alllayer_N3025.md", flush=True)
    print(f"[OK] wrote stages_arm3_K{rec_K}.json, stages_arm3_K4.json", flush=True)
    print(f"[OK] wrote Ksweep_outputs.json + 4 PNGs", flush=True)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
