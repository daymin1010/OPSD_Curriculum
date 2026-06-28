#!/usr/bin/env python3
"""
subject_grouping_alllayer_pooled.py
===================================
Re-derive the SUBJECT grouping on the ALL-36-layer average pooled THINKING ΔA
(N≈3025 = pilot1 + pilot2), decide K from the data (NOT fixed to 3), compute the
continuous σ ordering, and rebuild the difficulty×subject stage layout (arm ③).

DECISION CONTEXT
  We switch subject geometry from the mid-L11-15 window to the ALL-36-layer
  average (no principled criterion to single out mid layers; all-layer is the
  reviewer-defensible default). This invalidates the old grouping
  G1{Algebra, C&P, NumberTheory, Prealgebra} / G2{Geometry, IntAlg, Precalc} / Other.

REPRESENTATION (must match prior analysis EXACTLY — reuse similarity_analysis prims)
  - per-layer global-mean-centered ΔA (pooled μ), THINKING (`dA_thinking`).
  - group centroid = member mean (36, D).
  - S[g,h] = layer-averaged cosine of centroids (each of 36 layers L2-normed, then
    mean over layers — NOT a flattened 442k cosine). D = 1 − S.

DIFFICULTY AXIS (UNCHANGED, fixed): D1{1,2} D2{3,4} D3{5,6} D4{7,8}.

CLUSTERING
  - PRIMARY: average linkage on precomputed D. ROBUSTNESS: complete linkage.
  - NO Ward/centroid/median on D (only valid for Euclidean). A Ward view is run
    SEPARATELY on the Euclidean distance of flattened per-layer-L2-normed centroids
    and labeled non-primary robustness (expected to differ).
  - k=2..6: memberships, silhouette (precomputed), dendrogram merge-height gaps,
    cophenetic correlation. Recommend K from largest merge-gap + silhouette peak;
    if they disagree, report both and DEFER to human. Singletons allowed.

ORDERING
  - exact-enumerate open Hamiltonian path over 8 subjects (8!/2 = 20160), minimize
    sum of consecutive D. Report optimal + near-optimal + contiguity vs clusters.
  - σ = exact open path over the K cluster centroids (K! tiny).

STAGE LAYOUT (4 × K, snake)
  - difficulty monotone; within each difficulty traverse clusters in σ order,
    reversing σ at each difficulty transition (subject cluster held constant
    across every difficulty boundary).

OUTPUTS (clustering/ dir)
  - REPORT_subject_grouping_alllayer_N3025.md
  - stages_arm3.json, grouping_outputs.json
  - dendro_subjgroup_avg_alllayer.png, dendro_subjgroup_complete_alllayer.png,
    dendro_subjgroup_ward_euclid_robust.png, heatmap_subjS_reordered_alllayer.png

CPU only. Deterministic seed=42.
"""
from __future__ import annotations
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# ── paths ──────────────────────────────────────────────────────────────────
CLUSTERING = Path(__file__).resolve().parent
ANALYSIS = CLUSTERING.parent
ACT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))          # for `import similarity_analysis`
import similarity_analysis as sa            # noqa: E402

PILOT1_DIR = ACT / "outputs" / "pilot" / "shifts"
PILOT2_DIR = ACT / "outputs" / "pilot2" / "shifts"
SAVED_NPZ = ANALYSIS / "sim_matrices_pooled3025_levsubj.npz"

SEED = 42
LAYERS = 36
CANON_SUBJECTS = [
    "Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra",
    "Number Theory", "Other", "Prealgebra", "Precalculus",
]
DIFFICULTY = {  # fixed, unchanged
    "D1": [1, 2], "D2": [3, 4], "D3": [5, 6], "D4": [7, 8],
}
DIFF_ORDER = ["D1", "D2", "D3", "D4"]


# ── small helpers ───────────────────────────────────────────────────────────
def per_layer_center(DAT):
    """global-mean centering per layer (pooled μ). DAT (N,36,D) f16 -> f32 centered."""
    X = DAT.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    return X - mu


def subject_S(DAT_c, md, subjects):
    """recompute 8x8 subject centroid layer-avg-cosine S using sa primitives."""
    idxg = {s: md.index[md["subject"] == s].to_numpy() for s in subjects}
    cents = sa.centroids(DAT_c, idxg)               # subject -> (36,D)
    S = sa.sim_matrix(cents, subjects)              # 8x8 layer-avg cosine
    return S, cents, idxg


def open_path_exact(D, labels):
    """Exact min-cost OPEN Hamiltonian path over all nodes (enumerate perms, /2 by
    fixing direction). Returns sorted list of (cost, path_as_label_tuple)."""
    n = len(labels)
    idx = list(range(n))
    seen = set()
    results = []
    for p in permutations(idx):
        if p[0] > p[-1]:                            # dedup reverse direction
            continue
        key = p
        if key in seen:
            continue
        seen.add(key)
        cost = sum(D[p[i], p[i + 1]] for i in range(n - 1))
        results.append((cost, tuple(labels[i] for i in p), p))
    results.sort(key=lambda x: x[0])
    return results


def contiguous_segments(path_idx, label_of):
    """Given an ordered tuple of node indices and a dict idx->cluster, check whether
    every cluster forms one contiguous run in the path. Returns (bool, runs)."""
    seq = [label_of[i] for i in path_idx]
    runs = []
    for c in seq:
        if not runs or runs[-1][0] != c:
            runs.append([c, 1])
        else:
            runs[-1][1] += 1
    seen = set()
    ok = True
    for c, _ in runs:
        if c in seen:                               # cluster reappears -> not contiguous
            ok = False
        seen.add(c)
    return ok, runs


def cluster_centroids(cents, members_by_cluster):
    """cluster centroid (36,D) = mean of member-subject ΔA centroids (equal subject
    weight). Returns dict cluster -> (36,D)."""
    out = {}
    for c, subs in members_by_cluster.items():
        stk = np.stack([cents[s] for s in subs], axis=0)   # (k,36,D)
        out[c] = stk.mean(axis=0)
    return out


def matrix_corr(A, B):
    """Pearson correlation over off-diagonal upper-triangle entries of two square mats."""
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = A[iu], B[iu]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fmt_mat(M, labels, ndig=3):
    lab = [str(x) for x in labels]
    w = max(10, max(len(l) for l in lab) + 1)
    head = " " * w + "".join(f"{l[:9]:>10}" for l in lab)
    rows = [head]
    for i, l in enumerate(lab):
        rows.append(f"{l:>{w}}" + "".join(f"{M[i,j]:>10.{ndig}f}" for j in range(len(lab))))
    return "\n".join(rows)


# ── core: clustering sweep on a precomputed D ───────────────────────────────
def clustering_sweep(D, labels, method):
    """Hierarchical clustering on condensed precomputed D. Returns dict with linkage,
    cophenetic corr, merge heights/gaps, and per-k (labels, silhouette)."""
    cond = squareform(D, checks=False)
    Z = linkage(cond, method=method)
    coph_corr, _ = cophenet(Z, cond)
    merge_heights = Z[:, 2]                          # ascending
    # gaps between successive merge heights (large gap => natural cut)
    gaps = np.diff(merge_heights)
    per_k = {}
    for k in range(2, 7):
        lab = fcluster(Z, k, criterion="maxclust")
        sil = float("nan")
        if len(set(lab)) >= 2 and len(set(lab)) < len(labels):
            try:
                sil = float(silhouette_score(D, lab, metric="precomputed"))
            except Exception:
                sil = float("nan")
        per_k[k] = {"labels": lab.tolist(), "silhouette": sil}
    return {
        "Z": Z, "cophenetic_corr": float(coph_corr),
        "merge_heights": merge_heights.tolist(),
        "merge_gaps": gaps.tolist(),
        "per_k": per_k,
    }


def k_from_merge_gap(Z):
    """Recommend K = number of clusters implied by the LARGEST merge-height gap.
    For n leaves, cutting between merge m and m+1 (0-indexed ascending) yields
    K = n - (m+1) clusters at that gap. We restrict to K in 2..6."""
    n = Z.shape[0] + 1
    heights = Z[:, 2]
    gaps = np.diff(heights)
    # gap i is between merge i and i+1; cutting there -> clusters = n - (i+1)
    best = None
    for i, g in enumerate(gaps):
        K = n - (i + 1)
        if 2 <= K <= 6:
            if best is None or g > best[1]:
                best = (K, float(g), i)
    return best  # (K, gap, merge_index)


def members_by_cluster(labels_list, subjects):
    out = {}
    for s, c in zip(subjects, labels_list):
        out.setdefault(int(c), []).append(s)
    return out


# ── stage layout (snake) ────────────────────────────────────────────────────
def build_stages(sigma_clusters, members_by_cluster_map):
    """sigma_clusters: list of cluster ids in σ order. Returns ordered stage list."""
    stages = []
    sidx = 0
    for di, dname in enumerate(DIFF_ORDER):
        order = sigma_clusters if di % 2 == 0 else list(reversed(sigma_clusters))
        for c in order:
            subs = members_by_cluster_map[c]
            stages.append({
                "stage_index": sidx,
                "difficulty_cluster": dname,
                "subject_cluster": int(c),
                "subject_members": subs,
                "level_members": DIFFICULTY[dname],
            })
            sidx += 1
    return stages


# ── main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sa.rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    print("[load] pilot1 ...", flush=True)
    _DAF1, DAT1, md1 = sa.load_pilot(PILOT1_DIR, None)
    print("[load] pilot2 ...", flush=True)
    _DAF2, DAT2, md2 = sa.load_pilot(PILOT2_DIR, None)
    del _DAF1, _DAF2

    DAT = np.concatenate([DAT1, DAT2], axis=0)
    md1 = md1.copy(); md1["pilot"] = "pilot1"
    md2 = md2.copy(); md2["pilot"] = "pilot2"
    import pandas as pd
    md = pd.concat([md1, md2], ignore_index=True)
    n1, n2, N = len(md1), len(md2), len(md)
    print(f"[load] pooled N={N} (pilot1={n1}, pilot2={n2}) in {time.time()-t0:.0f}s", flush=True)

    # ── assertions ──
    subj_present = sorted(md["subject"].unique().tolist())
    assert subj_present == CANON_SUBJECTS, \
        f"subject labels mismatch:\n got {subj_present}\n exp {CANON_SUBJECTS}"
    assert N == n1 + n2, f"N mismatch {N} != {n1}+{n2}"
    assert abs(N - 3025) <= 5, f"pooled N {N} not ≈3025"
    print(f"[assert] 8 canonical subjects OK; N={N} ≈ 3025 OK", flush=True)

    # ── recompute pooled centered subject S and assert match with saved npz ──
    DAT_c = per_layer_center(DAT)
    S, cents, idxg = subject_S(DAT_c, md, CANON_SUBJECTS)

    saved = np.load(SAVED_NPZ, allow_pickle=True)
    S_saved = saved["THINKING_centered_subject_S"]
    order_saved = [str(x) for x in saved["THINKING_centered_subject_order"]]
    assert order_saved == CANON_SUBJECTS, f"saved order mismatch {order_saved}"
    max_abs = float(np.max(np.abs(S - S_saved)))
    print(f"[assert] recomputed S vs saved: max|Δ|={max_abs:.2e}", flush=True)
    assert np.allclose(S, S_saved, atol=1e-3), \
        f"recomputed S does not match saved (max|Δ|={max_abs:.2e})"
    print("[assert] pooled subject S matches saved npz (atol=1e-3) OK", flush=True)

    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, None)

    out = {"N": N, "n_pilot1": n1, "n_pilot2": n2,
           "subjects": CANON_SUBJECTS,
           "S": S.tolist(), "D": D.tolist(),
           "S_recompute_max_abs_diff_vs_saved": max_abs,
           "difficulty_axis": DIFFICULTY}

    L = []  # report lines
    L.append("# Subject Grouping — ALL-36-layer pooled THINKING ΔA (N=%d)" % N)
    L.append("")
    L.append("작성: subject_grouping_alllayer_pooled.py / pooled(pilot1+pilot2), THINKING ΔA, CPU, seed=42")
    L.append("")
    L.append("## 0. Setup / assertions")
    L.append(f"- pooled N = **{N}** (pilot1={n1}, pilot2={n2})")
    L.append(f"- subjects (8 canonical, exact strings): {CANON_SUBJECTS}")
    L.append(f"- representation: per-layer pooled-μ-centered ΔA; S[g,h]=layer-avg cosine of "
             f"36-layer L2-normed centroids; D=1−S.")
    L.append(f"- difficulty axis (FIXED, unchanged): {DIFFICULTY}")
    L.append(f"- **consistency check**: recomputed pooled subject S vs saved "
             f"`sim_matrices_pooled3025_levsubj.npz` → max|Δ|={max_abs:.2e} (atol 1e-3) → PASS")
    L.append("")
    L.append("## 1. Subject similarity S (8×8, all-36-layer, centered)")
    L.append("```\n" + fmt_mat(S, CANON_SUBJECTS) + "\n```")
    L.append("\n## 1b. Distance D = 1 − S")
    L.append("```\n" + fmt_mat(D, CANON_SUBJECTS) + "\n```")

    # ── clustering: average (primary), complete (robustness) ──
    L.append("\n## 2. Clustering — K decided from data (NOT fixed to 3)")
    L.append("PRIMARY = average linkage on precomputed D; ROBUSTNESS = complete linkage. "
             "Ward/centroid/median NOT used on D (valid only for Euclidean).")

    sweeps = {}
    for method in ["average", "complete"]:
        sw = clustering_sweep(D, CANON_SUBJECTS, method)
        sweeps[method] = sw
        rec = k_from_merge_gap(sw["Z"])
        sw["recommend_gap"] = rec
        L.append(f"\n### [{method}] linkage")
        L.append(f"- cophenetic correlation = {sw['cophenetic_corr']:+.3f}")
        L.append(f"- merge heights (ascending) = {[round(x,3) for x in sw['merge_heights']]}")
        L.append(f"- merge gaps = {[round(x,3) for x in sw['merge_gaps']]}")
        L.append(f"- largest-gap recommendation: K={rec[0]} (gap={rec[1]:.3f})")
        L.append("\n| K | silhouette(precomputed) | clusters |")
        L.append("|---|---|---|")
        for k in range(2, 7):
            lab = sw["per_k"][k]["labels"]
            mbc = members_by_cluster(lab, CANON_SUBJECTS)
            comp = "; ".join("{%s}" % ", ".join(v) for v in mbc.values())
            L.append(f"| {k} | {sw['per_k'][k]['silhouette']:+.3f} | {comp} |")

    # silhouette peak (primary = average)
    avg = sweeps["average"]
    sil_by_k = {k: avg["per_k"][k]["silhouette"] for k in range(2, 7)}
    sil_peak_k = max(sil_by_k, key=lambda k: (sil_by_k[k] if np.isfinite(sil_by_k[k]) else -9))
    gap_k = avg["recommend_gap"][0]
    L.append(f"\n### Recommended K (PRIMARY=average linkage)")
    L.append(f"- silhouette peak at K={sil_peak_k} (sil={sil_by_k[sil_peak_k]:+.3f})")
    L.append(f"- largest merge-gap implies K={gap_k}")
    if sil_peak_k == gap_k:
        rec_K = sil_peak_k
        L.append(f"- **agree → RECOMMEND K={rec_K}**")
        defer = False
    else:
        rec_K = sil_peak_k  # default to silhouette but flag
        defer = True
        L.append(f"- **DISAGREE (silhouette={sil_peak_k} vs gap={gap_k}) → report both, "
                 f"DEFER final K to human. Using K={rec_K} for downstream artifacts as a "
                 f"provisional default (silhouette peak).**")

    L.append("\n#### side-by-side memberships (k=2,3,4)")
    for k in [2, 3, 4]:
        mbc = members_by_cluster(avg["per_k"][k]["labels"], CANON_SUBJECTS)
        comp = "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc.items()))
        L.append(f"- k={k}: {comp}")

    rec_labels = avg["per_k"][rec_K]["labels"]
    mbc_rec = members_by_cluster(rec_labels, CANON_SUBJECTS)
    # report singletons honestly
    singles = [c for c, v in mbc_rec.items() if len(v) == 1]
    if singles:
        L.append(f"- singleton clusters at K={rec_K}: " +
                 ", ".join("C%d{%s}" % (c, mbc_rec[c][0]) for c in singles) +
                 " (kept, NOT force-merged).")

    out["clustering"] = {
        m: {"cophenetic_corr": sweeps[m]["cophenetic_corr"],
            "merge_heights": sweeps[m]["merge_heights"],
            "merge_gaps": sweeps[m]["merge_gaps"],
            "recommend_gap_K": sweeps[m]["recommend_gap"][0],
            "per_k": {k: sweeps[m]["per_k"][k] for k in range(2, 7)}}
        for m in sweeps
    }
    out["recommended_K"] = int(rec_K)
    out["recommended_K_defer_to_human"] = bool(defer)
    out["recommended_K_silhouette_peak"] = int(sil_peak_k)
    out["recommended_K_merge_gap"] = int(gap_k)
    out["cluster_members_recommended"] = {str(c): v for c, v in mbc_rec.items()}

    # ── dendrograms ──
    for method in ["average", "complete"]:
        Z = sweeps[method]["Z"]
        fig, ax = plt.subplots(figsize=(9, 5))
        dendrogram(Z, labels=CANON_SUBJECTS, leaf_rotation=45, leaf_font_size=9, ax=ax)
        ax.set_title(f"Subject dendrogram — {method} linkage (cosine D, all-36-layer)")
        fig.tight_layout()
        p = CLUSTERING / f"dendro_subjgroup_{method}_alllayer.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        L.append(f"- dendrogram [{method}]: {p.name}")

    # ── Ward on Euclidean of flattened per-layer-L2-normed centroids (NON-PRIMARY) ──
    L.append("\n### [robustness, NON-PRIMARY] Ward on Euclidean of flattened "
             "per-layer-L2-normed centroids")
    L.append("Ward minimizes Euclidean variance; this reintroduces high-norm-layer "
             "dominance (exactly what layer-averaged cosine avoids). Expected to differ "
             "from the primary cosine clustering — shown only for transparency.")
    feat = np.stack([sa.l2norm_rows(cents[s].astype(np.float32)).reshape(-1)
                     for s in CANON_SUBJECTS])          # (8, 36*D) per-layer normed, flattened
    Zw = linkage(feat, method="ward")                   # Euclidean on raw features
    coph_w, _ = cophenet(Zw, __import__("scipy.spatial.distance", fromlist=["pdist"]).pdist(feat))
    L.append(f"- ward cophenetic corr (Euclidean) = {float(coph_w):+.3f}")
    for k in [2, 3, 4]:
        labw = fcluster(Zw, k, criterion="maxclust")
        mbcw = members_by_cluster(labw.tolist(), CANON_SUBJECTS)
        comp = "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbcw.items()))
        L.append(f"- ward k={k}: {comp}")
    figw, axw = plt.subplots(figsize=(9, 5))
    dendrogram(Zw, labels=CANON_SUBJECTS, leaf_rotation=45, leaf_font_size=9, ax=axw)
    axw.set_title("Subject dendrogram — Ward/Euclidean (NON-PRIMARY robustness)")
    figw.tight_layout()
    pw = CLUSTERING / "dendro_subjgroup_ward_euclid_robust.png"
    figw.savefig(pw, dpi=130); plt.close(figw)
    out["ward_euclid_nonprimary"] = {
        "cophenetic_corr": float(coph_w),
        "per_k": {k: members_by_cluster(fcluster(Zw, k, criterion="maxclust").tolist(),
                                        CANON_SUBJECTS) for k in [2, 3, 4]},
    }

    # ── reordered heatmap by primary (average) dendrogram leaf order ──
    Z = sweeps["average"]["Z"]
    dn = dendrogram(Z, labels=CANON_SUBJECTS, no_plot=True)
    leaf_order = dn["ivl"]
    perm = [CANON_SUBJECTS.index(s) for s in leaf_order]
    S_re = S[np.ix_(perm, perm)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S_re, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(leaf_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(leaf_order, fontsize=8)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{S_re[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Subject centroid cosine S (reordered by average-linkage dendrogram)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ph = CLUSTERING / "heatmap_subjS_reordered_alllayer.png"
    fig.savefig(ph, dpi=130); plt.close(fig)
    L.append(f"- reordered S heatmap: {ph.name} (leaf order: {leaf_order})")
    out["dendro_leaf_order_average"] = leaf_order

    # ── 3. continuous open Hamiltonian path over 8 subjects (exact) ──
    L.append("\n## 3. Continuous subject ordering — exact open Hamiltonian path "
             "(min Σ consecutive D, 8!/2=20160)")
    paths = open_path_exact(D, CANON_SUBJECTS)
    best_cost, best_path, best_idx = paths[0]
    L.append(f"- **optimal path** (cost={best_cost:.3f}): {' → '.join(best_path)}")
    L.append("- near-optimal:")
    for c, pth, _ in paths[1:5]:
        L.append(f"    cost={c:.3f}: {' → '.join(pth)}")
    out["subject_path_optimal"] = {"cost": float(best_cost), "path": list(best_path)}
    out["subject_path_near_optimal"] = [
        {"cost": float(c), "path": list(p)} for c, p, _ in paths[1:5]]

    # contiguity of recommended-K clusters along the optimal path
    label_of = {CANON_SUBJECTS.index(s): int(c) for s, c in zip(CANON_SUBJECTS, rec_labels)}
    ok, runs = contiguous_segments(best_idx, label_of)
    runs_str = " | ".join(f"C{c}×{n}" for c, n in runs)
    L.append(f"\n### Contiguity check (recommended K={rec_K} clusters vs optimal path)")
    L.append(f"- path cluster runs: {runs_str}")
    L.append(f"- **clusters contiguous along optimal path? {'YES' if ok else 'NO'}** "
             + ("(clustering & ordering agree)" if ok else "(FLAG: non-contiguous)"))
    out["contiguity_recommended_K"] = {"contiguous": bool(ok),
                                       "runs": [[int(c), int(n)] for c, n in runs]}

    # ── 4. σ ordering of clusters ──
    L.append("\n## 4. σ ordering of the recommended-K clusters")
    clusters_sorted = sorted(mbc_rec.keys())
    ccents = cluster_centroids(cents, {c: mbc_rec[c] for c in clusters_sorted})
    Sc = sa.sim_matrix(ccents, clusters_sorted)
    Dc = 1.0 - Sc
    np.fill_diagonal(Dc, 0.0)
    Dc = np.clip((Dc + Dc.T) / 2.0, 0.0, None)
    L.append("- inter-cluster centroid cosine (analog of old G1-G2):")
    L.append("```\n" + fmt_mat(Sc, [f"C{c}" for c in clusters_sorted]) + "\n```")
    if len(clusters_sorted) >= 2:
        cpaths = open_path_exact(Dc, clusters_sorted)
        sigma = list(cpaths[0][1])
        L.append(f"- **σ (optimal cluster path, cost={cpaths[0][0]:.3f})** = "
                 + " → ".join(f"C{c}" for c in sigma))
    else:
        sigma = clusters_sorted
        L.append(f"- only one cluster; σ = {sigma}")
    out["sigma_cluster_order"] = [int(c) for c in sigma]
    out["inter_cluster_S"] = Sc.tolist()
    out["inter_cluster_order"] = [int(c) for c in clusters_sorted]

    # ── 5. stage layout (4 × K, snake) ──
    L.append("\n## 5. Stage layout (difficulty × subject, snake / σ-reversal)")
    L.append(f"- difficulty axis (fixed): D1{{1,2}} D2{{3,4}} D3{{5,6}} D4{{7,8}}")
    L.append(f"- subject clusters in σ order: " + " → ".join(f"C{c}" for c in sigma))
    L.append(f"- layout rule: difficulty advances monotonically; within each difficulty "
             f"traverse clusters in σ order, REVERSING σ at each difficulty transition "
             f"(subject cluster held constant across every difficulty boundary). "
             f"Total stages = 4 × {rec_K} = {4*rec_K}.")
    stages = build_stages(sigma, {c: mbc_rec[c] for c in clusters_sorted})
    L.append("\n| stage | difficulty | levels | subject_cluster | subject_members |")
    L.append("|---|---|---|---|---|")
    for st in stages:
        L.append(f"| {st['stage_index']} | {st['difficulty_cluster']} | "
                 f"{st['level_members']} | C{st['subject_cluster']} | "
                 f"{', '.join(st['subject_members'])} |")
    # verify boundary subject-continuity
    bound_ok = True
    for i in range(len(DIFF_ORDER) - 1):
        last_of_di = stages[(i + 1) * rec_K - 1]["subject_cluster"]
        first_of_di1 = stages[(i + 1) * rec_K]["subject_cluster"]
        if last_of_di != first_of_di1:
            bound_ok = False
    L.append(f"\n- difficulty-boundary subject-continuity (snake): "
             f"{'OK (subject cluster identical across every D-boundary)' if bound_ok else 'BROKEN'}")
    out["stages"] = stages
    out["n_stages"] = len(stages)

    # stages_arm3.json (harness format: each stage = {level_set, subject_set})
    arm3 = {
        "spec": "arm3_difficulty_x_subject_snake",
        "difficulty_axis": DIFFICULTY,
        "K_subject_clusters": int(rec_K),
        "K_defer_to_human": bool(defer),
        "sigma_cluster_order": [int(c) for c in sigma],
        "subject_clusters": {f"C{c}": mbc_rec[c] for c in clusters_sorted},
        "stages": [
            {"stage_index": st["stage_index"],
             "level_set": st["level_members"],
             "subject_set": st["subject_members"],
             "difficulty_cluster": st["difficulty_cluster"],
             "subject_cluster": f"C{st['subject_cluster']}"}
            for st in stages
        ],
    }
    (CLUSTERING / "stages_arm3.json").write_text(json.dumps(arm3, indent=2), encoding="utf-8")

    # downstream harness note
    L.append("\n### Downstream harness changes")
    L.append(f"- arm ④ (subject-order-shuffled): uses the SAME 4×{rec_K}={4*rec_K} cells; "
             f"only the σ visiting order of subject clusters is shuffled within each difficulty.")
    L.append(f"- arm ⑤ (subject-agnostic random split): must split each difficulty into "
             f"**K={rec_K}** random parts (NOT 3) to match the new granularity.")
    out["arm5_K_random_parts"] = int(rec_K)

    # ── 6. robustness: pilot1-only / pilot2-only / gen_len-balanced ──
    L.append("\n## 6. Robustness (report, NOT a gate)")

    def half_S(mask, name):
        DAT_h = per_layer_center(DAT[mask])
        md_h = md.loc[mask].reset_index(drop=True)
        Sh, _, _ = subject_S(DAT_h, md_h, CANON_SUBJECTS)
        Dh = 1.0 - Sh; np.fill_diagonal(Dh, 0.0); Dh = np.clip((Dh + Dh.T)/2, 0, None)
        swh = clustering_sweep(Dh, CANON_SUBJECTS, "average")
        labh = swh["per_k"][rec_K]["labels"]
        mbch = members_by_cluster(labh, CANON_SUBJECTS)
        corr = matrix_corr(Sh, S)
        return Sh, mbch, corr

    p1mask = (md["pilot"] == "pilot1").to_numpy()
    p2mask = (md["pilot"] == "pilot2").to_numpy()
    Sp1, mbc_p1, corr_p1 = half_S(p1mask, "pilot1")
    Sp2, mbc_p2, corr_p2 = half_S(p2mask, "pilot2")

    def clustering_equal(a, b):
        """compare two cluster partitions (dict cluster->subjects) as set-of-frozensets."""
        sa_ = {frozenset(v) for v in a.values()}
        sb_ = {frozenset(v) for v in b.values()}
        return sa_ == sb_

    L.append(f"### pilot1-only (re-centered, N={int(p1mask.sum())})")
    L.append("- " + "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_p1.items())))
    L.append(f"- S matrix correlation vs pooled = {corr_p1:+.3f}")
    L.append(f"### pilot2-only (re-centered, N={int(p2mask.sum())})")
    L.append("- " + "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_p2.items())))
    L.append(f"- S matrix correlation vs pooled = {corr_p2:+.3f}")
    same_p1 = clustering_equal(mbc_rec, mbc_p1)
    same_p2 = clustering_equal(mbc_rec, mbc_p2)
    L.append(f"- recommended-K membership stable: pilot1={'YES' if same_p1 else 'NO'}, "
             f"pilot2={'YES' if same_p2 else 'NO'}")
    out["robustness_pilot1"] = {"members": {str(c): v for c, v in mbc_p1.items()},
                                "S_corr_vs_pooled": corr_p1, "same_as_pooled": bool(same_p1)}
    out["robustness_pilot2"] = {"members": {str(c): v for c, v in mbc_p2.items()},
                                "S_corr_vs_pooled": corr_p2, "same_as_pooled": bool(same_p2)}

    # gen_len-balanced
    L.append("\n### gen_len-quintile-balanced subsample")
    bidx = sa.genlen_balanced_indices(md, "subject", CANON_SUBJECTS, n_quint=5)
    if bidx is None or len(bidx) < 2 * len(CANON_SUBJECTS):
        L.append("- balanced subsample unavailable.")
        out["robustness_genlen_balanced"] = None
    else:
        DAT_b = per_layer_center(DAT[bidx])
        md_b = md.loc[bidx].reset_index(drop=True)
        Sb, _, _ = subject_S(DAT_b, md_b, CANON_SUBJECTS)
        Db = 1.0 - Sb; np.fill_diagonal(Db, 0.0); Db = np.clip((Db + Db.T)/2, 0, None)
        swb = clustering_sweep(Db, CANON_SUBJECTS, "average")
        labb = swb["per_k"][rec_K]["labels"]
        mbc_b = members_by_cluster(labb, CANON_SUBJECTS)
        corr_b = matrix_corr(Sb, S)
        same_b = clustering_equal(mbc_rec, mbc_b)
        L.append(f"- balanced N={len(bidx)}")
        L.append("- " + "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_b.items())))
        L.append(f"- S matrix correlation vs pooled = {corr_b:+.3f}")
        L.append(f"- recommended-K membership survives length balancing: "
                 f"{'YES' if same_b else 'NO'}")
        L.append("- note: prior subject separability gap retained only ~57% under "
                 "balancing → grouping length-robustness explicitly checked here.")
        out["robustness_genlen_balanced"] = {
            "N": int(len(bidx)),
            "members": {str(c): v for c, v in mbc_b.items()},
            "S_corr_vs_pooled": corr_b, "same_as_pooled": bool(same_b)}

    # ── all-layer vs mid documentation ──
    L.append("\n## 7. All-layer vs mid-L11-15 (documentation of view choice; NOT a "
             "cross-view agreement claim)")
    L.append("- We adopt the ALL-36-layer average as the reviewer-defensible default "
             "(no principled criterion to single out mid layers).")
    L.append("- Documented difference: all-layer isolates Geometry and moves Algebra to "
             "the Intermediate-Algebra/Precalculus/Prealgebra side; mid-L11-15 grouped "
             "Geometry with IntAlg/Precalc and isolated 'Other'.")
    L.append("- This invalidates the OLD grouping "
             "G1{Algebra, Counting & Probability, Number Theory, Prealgebra} / "
             "G2{Geometry, Intermediate Algebra, Precalculus} / Other.")
    L.append("- The recommended grouping above is derived solely from the all-36-layer S; "
             "we do NOT claim membership agreement with the mid-layer view.")

    # ── write outputs ──
    (CLUSTERING / "grouping_outputs.json").write_text(
        json.dumps(out, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
        encoding="utf-8")
    (CLUSTERING / "REPORT_subject_grouping_alllayer_N3025.md").write_text(
        "\n".join(str(x) for x in L), encoding="utf-8")

    print(f"[OK] wrote REPORT_subject_grouping_alllayer_N3025.md", flush=True)
    print(f"[OK] wrote stages_arm3.json ({len(stages)} stages, K={rec_K})", flush=True)
    print(f"[OK] wrote grouping_outputs.json", flush=True)
    print(f"[OK] PNGs: dendro avg/complete/ward + reordered heatmap", flush=True)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
