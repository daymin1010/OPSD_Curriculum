#!/usr/bin/env python3
"""
subject_grouping_excludeOther_pooled.py
=======================================
Arm ③-main (FINAL MAIN) = EXCLUDE 'Other'. Cluster the 7 real subjects into the
4 principled clusters (3 tight pairs + Geometry), build the σ ordering, the
16-stage snake layout, and 28,771-row (29K minus Other) feasibility.

ONLY produces the EXCLUDE-Other artifacts. The with-Other K=5 variant
(stages_arm3_K5.json) is NOT regenerated here.

REPRESENTATION (identical to subject_grouping/Ksweep): all-36-layer, per-layer
pooled-μ-centered pooled THINKING ΔA. S[g,h]=layer-avg cosine of 36-layer
L2-normed centroids; D=1−S. Difficulty axis FIXED D1{1,2} D2{3,4} D3{5,6} D4{7,8}.
PRIMARY = average linkage on cosine D; ROBUSTNESS = complete linkage.
NO Ward/centroid/median on the cosine distance.

CPU only. Deterministic seed=42.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

CLUSTERING = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTERING))
import subject_grouping_alllayer_pooled as g          # noqa: E402  (prims)
import subject_Ksweep_alllayer_pooled as k            # noqa: E402  (sigma_metrics, feasibility_cells)

SEED = g.SEED
CANON = g.CANON_SUBJECTS                                # 8 canonical
SUBJ7 = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra",
         "Number Theory", "Prealgebra", "Precalculus"]  # drop 'Other'
DIFFICULTY = g.DIFFICULTY
DIFF_ORDER = g.DIFF_ORDER
K_EXCLUDE = 4
PARQUET_29K = k.PARQUET_29K

# expected (target) partition for the 7 subjects at K=4 (label ids irrelevant)
TARGET_PARTITION = {
    frozenset({"Intermediate Algebra", "Precalculus"}),
    frozenset({"Counting & Probability", "Number Theory"}),
    frozenset({"Algebra", "Prealgebra"}),
    frozenset({"Geometry"}),
}


def partition_set(mbc):
    return {frozenset(v) for v in mbc.values()}


def main():
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
    assert N == n1 + n2 and abs(N - 3025) <= 5, f"pooled N={N} unexpected"

    # ── pooled 8×8 S, assert vs saved npz ──
    DAT_c = g.per_layer_center(DAT)
    S8, cents8, _ = g.subject_S(DAT_c, md, CANON)
    saved = np.load(g.SAVED_NPZ, allow_pickle=True)
    S_saved = saved["THINKING_centered_subject_S"]
    order_saved = [str(x) for x in saved["THINKING_centered_subject_order"]]
    assert order_saved == CANON, f"saved order mismatch {order_saved}"
    max_abs8 = float(np.max(np.abs(S8 - S_saved)))
    assert np.allclose(S8, S_saved, atol=1e-3), f"8×8 S vs saved max|Δ|={max_abs8:.2e}"
    print(f"[assert] pooled 8×8 S matches saved npz (max|Δ|={max_abs8:.2e}) OK", flush=True)

    # ── 7×7 S = 8×8 sub-block (slice), and independent recompute over 7 subjects ──
    ix7 = [CANON.index(s) for s in SUBJ7]
    S7_slice = S8[np.ix_(ix7, ix7)]
    S7_recomp, cents7, idxg7 = g.subject_S(DAT_c, md, SUBJ7)
    max_abs7 = float(np.max(np.abs(S7_slice - S7_recomp)))
    assert np.allclose(S7_slice, S7_recomp, atol=1e-6), \
        f"7×7 S slice != recompute (max|Δ|={max_abs7:.2e})"
    print(f"[assert] 7×7 S == 8×8 sub-block (max|Δ|={max_abs7:.2e}, atol=1e-6) OK", flush=True)
    S7 = S7_recomp
    D7 = 1.0 - S7
    np.fill_diagonal(D7, 0.0)
    D7 = np.clip((D7 + D7.T) / 2.0, 0.0, None)

    # ── clustering K=4 (average primary) ──
    sw_avg = g.clustering_sweep(D7, SUBJ7, "average")
    lab4 = fcluster(sw_avg["Z"], K_EXCLUDE, criterion="maxclust").tolist()
    mbc = g.members_by_cluster(lab4, SUBJ7)
    got = partition_set(mbc)
    partition_ok = (got == TARGET_PARTITION)
    print(f"[assert] K=4 partition matches target? {partition_ok}", flush=True)

    # complete-linkage robustness K=4
    sw_comp = g.clustering_sweep(D7, SUBJ7, "complete")
    lab4_c = fcluster(sw_comp["Z"], K_EXCLUDE, criterion="maxclust").tolist()
    mbc_c = g.members_by_cluster(lab4_c, SUBJ7)

    L = []
    L.append(f"# Subject Grouping — EXCLUDE 'Other' (arm ③-main, FINAL MAIN), all-36-layer pooled THINKING ΔA (N={N})")
    L.append("")
    L.append("작성: subject_grouping_excludeOther_pooled.py / pooled(pilot1+pilot2), THINKING ΔA, CPU, seed=42")
    L.append("")
    L.append("## 0. Setup / assertions")
    L.append(f"- pooled N = **{N}** (pilot1={n1}, pilot2={n2})")
    L.append(f"- 7 subjects (drop 'Other'): {SUBJ7}")
    L.append(f"- representation: per-layer pooled-μ-centered ΔA; S=layer-avg cosine of 36-layer L2-normed centroids; D=1−S.")
    L.append(f"- difficulty axis (FIXED): {DIFFICULTY}")
    L.append(f"- 8×8 pooled S vs saved npz: max|Δ|={max_abs8:.2e} (atol 1e-3) → PASS")
    L.append(f"- **7×7 S == 8×8 sub-block** (slice vs recompute): max|Δ|={max_abs7:.2e} (atol 1e-6) → PASS")

    L.append("\n## 1. 7×7 subject similarity S (exclude Other)")
    L.append("```\n" + g.fmt_mat(S7, SUBJ7) + "\n```")
    L.append("\n### Distance D = 1 − S")
    L.append("```\n" + g.fmt_mat(D7, SUBJ7) + "\n```")

    L.append("\n## 2. Clustering — K=4 (average linkage primary; complete robustness)")
    L.append(f"- average cophenetic corr = {sw_avg['cophenetic_corr']:+.3f}; "
             f"complete cophenetic corr = {sw_comp['cophenetic_corr']:+.3f}")
    L.append("- **K=4 partition (average linkage):** " +
             "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc.items())))
    L.append("- K=4 partition (complete linkage, robustness): " +
             "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_c.items())))
    L.append(f"- **target partition match (average): {'YES' if partition_ok else 'NO'}** "
             f"(target = {{IntAlg,Precalc}},{{C&P,NumberTheory}},{{Algebra,Prealgebra}},{{Geometry}})")
    L.append("- rationale: K=5 would split the +0.666 {IntAlg,Precalc} pair (wrong); "
             "K=3 remakes an incoherent bucket. K=4 is the principled granularity for 7 subjects.")

    out = {
        "spec": "arm3_main_exclude_Other",
        "N": N, "n_pilot1": n1, "n_pilot2": n2,
        "subjects7": SUBJ7, "K": K_EXCLUDE,
        "S7": S7.tolist(), "D7": D7.tolist(),
        "S8_recompute_max_abs_diff_vs_saved": max_abs8,
        "S7_slice_vs_recompute_max_abs_diff": max_abs7,
        "difficulty_axis": DIFFICULTY,
        "partition_average": {str(c): v for c, v in mbc.items()},
        "partition_complete": {str(c): v for c, v in mbc_c.items()},
        "partition_matches_target": bool(partition_ok),
        "cophenetic_corr": {"average": sw_avg["cophenetic_corr"],
                            "complete": sw_comp["cophenetic_corr"]},
    }

    if not partition_ok:
        # STOP for review: do NOT emit downstream stages.
        L.append("\n## ⚠ STOP — partition differs from the principled K=4 target. "
                 "Downstream σ/stages/feasibility NOT generated. Review the 7×7 clustering above.")
        (CLUSTERING / "grouping_excludeOther_outputs.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        (CLUSTERING / "REPORT_subject_grouping_excludeOther_N3025.md").write_text(
            "\n".join(str(x) for x in L), encoding="utf-8")
        print("[STOP] partition mismatch — wrote report + json, no stages emitted.", flush=True)
        return

    clusters_sorted = sorted(mbc.keys())

    # ── 3. σ + smoothness over the 4 cluster centroids ──
    sm = k.sigma_metrics(mbc, cents7)          # reuse: σ exact open path + per-transition + headroom
    Sc, Dc = sm["Sc"], sm["Dc"]
    sigma = sm["sigma_order"]
    # near-optimal σ paths (top-4; report next 3)
    cpaths = g.open_path_exact(Dc, clusters_sorted)

    L.append("\n## 3. σ ordering + smoothness (4 cluster centroids)")
    L.append("- inter-cluster centroid cosine S (4×4):")
    L.append("```\n" + g.fmt_mat(Sc, ["C%d" % c for c in clusters_sorted]) + "\n```")
    L.append(f"- **σ (optimal open path, total cost={sm['sigma_cost']:.3f})** = "
             + " → ".join("C%d{%s}" % (c, ", ".join(mbc[c])) for c in sigma))
    L.append(f"- mean per-transition distance = {sm['sigma_mean']:.3f}; "
             f"max per-transition = {sm['sigma_max']:.3f}")
    L.append(f"- random-order mean (mean off-diag Dc) = {sm['rand_mean']:.3f}")
    L.append(f"- **headroom_mean = rand_mean − σ_mean = {sm['headroom_mean']:.3f}** "
             f"(modest, dragged by Geometry's orthogonality — the honest smoothness for the real subjects)")
    L.append("- next 3 near-optimal σ paths (Geometry ~orthogonal → its σ position floats):")
    for c, pth, _ in cpaths[1:4]:
        L.append(f"    cost={c:.3f}: " + " → ".join("C%d" % cc for cc in pth))

    out["sigma_cluster_order"] = [int(c) for c in sigma]
    out["sigma_cost"] = sm["sigma_cost"]
    out["sigma_mean"] = sm["sigma_mean"]
    out["sigma_max"] = sm["sigma_max"]
    out["rand_mean"] = sm["rand_mean"]
    out["headroom_mean"] = sm["headroom_mean"]
    out["inter_cluster_S"] = Sc.tolist()
    out["inter_cluster_order"] = [int(c) for c in clusters_sorted]
    out["sigma_near_optimal"] = [{"cost": float(c), "path": [int(x) for x in p]}
                                 for c, p, _ in cpaths[1:4]]

    # ── 4. 16-stage snake layout ──
    stages = g.build_stages(sigma, {c: mbc[c] for c in clusters_sorted})
    L.append("\n## 4. Stage layout (4 difficulty × 4 cluster = 16 stages, snake / σ-reversal)")
    L.append(f"- subject clusters in σ order: " + " → ".join("C%d" % c for c in sigma))
    L.append("- rule: difficulty monotone; within each difficulty traverse clusters in σ order, "
             "reversing σ at each difficulty transition (subject cluster held constant across every D-boundary).")
    L.append("\n| stage | difficulty | levels | subject_cluster | subject_members |")
    L.append("|---|---|---|---|---|")
    for st in stages:
        L.append(f"| {st['stage_index']} | {st['difficulty_cluster']} | {st['level_members']} | "
                 f"C{st['subject_cluster']} | {', '.join(st['subject_members'])} |")
    bound_ok = all(
        stages[(i + 1) * K_EXCLUDE - 1]["subject_cluster"] == stages[(i + 1) * K_EXCLUDE]["subject_cluster"]
        for i in range(len(DIFF_ORDER) - 1))
    L.append(f"\n- difficulty-boundary subject-continuity (snake): "
             f"{'OK' if bound_ok else 'BROKEN'}")
    L.append("- harness: arm ④ uses these SAME 16 cells (shuffle subject-cluster visiting order); "
             "arm ⑤ = 4 subject-agnostic random parts per difficulty.")
    out["stages"] = stages
    out["n_stages"] = len(stages)
    out["arm5_K_random_parts"] = K_EXCLUDE

    arm3 = {
        "spec": "arm3_main_exclude_Other_difficulty_x_subject_snake",
        "difficulty_axis": DIFFICULTY,
        "K_subject_clusters": K_EXCLUDE,
        "exclude_subject": "Other",
        "sigma_cluster_order": [int(c) for c in sigma],
        "subject_clusters": {f"C{c}": mbc[c] for c in clusters_sorted},
        "stages": [
            {"stage_index": st["stage_index"],
             "level_set": st["level_members"],
             "subject_set": st["subject_members"],
             "difficulty_cluster": st["difficulty_cluster"],
             "subject_cluster": f"C{st['subject_cluster']}"}
            for st in stages
        ],
    }
    (CLUSTERING / "stages_arm3_excludeOther.json").write_text(
        json.dumps(arm3, indent=2), encoding="utf-8")

    # ── 5. Feasibility on the 28,771 (29K minus Other) ──
    assert PARQUET_29K.exists(), f"29K labels not found: {PARQUET_29K}"
    df29 = pd.read_parquet(PARQUET_29K, columns=["problem_id", "subject", "level"])
    assert sorted(df29["subject"].unique().tolist()) == CANON, "29K subjects mismatch"
    n_other = int((df29["subject"] == "Other").sum())
    n_total = len(df29)
    n_setA = n_total - n_other
    df_no = df29[df29["subject"] != "Other"].copy()
    cells, fsum = k.feasibility_cells(mbc, df_no)

    L.append("\n## 5. Feasibility on the non-Other rows (Set-A)")
    L.append(f"- 29K total = {n_total}; N(Other) = {n_other}; **Set-A total (non-Other) = {n_setA}**")
    L.append(f"- 4×4 cell counts: min_cell={fsum['min_cell']}, #empty={fsum['n_empty']}, "
             f"#<300={fsum['n_lt_300']}, #<150={fsum['n_lt_150']}")
    L.append("\n| difficulty | " + " | ".join("C%d" % c for c in clusters_sorted) + " |")
    L.append("|---|" + "---|" * len(clusters_sorted))
    for d in DIFF_ORDER:
        row = " | ".join(str(cells[(d, int(c))]) for c in clusters_sorted)
        L.append(f"| {d}{DIFFICULTY[d]} | {row} |")
    L.append("- clusters: " + "; ".join("C%d{%s}" % (c, ", ".join(mbc[c])) for c in clusters_sorted))
    # Prealgebra-difficulty confound flag
    ap_cluster = [c for c in clusters_sorted if set(mbc[c]) == {"Algebra", "Prealgebra"}]
    if ap_cluster:
        c = ap_cluster[0]
        L.append(f"\n- **CONFOUND FLAG (documented, not a gate):** cluster C{c}{{Algebra,Prealgebra}} "
                 f"is Prealgebra-dominated at D1 and Algebra-ONLY at D4 (Prealgebra empty above L4). "
                 f"Cluster composition shifts across difficulty: "
                 f"D1={cells[('D1',int(c))]}, D2={cells[('D2',int(c))]}, "
                 f"D3={cells[('D3',int(c))]}, D4={cells[('D4',int(c))]}.")
    out["feasibility_setA"] = {
        "n_total_29K": n_total, "n_other": n_other, "n_setA": n_setA,
        "cells": {f"{d}|C{c}": v for (d, c), v in cells.items()},
        "summary": fsum,
    }

    # ── 6. Robustness (report, not a gate) ──
    L.append("\n## 6. Robustness (report, NOT a gate)")

    def redo(mask, name):
        DAT_h = g.per_layer_center(DAT[mask])
        md_h = md.loc[mask].reset_index(drop=True)
        Sh, _, _ = g.subject_S(DAT_h, md_h, SUBJ7)
        Dh = 1.0 - Sh; np.fill_diagonal(Dh, 0.0); Dh = np.clip((Dh + Dh.T) / 2, 0, None)
        swh = g.clustering_sweep(Dh, SUBJ7, "average")
        labh = fcluster(swh["Z"], K_EXCLUDE, criterion="maxclust").tolist()
        mbch = g.members_by_cluster(labh, SUBJ7)
        corr = g.matrix_corr(Sh, S7)
        same = (partition_set(mbch) == TARGET_PARTITION)
        return mbch, corr, same

    p1mask = (md["pilot"] == "pilot1").to_numpy()
    p2mask = (md["pilot"] == "pilot2").to_numpy()
    mbc_p1, corr_p1, same_p1 = redo(p1mask, "pilot1")
    mbc_p2, corr_p2, same_p2 = redo(p2mask, "pilot2")
    L.append(f"### pilot1-only (re-centered, N={int(p1mask.sum())})")
    L.append("- " + "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_p1.items())))
    L.append(f"- S-corr vs pooled = {corr_p1:+.3f}; target partition match: {'YES' if same_p1 else 'NO'}")
    L.append(f"### pilot2-only (re-centered, N={int(p2mask.sum())})")
    L.append("- " + "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_p2.items())))
    L.append(f"- S-corr vs pooled = {corr_p2:+.3f}; target partition match: {'YES' if same_p2 else 'NO'}")
    out["robustness_pilot1"] = {"members": {str(c): v for c, v in mbc_p1.items()},
                                "S_corr_vs_pooled": corr_p1, "match_target": bool(same_p1)}
    out["robustness_pilot2"] = {"members": {str(c): v for c, v in mbc_p2.items()},
                                "S_corr_vs_pooled": corr_p2, "match_target": bool(same_p2)}

    L.append("\n### gen_len-quintile-balanced subsample (re-centered)")
    bidx = g.sa.genlen_balanced_indices(md, "subject", SUBJ7, n_quint=5)
    if bidx is None or len(bidx) < 2 * len(SUBJ7):
        L.append("- balanced subsample unavailable.")
        out["robustness_genlen_balanced"] = None
    else:
        DAT_b = g.per_layer_center(DAT[bidx])
        md_b = md.loc[bidx].reset_index(drop=True)
        Sb, _, _ = g.subject_S(DAT_b, md_b, SUBJ7)
        Db = 1.0 - Sb; np.fill_diagonal(Db, 0.0); Db = np.clip((Db + Db.T) / 2, 0, None)
        swb = g.clustering_sweep(Db, SUBJ7, "average")
        labb = fcluster(swb["Z"], K_EXCLUDE, criterion="maxclust").tolist()
        mbc_b = g.members_by_cluster(labb, SUBJ7)
        corr_b = g.matrix_corr(Sb, S7)
        same_b = (partition_set(mbc_b) == TARGET_PARTITION)
        L.append(f"- balanced N={len(bidx)}")
        L.append("- " + "; ".join("C%d{%s}" % (c, ", ".join(v)) for c, v in sorted(mbc_b.items())))
        L.append(f"- S-corr vs pooled = {corr_b:+.3f}; target partition match: {'YES' if same_b else 'NO'}")
        L.append("- note: balanced subsample may split {Algebra,Prealgebra} via Prealgebra's "
                 "length/difficulty confound — documented, not a gate.")
        out["robustness_genlen_balanced"] = {
            "N": int(len(bidx)), "members": {str(c): v for c, v in mbc_b.items()},
            "S_corr_vs_pooled": corr_b, "match_target": bool(same_b)}

    # ── PNGs: dendrograms (avg + complete) and reordered 7×7 heatmap ──
    for method, sw in [("average", sw_avg), ("complete", sw_comp)]:
        fig, ax = plt.subplots(figsize=(9, 5))
        dendrogram(sw["Z"], labels=SUBJ7, leaf_rotation=45, leaf_font_size=9, ax=ax)
        ax.set_title(f"Subject dendrogram (exclude Other) — {method} linkage (cosine D, all-36-layer)")
        fig.tight_layout()
        p = CLUSTERING / f"dendro_subjgroup_excludeOther_{method}.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        L.append(f"- dendrogram [{method}]: {p.name}")

    dn = dendrogram(sw_avg["Z"], labels=SUBJ7, no_plot=True)
    leaf_order = dn["ivl"]
    perm = [SUBJ7.index(s) for s in leaf_order]
    S_re = S7[np.ix_(perm, perm)]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(S_re, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(leaf_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(leaf_order, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{S_re[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("7×7 subject cosine S (exclude Other, reordered by average-linkage dendrogram)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ph = CLUSTERING / "heatmap_subjS7_reordered_excludeOther.png"
    fig.savefig(ph, dpi=130); plt.close(fig)
    L.append(f"- reordered 7×7 S heatmap: {ph.name} (leaf order: {leaf_order})")
    out["dendro_leaf_order_average"] = leaf_order

    # ── write outputs ──
    (CLUSTERING / "grouping_excludeOther_outputs.json").write_text(
        json.dumps(out, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
        encoding="utf-8")
    (CLUSTERING / "REPORT_subject_grouping_excludeOther_N3025.md").write_text(
        "\n".join(str(x) for x in L), encoding="utf-8")

    print(f"[OK] partition match={partition_ok}, σ order={sigma}, headroom_mean={sm['headroom_mean']:.3f}", flush=True)
    print(f"[OK] Set-A total (non-Other) = {n_setA} (N(Other)={n_other}); "
          f"feasibility min_cell={fsum['min_cell']}, empty={fsum['n_empty']}", flush=True)
    print("[OK] wrote REPORT_subject_grouping_excludeOther_N3025.md", flush=True)
    print("[OK] wrote stages_arm3_excludeOther.json + grouping_excludeOther_outputs.json + 3 PNGs", flush=True)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
