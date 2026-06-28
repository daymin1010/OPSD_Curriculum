#!/usr/bin/env python3
"""
level_subject_similarity_pooled.py
==================================
Pooled (pilot1 + pilot2 = canonical 3025) recomputation of the SUBJECT and
LEVEL centroid-cosine similarity matrices for THINKING ΔA.

Reuses the *validated* primitives from similarity_analysis.py
(load_pilot, centroids, sim_matrix, within_between, perm_pvalue, spearman,
 fmt_matrix, heatmap) so the methodology is identical to the per-pilot run;
only the data is pooled.

Method (identical to similarity_analysis.py):
  - representation: ΔA_i ∈ R^(36 x 12288), layer-averaged cosine.
  - CENTERED (primary): subtract per-layer global mean μ = mean_i(ΔA_i).
  - S[g,h] = mean_l cos(C_g[l], C_h[l]); gap = within_mean - between_mean.
  - LEVEL ordinality: Spearman ρ(S[a,b], -|a-b|).
  - permutation p (label shuffle, N_PERM=200).

Outputs (analysis/ dir):
  - sim_matrices_pooled3025_levsubj.npz   (centered THINKING subject/level S + order)
  - SIM_pooled3025_levsubj.txt            (formatted matrices + stats, for the report)
  - heatmap_{subject,level}_THINKING_pooled3025.png

CPU only.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

import similarity_analysis as sa

BASE = Path("/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/"
            "reasoning_pivot/activation")
PILOT1 = BASE / "outputs/pilot/shifts"
PILOT2 = BASE / "outputs/pilot2/shifts"
OUT = BASE / "analysis"


def run_group(DA_c, md, col, nm, lines):
    """Compute centered centroid-cosine matrix + stats for one grouping."""
    vc = md[col].value_counts()
    if col == "level":
        order = sorted(vc.index.tolist(), key=lambda x: int(x))
    else:
        order = sorted(vc.index.tolist())
    idxg = {g: md.index[md[col] == g].to_numpy() for g in order}

    cents = sa.centroids(DA_c, idxg)
    S = sa.sim_matrix(cents, order)
    wm, bm, gap, _ = sa.within_between(DA_c, idxg, cents, order)

    labels_arr = md[col].to_numpy()
    keep_mask = np.isin(labels_arr, order)
    p = sa.perm_pvalue(DA_c[keep_mask], labels_arr[keep_mask], order, gap)

    lines.append(f"\n### {nm} grouping (groups={len(order)})")
    lines.append("group sizes: " + ", ".join(f"{g}:{len(idxg[g])}" for g in order))
    lines.append(f"- within_mean cos = {wm:+.4f} | between_mean cos = {bm:+.4f} "
                 f"| gap = {gap:+.4f}")
    lines.append(f"- permutation p(gap >= obs) = {p:.4f} (N_PERM={sa.N_PERM})")

    if col == "level":
        levs = [int(x) for x in order]
        pc, pnd = [], []
        for a in range(len(levs)):
            for b in range(a + 1, len(levs)):
                pc.append(S[a, b]); pnd.append(-abs(levs[a] - levs[b]))
        rho_all = sa.spearman(pc, pnd)
        # also L1-L7 only (drop L8 per caveat)
        pc7, pnd7 = [], []
        for a in range(len(levs)):
            for b in range(a + 1, len(levs)):
                if levs[a] <= 7 and levs[b] <= 7:
                    pc7.append(S[a, b]); pnd7.append(-abs(levs[a] - levs[b]))
        rho7 = sa.spearman(pc7, pnd7)
        lines.append(f"- ORDINALITY rho(cos,-|dlevel|): L1-L8 = {rho_all:+.4f} | "
                     f"L1-L7 = {rho7:+.4f}")

    lines.append("- centroid cosine matrix (centered, THINKING):")
    lines.append("```\n" + sa.fmt_matrix(S, order) + "\n```")

    png = OUT / f"heatmap_{col}_THINKING_pooled3025.png"
    sa.heatmap(S, order, f"THINKING {nm} centroid cosine (pooled 3025)", png)

    return order, S


def main():
    print("[load] pilot1 ...", flush=True)
    _, DAT1, md1 = sa.load_pilot(PILOT1, None)
    print("[load] pilot2 ...", flush=True)
    _, DAT2, md2 = sa.load_pilot(PILOT2, None)

    DAT = np.concatenate([DAT1, DAT2], axis=0)
    import pandas as pd
    md = pd.concat([md1, md2], ignore_index=True)
    N = len(md)
    print(f"[pooled] N = {N} (pilot1 {len(md1)} + pilot2 {len(md2)})", flush=True)
    assert N == len(DAT)

    # per-layer global-mean centering (primary)
    mu = DAT.astype(np.float32).mean(axis=0, keepdims=True)
    DA_c = DAT.astype(np.float32) - mu

    lines = [f"# POOLED 3025 — SUBJECT/LEVEL centroid-cosine (THINKING, centered)"]
    lines.append(f"N = {N} (pilot1 {len(md1)} + pilot2 {len(md2)}); "
                 f"subjects={md['subject'].nunique()}, "
                 f"levels={sorted(md['level'].unique().tolist())}")

    saved = {}
    for col, nm in [("subject", "SUBJECT"), ("level", "LEVEL")]:
        order, S = run_group(DA_c, md, col, nm, lines)
        saved[f"THINKING_centered_{col}_S"] = S
        saved[f"THINKING_centered_{col}_order"] = np.array([str(o) for o in order])

    np.savez(OUT / "sim_matrices_pooled3025_levsubj.npz", **saved)
    (OUT / "SIM_pooled3025_levsubj.txt").write_text(
        "\n".join(str(x) for x in lines), encoding="utf-8")
    print("\n".join(str(x) for x in lines))
    print(f"\n[OK] wrote {OUT/'SIM_pooled3025_levsubj.txt'} and .npz")


if __name__ == "__main__":
    main()
