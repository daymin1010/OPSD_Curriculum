#!/usr/bin/env python3
"""
residualize_analysis.py
=======================
ABLATION on the common-component removal step of similarity_analysis.py.

The primary analysis used GLOBAL-MEAN centering (subtract μ = mean_i ΔA_i).
Here we test a more aggressive removal: strip the top-K PRINCIPAL COMPONENTS
of ΔA (per layer) instead of just the mean. The hypothesis: a single shared
"reasoning-shift" axis dominates ΔA; removing 1-2 PCs should make the
subject/level separability gap *larger and cleaner* than mean-centering.

Per layer l (36) and a chosen ΔA (THINKING / FAITHFUL):
  X = ΔA[:, l, :]                      (N, 12288)
  X0 = X - mean(X)                     (also remove mean, like centering)
  V  = top-K right singular vectors of X0   (K, 12288)
  X_res = X0 - (X0 @ V^T) @ V          (project out the top-K shared axes)
We then run the SAME subject/level groupings (centroid cosine, within/between
gap, permutation p, LEVEL ordinality) as similarity_analysis.py, reusing its
helpers, and tabulate gap/p/ordinality for K ∈ {0(mean-only), 1, 2}.

K=0 reproduces the primary mean-centered numbers (sanity cross-check).

CPU only. Reuses load + grouping from similarity_analysis.py (same dir).
Outputs:
  - REPORT_residualize_<tag>.md
  - residualize_summary_<tag>.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

# import helpers from the proven sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
import similarity_analysis as SA  # noqa: E402


def residualize_perlayer(DA_f32, K):
    """DA_f32: (N,36,D) float32. Remove per-layer mean, then project out top-K PCs.
    K=0 => mean-only (centering). Returns (N,36,D) float32."""
    N, L, D = DA_f32.shape
    out = np.empty_like(DA_f32)
    for l in range(L):
        X = DA_f32[:, l, :]
        X0 = X - X.mean(axis=0, keepdims=True)
        if K > 0:
            # economy SVD: X0 = U S Vt ; rows of Vt are principal axes
            # full_matrices=False -> Vt is (min(N,D), D)
            _, _, Vt = np.linalg.svd(X0, full_matrices=False)
            V = Vt[:K]                      # (K, D)
            X0 = X0 - (X0 @ V.T) @ V        # project out top-K shared axes
        out[:, l, :] = X0
        if (l + 1) % 12 == 0:
            print(f"    [resid K={K}] layer {l+1}/{L}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--ks", default="0,1,2", help="comma list of K values")
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    Ks = [int(x) for x in args.ks.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    DAF, DAT, md = SA.load_pilot(Path(args.shifts_dir), args.max_n)
    N = len(md)

    lines = [f"# ΔA Residualization Ablation (PCA top-K removal) — {args.tag}", ""]
    lines.append(f"- N = **{N}**, K values tested = {Ks}")
    lines.append("\n## METHOD")
    lines.append(__doc__.strip())

    summary = []  # (da, K, col, within, between, gap, p, ordinality)
    for da_name, DA in [("THINKING", DAT), ("FAITHFUL", DAF)]:
        DAf = DA.astype(np.float32)
        for K in Ks:
            print(f"[run] {da_name} K={K}", flush=True)
            DA_r = residualize_perlayer(DAf, K)
            label = f"{da_name}_resid{K}"
            lines.append(f"\n## ===== {da_name} :: residualize K={K} "
                         f"({'mean-only' if K == 0 else f'mean + top-{K} PCs removed'}) =====")
            for col, nm in [("subject", "SUBJECT"), ("level", "LEVEL")]:
                r = SA.run_grouping(DA_r, md, col, nm, lines, out_dir, args.tag,
                                    label, make_png=False)
                if r is None:
                    continue
                ordv = ""
                if col == "level":
                    # recompute ordinality value for the summary row
                    order = [int(x) for x in r["order"]]
                    S = r["S"]
                    pc, pn = [], []
                    for a in range(len(order)):
                        for b in range(a + 1, len(order)):
                            pc.append(S[a, b]); pn.append(-abs(order[a] - order[b]))
                    ordv = SA.spearman(pc, pn)
                summary.append((da_name, K, col, r["within"], r["between"],
                                r["gap"], r["p"], ordv))

    # summary table
    lines.append("\n## ===== SUMMARY (gap & significance vs K) =====")
    lines.append("```")
    lines.append(f"{'dA':<9}{'K':>3}{'group':>9}{'within':>9}{'between':>9}"
                 f"{'gap':>9}{'p':>9}{'ord_rho':>9}")
    for (da, K, col, w, b, g, p, ordv) in summary:
        ords = f"{ordv:+.3f}" if isinstance(ordv, float) else "   -   "
        lines.append(f"{da:<9}{K:>3}{col:>9}{w:>9.3f}{b:>9.3f}{g:>9.3f}{p:>9.4f}{ords:>9}")
    lines.append("```")

    rep = out_dir / f"REPORT_residualize_{args.tag}.md"
    rep.write_text("\n".join(str(x) for x in lines), encoding="utf-8")

    # csv
    import csv
    csvp = out_dir / f"residualize_summary_{args.tag}.csv"
    with open(csvp, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["dA", "K", "group", "within", "between", "gap", "p", "ordinality_rho"])
        for row in summary:
            wtr.writerow(row)

    print(f"[OK] wrote {rep}")
    print(f"[OK] wrote {csvp}")


if __name__ == "__main__":
    main()
