#!/usr/bin/env python3
"""
dendrogram_subjects.py
======================
Hierarchical clustering / dendrogram of the SUBJECT centroid-cosine
similarity matrices that were already computed and saved by
`similarity_analysis.py` into `sim_matrices_<tag>.npz`.

We DO NOT recompute anything heavy here — we read the centered SUBJECT
cosine matrices (THINKING + FAITHFUL) and turn them into a distance matrix
  D[a,b] = 1 - S[a,b]
then run average-linkage agglomerative clustering (scipy) and draw a
dendrogram. This validates the "subject family" structure
(continuous/algebra cluster vs discrete cluster) seen in the cosine tables.

CPU only, < 1 s. Outputs:
  - dendrogram_subject_THINKING_<tag>.png
  - dendrogram_subject_FAITHFUL_<tag>.png
  - cluster_subject_families_<tag>.txt   (flat 2-cluster membership)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


def build_dendrogram(S, order, title, png_path):
    """S: (n,n) symmetric cosine sim in [-1,1]. order: list of labels."""
    n = len(order)
    # distance = 1 - cosine, clamp to [0, 2], force exact-zero diagonal
    D = 1.0 - S
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * n + 2), 5))
    dendrogram(Z, labels=[str(o) for o in order], ax=ax,
               leaf_rotation=45, leaf_font_size=9, color_threshold=0.7 * D.max())
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("distance (1 - centroid cosine)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    # flat 2-cluster cut for a concrete "family" assignment
    fam = fcluster(Z, t=2, criterion="maxclust")
    families = {}
    for lab, f in zip(order, fam):
        families.setdefault(int(f), []).append(str(lab))
    return Z, families


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="sim_matrices_<tag>.npz from similarity_analysis.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz, allow_pickle=True)

    txt = [f"# Subject hierarchical clustering — {args.tag}", ""]
    for da in ("THINKING", "FAITHFUL"):
        Skey = f"{da}_centered_subject_S"
        Okey = f"{da}_centered_subject_order"
        if Skey not in z:
            txt.append(f"## {da}: matrix '{Skey}' not in npz — skipped"); continue
        S = z[Skey].astype(np.float64)
        order = [str(o) for o in z[Okey].tolist()]
        png = out_dir / f"dendrogram_subject_{da}_{args.tag}.png"
        Z, families = build_dendrogram(
            S, order,
            f"{da} centered subject clustering ({args.tag})\naverage-linkage, dist = 1 - centroid cosine",
            png)
        txt.append(f"## {da} (centered subject)")
        txt.append(f"- dendrogram: {png.name}")
        txt.append("- flat 2-cluster cut (subject families):")
        for fid, members in sorted(families.items()):
            txt.append(f"    family {fid}: {', '.join(members)}")
        txt.append("")
        print(f"[OK] {da} -> {png.name}; families={families}")

    rep = out_dir / f"cluster_subject_families_{args.tag}.txt"
    rep.write_text("\n".join(txt), encoding="utf-8")
    print(f"[OK] wrote {rep}")


if __name__ == "__main__":
    main()
