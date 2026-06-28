#!/usr/bin/env python3
"""
similarity_analysis.py
======================
Group-centroid SIMILARITY analysis of activation-shifts (ΔA) for the
thinking-mode pilot.  Measures how similar groups are to each other along
three groupings:

  (1) UNIT    = (subject, level) cell  (cells with n >= MIN_N)
  (2) SUBJECT = 8 canonical subjects
  (3) LEVEL   = 1..8 (ordinal)

METHOD (mirrored verbatim into the report)
------------------------------------------
Representation:
  Each problem i -> ΔA_i ∈ R^(36 x 12288). We analyze BOTH `dA_thinking`
  and `dA_faithful`. Similarity is computed on the LAYER-AVERAGED cosine,
  i.e. for two vectors we average the per-layer cosine over all 36 layers
  (each layer's 12288-d vector is L2-normalized before the dot product).
  This avoids a few high-norm layers dominating a flattened 442k-d cosine.

Common-component removal (CENTERING):
  ΔA has a dominant shared direction (in prior unit_analysis, within- and
  between-unit cosines were both ~+0.9). Raw centroid cosines are therefore
  ~1 everywhere and uninformative. We report TWO variants:
    - raw      : centroids computed on ΔA_i directly
    - centered : subtract the GLOBAL mean μ = mean_i(ΔA_i) from every ΔA_i
                 first (per layer), then compute centroids. This is the
                 PRIMARY result; raw is shown for transparency.

Group centroid:
  For a group g, centroid C_g[l] = mean over members i∈g of (ΔA_i[l]).
  (On centered data this is the mean of centered vectors.)

Similarity matrix:
  S[g,h] = mean_l cos( C_g[l], C_h[l] )   (layer-averaged cosine of centroids)

Separability (is the grouping meaningful?):
  within(g)  = mean over members i∈g of layer-avg cos(ΔA_i, C_g)
  between    = mean over g!=h of layer-avg cos(C_g, C_h)
  gap = within_mean - between_mean   (higher => groups internally coherent
        and mutually distinct).

Ordinality (LEVEL only):
  Since level is ordinal we test whether adjacent levels are more similar:
  Spearman ρ between S[a,b] (off-diagonal centroid cosine) and -|a-b|.
  Positive ρ => closer levels look more alike.

Significance (permutation test):
  Null = shuffle group labels (LABEL permutation) N_PERM times; recompute
  `gap`. p = fraction of permuted gaps >= observed gap. Reported per grouping.

Length-confound robustness:
  Centroid cosines may merely reflect gen_len differences. We recompute the
  centered SUBJECT and LEVEL matrices on a gen_len-balanced subsample
  (stratify by gen_len quintile, take min per-quintile count per group when
  feasible) and report whether the separability gap survives.

OUTPUTS (analysis/ dir)
  - REPORT_similarity_<tag>.md   (this method text + all matrices + verdict)
  - sim_matrices_<tag>.npz       (raw/centered x thinking/faithful matrices)
  - heatmap_{subject,level,unit}_<tag>.png

CPU only. Memory: fp16 (N,36,12288)*2 ~ 2.7 GB.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_N = 10          # min members for a UNIT cell
N_PERM = 200        # p-value resolution 1/(N_PERM+1); plenty for screening
SEED = 42
rng = np.random.default_rng(SEED)


# ───────────────────────────── helpers ────────────────────────────────────
def spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return float("nan")
    ra, rb = rankdata(a[m]), rankdata(b[m])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def l2norm_rows(M, eps=1e-8):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + eps)


def layeravg_cos(A, B):
    """A,B: (36,D). Return mean over layers of cos(A[l],B[l])."""
    An = l2norm_rows(A.astype(np.float32))
    Bn = l2norm_rows(B.astype(np.float32))
    return float((An * Bn).sum(axis=1).mean())


def centroids(DA, idx_by_group):
    """Return dict group-> (36,D) centroid."""
    return {g: DA[idx].astype(np.float32).mean(axis=0) for g, idx in idx_by_group.items()}


def sim_matrix(cents, order):
    n = len(order)
    S = np.eye(n, dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            c = layeravg_cos(cents[order[a]], cents[order[b]])
            S[a, b] = S[b, a] = c
    return S


def normalize_members(DA):
    """Per-layer L2-normalize every member once. DA:(N,36,D) -> (N,36,D) f32."""
    V = DA.astype(np.float32)
    return V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-8)


def within_between(DA, idx_by_group, cents, order, DAn=None):
    """within = mean member-cos to own centroid; between = mean centroid-cos.
    DAn (precomputed per-layer L2-normalized members) avoids re-normalizing
    inside the permutation loop (the dominant cost)."""
    if DAn is None:
        DAn = normalize_members(DA)
    withins = []
    for g in order:
        idx = idx_by_group[g]
        Cn = l2norm_rows(cents[g].astype(np.float32))   # (36,D)
        cos_ml = np.einsum("mld,ld->ml", DAn[idx], Cn)  # (m,36)
        withins.append(float(cos_ml.mean()))
    within_mean = float(np.mean(withins))

    n = len(order)
    bet = [layeravg_cos(cents[order[a]], cents[order[b]])
           for a in range(n) for b in range(a + 1, n)]
    between_mean = float(np.mean(bet)) if bet else float("nan")
    return within_mean, between_mean, within_mean - between_mean, withins


def perm_pvalue(DA, labels, order, observed_gap, DAn=None):
    """Shuffle labels, recompute gap. p = P(gap_perm >= observed).
    DAn is precomputed once so each iteration only indexes + computes centroids."""
    if DAn is None:
        DAn = normalize_members(DA)
    labels = np.asarray(labels)
    ge = 0
    log_every = max(1, N_PERM // 10)
    for it in range(N_PERM):
        perm = rng.permutation(labels)
        idxg = {g: np.where(perm == g)[0] for g in order}
        idxg = {g: ix for g, ix in idxg.items() if len(ix) >= 2}
        if len(idxg) < 2:
            continue
        cents = centroids(DA, idxg)
        oo = list(idxg.keys())
        _, _, gap, _ = within_between(DA, idxg, cents, oo, DAn=DAn)
        if gap >= observed_gap:
            ge += 1
        if (it + 1) % log_every == 0:
            print(f"    perm {it+1}/{N_PERM} (ge={ge})", flush=True)
    return (ge + 1) / (N_PERM + 1)



def fmt_matrix(S, order):
    labels = [str(o) for o in order]
    w = max(8, max(len(l) for l in labels) + 1)
    head = " " * w + "".join(f"{l:>8}" for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        rows.append(f"{l:>{w}}" + "".join(f"{S[i,j]:>8.3f}" for j in range(len(labels))))
    return "\n".join(rows)


def heatmap(S, order, title, path):
    fig, ax = plt.subplots(figsize=(1.1 * len(order) + 2, 1.1 * len(order) + 2))
    im = ax.imshow(S, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels([str(o) for o in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(o) for o in order], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ───────────────────────────── load ───────────────────────────────────────
def load_pilot(shifts_dir: Path, max_n):
    files = sorted(shifts_dir.glob("*.pt"))
    if max_n:
        files = files[:max_n]
    n = len(files)
    print(f"[load] {n} .pt files")
    DAF = np.zeros((n, 36, 12288), dtype=np.float16)
    DAT = np.zeros((n, 36, 12288), dtype=np.float16)
    meta = []
    keep = 0
    for i, pf in enumerate(files):
        try:
            d = torch.load(pf, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [skip] {pf.name}: {e}"); continue
        DAF[keep] = d["dA_faithful"].to(torch.float16).numpy()
        DAT[keep] = d["dA_thinking"].to(torch.float16).numpy()
        meta.append({
            "problem_id": d.get("problem_id"),
            "subject": str(d.get("subject")),
            "level": int(d.get("level", -1)),
            "gen_len": int(d.get("gen_len", -1)),
        })
        keep += 1
        if (i + 1) % 300 == 0:
            print(f"  ...{i+1}/{n}")
    DAF = DAF[:keep]; DAT = DAT[:keep]
    md = pd.DataFrame(meta)
    md["unit"] = md["subject"] + "|L" + md["level"].astype(str)
    print(f"[load] kept {keep}")
    return DAF, DAT, md


# ───────────────────────────── per-grouping driver ────────────────────────
def run_grouping(DA, md, col, name, lines, out_dir, tag, da_name, make_png=False):
    vc = md[col].value_counts()
    if col == "unit":
        order = sorted([g for g in vc.index if vc[g] >= MIN_N])
    elif col == "level":
        order = sorted([g for g in vc.index], key=lambda x: int(x))
    else:
        order = sorted(vc.index.tolist())
    if len(order) < 2:
        lines.append(f"\n### [{da_name}] {name}: <2 groups, skipped"); return None

    idxg = {g: md.index[md[col] == g].to_numpy() for g in order}
    lines.append(f"\n### [{da_name}] {name} grouping  (groups={len(order)}, MIN_N={MIN_N if col=='unit' else 1})")
    lines.append("group sizes: " + ", ".join(f"{g}:{len(idxg[g])}" for g in order))

    cents = centroids(DA, idxg)
    S = sim_matrix(cents, order)
    wm, bm, gap, withins = within_between(DA, idxg, cents, order)
    labels_arr = md[col].to_numpy()
    # permutation only over members that belong to `order`
    keep_mask = np.isin(labels_arr, order)
    p = perm_pvalue(DA[keep_mask], labels_arr[keep_mask], order, gap)

    lines.append(f"- within_mean cos = {wm:+.3f} | between_mean cos = {bm:+.3f} "
                 f"| gap = {gap:+.3f}")
    lines.append(f"- permutation p(gap >= obs) = {p:.4f}  (N_PERM={N_PERM}, label shuffle)")
    lines.append("- centroid cosine matrix:")
    lines.append("```\n" + fmt_matrix(S, order) + "\n```")

    if col == "level":
        levs = [int(x) for x in order]
        pairs_cos, pairs_negdist = [], []
        for a in range(len(levs)):
            for b in range(a + 1, len(levs)):
                pairs_cos.append(S[a, b]); pairs_negdist.append(-abs(levs[a] - levs[b]))
        rho = spearman(pairs_cos, pairs_negdist)
        lines.append(f"- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = {rho:+.3f}  "
                     f"(positive => adjacent levels more similar)")

    if make_png:
        png = out_dir / f"heatmap_{col}_{da_name}_{tag}.png"
        heatmap(S, order, f"{da_name} {name} centroid cosine ({tag})", png)
        lines.append(f"- heatmap: {png.name}")

    return {"order": [str(o) for o in order], "S": S,
            "within": wm, "between": bm, "gap": gap, "p": p}


def genlen_balanced_indices(md, col, order, n_quint=5):
    """Subsample indices so each group has equal representation across gen_len quintiles."""
    gl = md["gen_len"].to_numpy()
    try:
        q = pd.qcut(gl, n_quint, labels=False, duplicates="drop")
    except Exception:
        return None
    keep = []
    for qi in np.unique(q):
        # within this quintile, take min group count across `order`
        per_group = {g: md.index[(md[col] == g) & (q == qi)].to_numpy() for g in order}
        per_group = {g: ix for g, ix in per_group.items() if len(ix) > 0}
        if len(per_group) < 2:
            continue
        m = min(len(ix) for ix in per_group.values())
        for g, ix in per_group.items():
            sel = rng.choice(ix, size=m, replace=False)
            keep.extend(sel.tolist())
    return np.array(sorted(set(keep))) if keep else None


# ───────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    DAF, DAT, md = load_pilot(Path(args.shifts_dir), args.max_n)
    N = len(md)

    lines = [f"# ΔA Group-Similarity Analysis — {args.tag}", ""]
    lines.append(f"- N = **{N}**, subjects={md['subject'].nunique()}, "
                 f"levels={sorted(md['level'].unique().tolist())}, "
                 f"units(n>= {MIN_N})={ (md['unit'].value_counts()>=MIN_N).sum() }")
    lines.append("\n## METHOD")
    lines.append(__doc__.strip())

    saved = {}
    for da_name, DA in [("THINKING", DAT), ("FAITHFUL", DAF)]:
        # global-mean centering (per layer) -> primary
        mu = DA.astype(np.float32).mean(axis=0, keepdims=True)
        DA_c = (DA.astype(np.float32) - mu)

        lines.append(f"\n## ===== {da_name} :: CENTERED (primary) =====")
        for col, nm, png in [("subject", "SUBJECT", True),
                             ("level", "LEVEL", True),
                             ("unit", "UNIT (subject×level)", da_name == "THINKING")]:
            r = run_grouping(DA_c, md, col, nm, lines, out_dir, args.tag, da_name, make_png=png)
            if r is not None:
                saved[f"{da_name}_centered_{col}_S"] = r["S"]
                saved[f"{da_name}_centered_{col}_order"] = np.array(r["order"])

        lines.append(f"\n## ===== {da_name} :: RAW (transparency) =====")
        for col, nm in [("subject", "SUBJECT"), ("level", "LEVEL")]:
            r = run_grouping(DA, md, col, nm, lines, out_dir, args.tag, da_name + "_raw", make_png=False)
            if r is not None:
                saved[f"{da_name}_raw_{col}_S"] = r["S"]
                saved[f"{da_name}_raw_{col}_order"] = np.array(r["order"])

        # length-confound robustness (centered subject & level)
        lines.append(f"\n## ===== {da_name} :: gen_len-balanced robustness (centered) =====")
        for col, nm in [("subject", "SUBJECT"), ("level", "LEVEL")]:
            vc = md[col].value_counts()
            order = sorted(vc.index.tolist(), key=lambda x: (int(x) if col == "level" else x))
            bidx = genlen_balanced_indices(md, col, order)
            if bidx is None or len(bidx) < 2 * len(order):
                lines.append(f"\n### [{da_name}] {nm}: gen_len-balanced subsample unavailable")
                continue
            md_b = md.loc[bidx].reset_index(drop=True)
            DA_b = DA_c[bidx]
            lines.append(f"\n(gen_len-balanced subsample N={len(bidx)})")
            run_grouping(DA_b, md_b, col, nm, lines, out_dir, args.tag,
                         da_name + "_balanced", make_png=False)

    np.savez(out_dir / f"sim_matrices_{args.tag}.npz", **saved)
    rep = out_dir / f"REPORT_similarity_{args.tag}.md"
    rep.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print(f"[OK] wrote {rep}")


if __name__ == "__main__":
    main()
