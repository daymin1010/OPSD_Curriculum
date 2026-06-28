#!/usr/bin/env python3
"""
unit_similarity_pooled3025.py — Inter-unit prototype similarity on the
POOLED Qwen3-8B set (pilot1 + pilot2, canonical N≈3025).
========================================================================
BASE  : OPSD_Curriculum/reasoning_pivot/activation/analysis
        (data loader reuse: pooled_analysis.load_pooled -> similarity_analysis.load_pilot)
        --> the *3025* set is Qwen/Qwen3-8B (extract_thinking_pilot.py
            DEFAULT_MODEL_ID="Qwen/Qwen3-8B", spec thinking_8k_v1 / _v2).
REF   : 4.6_Task2/activation/analysis/nait_unit_similarity.py
        (mechanism: sign-calibrated PC1 prototypes, residualize global PC1,
         block averages, dendrogram) — kept IDENTICAL; only the data source
         changes (delta_cache(N=2666) -> pooled pilot1+pilot2(N≈3025)).

UNIT = subject × level   (e.g. "Algebra_L3").

Two versions of the SAME mechanism:
  - raw   ΔA : prototypes on raw THINKING activation shifts (dA_thinking)
  - resid ΔA : prototypes after removing per-layer global PC1

Definitions (per unit u):
  v_u^l       = sign-calibrated PC1 of {ΔA_s^l : s ∈ u}   (unit-norm, Eq.4)
  S_l[u,u']   = v_u^l · v_{u'}^l                          (= cos)
  S_agg[u,u'] = mean_l  v_u^l · v_{u'}^l

NEW quantification (does unit cluster by subject or by level?):
  - silhouette of units on D = 1 - S_agg, labelled by subject and by level
  - cophenetic correlation of the average-linkage dendrogram

CPU only.  No GPU. ~10-20 min.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# analysis dir on path so the OPSD loaders import cleanly
ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

try:
    import pandas as pd
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
    from scipy.spatial.distance import squareform
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

try:
    from sklearn.metrics import silhouette_score
    _HAVE_SK = True
except ImportError:
    _HAVE_SK = False

import similarity_analysis as sa          # validated OPSD metric helpers
import pooled_analysis as pooled          # load_pooled(pilot1+pilot2)

# ─────────────────────────────────────────────────────────────────────────────
# Output paths (OPSD reasoning_pivot side — does NOT touch analysis_qwen3_8b)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = ANALYSIS / "outputs" / "unit_similarity_pooled3025"
FIG_DIR = OUT_DIR / "figures"
REPORT  = OUT_DIR / "unit_similarity_pooled3025_report.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_N   = getattr(sa, "MIN_N", 4)
PCA_Q   = 6
EXCLUDE_SUBJECTS = {"Other"}
EXPECT_N = 3025                 # canonical pooled N (soft check)
EXPECT_N_TOL = 50              # allow small drift (non-finite drops etc.)

torch.set_num_threads(max(1, os.cpu_count() or 1))


# ─────────────────────────────────────────────────────────────────────────────
# PC1 mechanism (REF: nait_unit_similarity / unit_similarity_qwen3_8b)
# ─────────────────────────────────────────────────────────────────────────────
def pc1_with_sign(X: np.ndarray) -> np.ndarray:
    """PC1 (unit-norm) with sign calibration:  v <- -v if mean·v < 0."""
    t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    n = t.shape[0]
    q = max(1, min(PCA_Q, n - 1))
    _, _, V = torch.pca_lowrank(t, q=q, center=True, niter=4)
    v = V[:, 0].numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    mu = X.mean(axis=0)
    if float(mu @ v) < 0:
        v = -v
    return v


def global_pc1(X: np.ndarray) -> np.ndarray:
    """Per-layer global PC1 (unit-norm, no sign calib) for residualization."""
    t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    q = max(1, min(PCA_Q, t.shape[0] - 1))
    _, _, V = torch.pca_lowrank(t, q=q, center=True, niter=4)
    v = V[:, 0].numpy().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def build_prototypes(delta, unit_groups, unit_names, n_layers, d_hid):
    """delta: (N, L, D) ndarray. Returns V_raw, V_resid each (L, U, D) float32."""
    U = len(unit_names)
    V_raw   = np.zeros((n_layers, U, d_hid), dtype=np.float32)
    V_resid = np.zeros((n_layers, U, d_hid), dtype=np.float32)
    t0 = time.time()
    for l in range(n_layers):
        Xl = np.ascontiguousarray(delta[:, l, :], dtype=np.float32)   # (N, D)
        g = global_pc1(Xl)
        coeff = Xl @ g
        Xl_resid = Xl - np.outer(coeff, g)
        for u_idx, u in enumerate(unit_names):
            idx = unit_groups[u]
            V_raw[l, u_idx]   = pc1_with_sign(Xl[idx])
            V_resid[l, u_idx] = pc1_with_sign(Xl_resid[idx])
        if (l + 1) % 6 == 0 or l == n_layers - 1:
            print(f"  [proto] layer {l+1}/{n_layers}  ({time.time()-t0:.1f}s)", flush=True)
    return V_raw, V_resid


def layer_cos(V_layer: np.ndarray) -> np.ndarray:
    return V_layer @ V_layer.T


def agg_cos(V: np.ndarray) -> np.ndarray:
    L = V.shape[0]
    return np.mean([layer_cos(V[l]) for l in range(L)], axis=0)


def subj_of(u: str) -> str: return u.rsplit("_L", 1)[0]
def lvl_of (u: str) -> str: return "L" + u.rsplit("_L", 1)[1]


def short(u: str) -> str:
    return (u.replace("Intermediate Algebra", "IA")
             .replace("Counting & Probability", "C&P")
             .replace("Number Theory", "NT")
             .replace("Precalculus", "Pcalc")
             .replace("Prealgebra", "Prealg")
             .replace("Geometry", "Geom"))


def block_average(S_agg, unit_names, group_fn):
    groups = {}
    for i, u in enumerate(unit_names):
        groups.setdefault(group_fn(u), []).append(i)
    gnames = sorted(groups.keys())
    B = len(gnames)
    M = np.zeros((B, B))
    for i, gi in enumerate(gnames):
        for j, gj in enumerate(gnames):
            sub = S_agg[np.ix_(groups[gi], groups[gj])]
            if gi == gj:
                mask = ~np.eye(sub.shape[0], dtype=bool)
                M[i, j] = sub[mask].mean() if mask.any() else np.nan
            else:
                M[i, j] = sub.mean()
    return gnames, M


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap(M, labels, title, path, vmin=-1, vmax=1, cmap="RdBu_r", annot=False):
    n = M.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, n*0.4+2), max(8, n*0.36+2)))
    sns.heatmap(M, xticklabels=labels, yticklabels=labels, cmap=cmap,
                vmin=vmin, vmax=vmax, square=True, ax=ax, annot=annot,
                fmt=".2f", annot_kws={"size": 6}, linewidths=0.2)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=6); plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()


def make_linkage(S_agg):
    D = np.clip(1.0 - S_agg, 0, 2)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="average")
    return D, cond, Z


def fig_dendrogram(Z, unit_names, path, tag):
    fig, ax = plt.subplots(figsize=(max(14, len(unit_names)*0.35), 6))
    dendrogram(Z, labels=[short(u) for u in unit_names], leaf_rotation=75,
               leaf_font_size=7, ax=ax, color_threshold=0.5 * max(Z[:, 2]))
    ax.set_ylabel("1 − cos_sim (avg over layers)")
    ax.set_title(f"Unit Prototype Dendrogram  [{tag}]")
    plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()


def silhouettes(D, unit_names):
    """Silhouette of units on precomputed distance D, labelled by subj / level."""
    out = {}
    if not _HAVE_SK:
        return {"subject": None, "level": None}
    subj = np.array([subj_of(u) for u in unit_names])
    lvl  = np.array([lvl_of(u)  for u in unit_names])
    for name, lab in [("subject", subj), ("level", lvl)]:
        uniq = np.unique(lab)
        if 2 <= len(uniq) < len(lab):
            try:
                out[name] = float(silhouette_score(D, lab, metric="precomputed"))
            except Exception as e:
                print(f"  [silhouette {name} skip] {e}"); out[name] = None
        else:
            out[name] = None
    return out


def make_section(tag, S_agg, unit_names):
    U = len(unit_names)
    subj_n, subj_M = block_average(S_agg, unit_names, subj_of)
    within_subj = np.nanmean(np.diag(subj_M))
    off = subj_M.copy(); np.fill_diagonal(off, np.nan); across_subj = np.nanmean(off)

    lvl_n, lvl_M = block_average(S_agg, unit_names, lvl_of)
    within_lvl = np.nanmean(np.diag(lvl_M))
    off_l = lvl_M.copy(); np.fill_diagonal(off_l, np.nan); across_lvl = np.nanmean(off_l)

    pairs = []
    for i in range(U):
        for j in range(i+1, U):
            pairs.append((S_agg[i, j], unit_names[i], unit_names[j]))
    pairs.sort(reverse=True)

    D, cond, Z = make_linkage(S_agg)
    coph, _ = cophenet(Z, cond)
    sil = silhouettes(D, unit_names)

    lines = [f"\n## {tag}\n\n",
             f"- **Subject block** (avg cos): within = **{within_subj:.3f}**, across = **{across_subj:.3f}**, ratio = **{within_subj/(abs(across_subj)+1e-9):.2f}x**\n",
             f"- **Level   block** (avg cos): within = **{within_lvl:.3f}**, across = **{across_lvl:.3f}**, ratio = **{within_lvl/(abs(across_lvl)+1e-9):.2f}x**\n",
             f"- **Silhouette** (units on 1−S_agg): by subject = **{sil['subject']}**, by level = **{sil['level']}**  (higher ⇒ that label explains clustering)\n",
             f"- **Cophenetic corr** (avg-linkage dendrogram fidelity) = **{coph:.3f}**\n",
             "\n### Top 15 most similar unit pairs (aggregated cos)\n\n| cos | u1 | u2 | same subj? | same lvl? |\n|---|---|---|---|---|\n"]
    for c, a, b in pairs[:15]:
        lines.append(f"| {c:.3f} | {short(a)} | {short(b)} | {'✓' if subj_of(a)==subj_of(b) else '·'} | {'✓' if lvl_of(a)==lvl_of(b) else '·'} |\n")
    lines.append("\n### Bottom 10 most DIS-similar pairs\n\n| cos | u1 | u2 |\n|---|---|---|\n")
    for c, a, b in pairs[-10:][::-1]:
        lines.append(f"| {c:.3f} | {short(a)} | {short(b)} |\n")
    return "".join(lines), subj_n, subj_M, lvl_n, lvl_M, Z, sil, coph


# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None,
                    help="smoke: limit .pt loaded per pilot")
    args = ap.parse_args()

    print("=" * 72)
    print("  POOLED Qwen3-8B (pilot1+pilot2) Unit Prototype Similarity")
    print("  raw=과거 / resid=현재   |   UNIT = subject × level")
    print("=" * 72, flush=True)

    t0 = time.time()
    DAF, DAT, md, ninfo = pooled.load_pooled(args.max_n)
    del DAF  # we only need THINKING ΔA (dA_thinking)
    import gc; gc.collect()

    delta = np.ascontiguousarray(DAT, dtype=np.float32)   # (N, L, D)
    del DAT; gc.collect()
    N, n_layers, d_hid = delta.shape
    print(f"[load] N={N} L={n_layers} D={d_hid}  ({time.time()-t0:.0f}s)", flush=True)

    # ── GATE: pooled N sanity ────────────────────────────────────────────────
    assert delta.shape[0] == len(md), f"delta N={delta.shape[0]} vs md={len(md)}"
    if args.max_n is None:
        assert abs(N - EXPECT_N) <= EXPECT_N_TOL, \
            f"pooled N={N} not within {EXPECT_N}±{EXPECT_N_TOL} (3025셋 게이트 실패)"
        print(f"[GATE] pooled N={N} ≈ {EXPECT_N} ✓", flush=True)
    else:
        print(f"[GATE] SMOKE (max_n={args.max_n}); N={N} (게이트 skip)", flush=True)

    # ── units (subject × level) ──────────────────────────────────────────────
    # NOTE: similarity_analysis builds md["unit"] with a "|L" separator, but our
    # subj_of/lvl_of parsers split on "_L". Rebuild canonically with "_L".
    md = md.copy()
    md["unit"] = md["subject"].astype(str) + "_L" + md["level"].astype(int).astype(str)


    unit_groups = {}
    for pos, row in enumerate(md.itertuples(index=False)):
        if str(getattr(row, "subject")) in EXCLUDE_SUBJECTS:
            continue
        unit_groups.setdefault(getattr(row, "unit"), []).append(pos)
    unit_groups = {u: p for u, p in unit_groups.items() if len(p) >= MIN_N}
    unit_names = sorted(unit_groups.keys())
    U = len(unit_names)
    print(f"  U={U} (excludeOther, MIN_N={MIN_N})", flush=True)
    print(f"  units: {[short(u) for u in unit_names]}", flush=True)
    sizes = {short(u): len(unit_groups[u]) for u in unit_names}
    print(f"  unit sizes: {sizes}", flush=True)
    assert U >= 2, "need >=2 units"

    # ── prototypes (raw + resid) ────────────────────────────────────────────
    V_raw, V_resid = build_prototypes(delta, unit_groups, unit_names, n_layers, d_hid)
    lbls = [short(u) for u in unit_names]

    SAMPLE_LAYERS = [3, n_layers // 2, n_layers - 6]
    for tag, V in [("raw", V_raw), ("resid", V_resid)]:
        for l in SAMPLE_LAYERS:
            fig_heatmap(layer_cos(V[l]), lbls,
                        f"Unit-prototype cos  layer {l}  [{tag}]",
                        FIG_DIR / f"unit_sim_layer_{tag}_L{l}.png", annot=(U <= 30))

    S_raw   = agg_cos(V_raw)
    S_resid = agg_cos(V_resid)
    fig_heatmap(S_raw,   lbls, "Aggregated unit-prototype cos (mean over layers) [raw / 과거]",
                FIG_DIR / "unit_sim_agg_raw.png",   annot=(U <= 30))
    fig_heatmap(S_resid, lbls, "Aggregated unit-prototype cos (mean over layers) [resid / 현재]",
                FIG_DIR / "unit_sim_agg_resid.png", annot=(U <= 30))

    sec_raw,   sn_r, sM_r, ln_r, lM_r, Z_r, sil_r, coph_r = make_section("raw ΔA (과거)",   S_raw,   unit_names)
    sec_resid, sn_d, sM_d, ln_d, lM_d, Z_d, sil_d, coph_d = make_section("resid ΔA (현재)", S_resid, unit_names)

    fig_heatmap(sM_r, sn_r, "Subject×Subject block avg cos [raw]",
                FIG_DIR/"unit_sim_block_subject_raw.png",   vmin=-0.5, vmax=1, annot=True)
    fig_heatmap(sM_d, sn_d, "Subject×Subject block avg cos [resid]",
                FIG_DIR/"unit_sim_block_subject_resid.png", vmin=-0.5, vmax=1, annot=True)
    fig_heatmap(lM_r, ln_r, "Level×Level block avg cos [raw]",
                FIG_DIR/"unit_sim_block_level_raw.png",     vmin=-0.5, vmax=1, annot=True)
    fig_heatmap(lM_d, ln_d, "Level×Level block avg cos [resid]",
                FIG_DIR/"unit_sim_block_level_resid.png",   vmin=-0.5, vmax=1, annot=True)

    fig_dendrogram(Z_r, unit_names, FIG_DIR/"unit_sim_dendrogram_raw.png",   "raw")
    fig_dendrogram(Z_d, unit_names, FIG_DIR/"unit_sim_dendrogram_resid.png", "resid")

    body = [
        "# POOLED Qwen3-8B Unit Prototype Similarity (raw=과거 / resid=현재)\n\n",
        "**Data:** OPSD reasoning_pivot pilot1+pilot2 (THINKING ΔA = dA_thinking), "
        f"model = **Qwen/Qwen3-8B**, pooled finite N = **{N}** "
        f"(raw loaded {ninfo['n_loaded']}, non-finite dropped {ninfo['n_nonfinite']}).\n\n",
        "## Method (REF: 4.6_Task2 nait_unit_similarity; mechanism unchanged)\n",
        "- UNIT = subject × level.\n",
        "- v_u^l = sign-calibrated PC1 of {ΔA_s^l : s∈u} (unit-norm, Eq.4).\n",
        "- Per-layer sim:  S_l[u,u'] = v_u^l · v_{u'}^l (= cos).\n",
        "- Aggregated:     S_agg[u,u'] = mean_l v_u^l · v_{u'}^l.\n",
        "- **raw (과거)**: prototypes on raw ΔA.  **resid (현재)**: after removing per-layer global PC1.\n",
        "- PC1 via torch.pca_lowrank(q=6).\n",
        f"- U={U}, L={n_layers}, N={N}, D={d_hid}, MIN_N={MIN_N}, excludeOther.\n",
        "\n## Clustering quantification (subject vs level)\n",
        "| version | silhouette(subject) | silhouette(level) | cophenetic |\n",
        "|---|---|---|---|\n",
        f"| raw   | {sil_r['subject']} | {sil_r['level']} | {coph_r:.3f} |\n",
        f"| resid | {sil_d['subject']} | {sil_d['level']} | {coph_d:.3f} |\n",
        "\nHigher silhouette under a label ⇒ units cluster primarily by that label.\n",
    ]
    body.append(sec_raw); body.append(sec_resid)
    body.append("\n## Figures\n"
                "- `unit_sim_layer_{raw,resid}_L*.png`\n"
                "- `unit_sim_agg_{raw,resid}.png`\n"
                "- `unit_sim_block_subject_{raw,resid}.png`\n"
                "- `unit_sim_block_level_{raw,resid}.png`\n"
                "- `unit_sim_dendrogram_{raw,resid}.png`\n")
    REPORT.write_text("".join(body), encoding="utf-8")
    print(f"\n  [REPORT] {REPORT}")
    print(f"Done. (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
