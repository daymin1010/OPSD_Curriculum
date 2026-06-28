"""
unit_similarity_qwen3_8b.py — Inter-unit prototype similarity (Qwen3-8B pilot 2,666)
====================================================================================
BASE  : OPSD_Curriculum/analysis_qwen3_8b  (data loading / paths / delta_cache reuse)
REF   : 4.6_Task2/activation/analysis/nait_unit_similarity.py + _nait_common.py
        (mechanism: sign-calibrated PC1 prototypes, residualize, block averages,
         dendrogram).

Two versions of the SAME mechanism (= "과거/현재 두 종류"):
  - raw   Δ𝒜  ("과거"):  prototypes on raw activation shifts
  - resid Δ𝒜  ("현재"):  prototypes after removing per-layer global PC1

Definitions (per unit u = subject × level):
  v_u^l = sign-calibrated PC1 of {Δ𝒜_s^l : s ∈ u}            (unit-norm, Eq.4)
  S_l[u,u']   = v_u^l · v_{u'}^l                              (= cos)
  S_agg[u,u'] = mean_l  v_u^l · v_{u'}^l

Speed: PC1 via torch.pca_lowrank (q=6) instead of full np.linalg.svd
       — numerically equivalent for the leading component, far faster at D=12288.
       Processes one layer at a time from the mmapped (N,L,D) cache → low RAM.

CPU only.  Expected ~8-15 min (cache hit).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
except ImportError as e:
    import sys; print(f"[ERROR] {e}"); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants  (BASE = OPSD_Curriculum)
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path("/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b")
SHIFT_DIR = BASE / "activation/outputs/shifts"
META_JSONL = SHIFT_DIR / "shifts_metadata.jsonl"
CACHE_PATH = BASE / "nait/outputs/delta_cache.npy"          # (N, L, D) float32
CACHE_META = BASE / "nait/outputs/delta_cache_meta.parquet" # ordered ids

OUT_DIR = BASE / "outputs/unit_similarity"
FIG_DIR = OUT_DIR / "figures"
REPORT  = OUT_DIR / "unit_similarity_report.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 36
D_HID    = 12288
MIN_N    = 4               # min samples per unit (matches 4.6)
PCA_Q    = 6               # lowrank rank for PC1
EXCLUDE_SUBJECTS = {"Other"}
SAMPLE_LAYERS = [3, 18, 30]   # early / mid / late (36 layers)

torch.set_num_threads(max(1, os.cpu_count() or 1))


# ─────────────────────────────────────────────────────────────────────────────
# PC1 (sign-calibrated) via torch.pca_lowrank
# ─────────────────────────────────────────────────────────────────────────────
def pc1_with_sign(X: np.ndarray) -> np.ndarray:
    """
    PC1 (unit-norm) + sign calibration (paper Eq.4):
        v ← -v if mean_diff · v < 0
    X: (n, D) float32.  Uses torch.pca_lowrank (centered) for the leading PC.
    """
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
    _, _, V = torch.pca_lowrank(t, q=PCA_Q, center=True, niter=4)
    v = V[:, 0].numpy().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (REF: direction_calibrated.load_metadata_df ; reuse delta_cache)
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata_indexed() -> pd.DataFrame:
    """Load metadata keyed by id, aligned to delta_cache row order."""
    rows = {}
    with open(META_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") not in {"ok", "ok (skipped)", "completed"}:
                continue
            rows[str(r["id"])] = r
    meta_by_id = pd.DataFrame(list(rows.values()))
    meta_by_id["id"] = meta_by_id["id"].astype(str)
    meta_by_id = meta_by_id.set_index("id")

    # canonical order = delta_cache_meta (the row order of delta_cache.npy)
    order = pd.read_parquet(CACHE_META)
    order["id"] = order["id"].astype(str)
    df = order[["id"]].copy()
    df["subject"] = df["id"].map(meta_by_id["subject"]).astype(str)
    df["level"]   = pd.to_numeric(df["id"].map(meta_by_id["level"]), errors="coerce").fillna(0).astype(int)
    assert df["subject"].notna().all(), "subject mapping failed"
    df["unit"] = df["subject"] + "_L" + df["level"].astype(str)
    return df


def unit_positions(df: pd.DataFrame):
    """{unit: [row positions]} (excluding EXCLUDE_SUBJECTS, MIN_N filter), sorted names."""
    groups = {}
    for pos, row in enumerate(df.itertuples(index=False)):
        if row.subject in EXCLUDE_SUBJECTS:
            continue
        groups.setdefault(row.unit, []).append(pos)
    groups = {u: p for u, p in groups.items() if len(p) >= MIN_N}
    return groups, sorted(groups.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Build prototypes  (one layer at a time; raw + resid simultaneously)
# ─────────────────────────────────────────────────────────────────────────────
def build_prototypes(delta, unit_groups, unit_names):
    """
    delta: mmapped (N, L, D).  Returns V_raw, V_resid each (L, U, D) float32.
    Per layer: load (N,D), compute global PC1 → residualize, then per-unit PC1.
    """
    U = len(unit_names)
    V_raw   = np.zeros((N_LAYERS, U, D_HID), dtype=np.float32)
    V_resid = np.zeros((N_LAYERS, U, D_HID), dtype=np.float32)
    t0 = time.time()
    for l in range(N_LAYERS):
        Xl = np.ascontiguousarray(delta[:, l, :], dtype=np.float32)   # (N, D)
        g = global_pc1(Xl)                                            # (D,)
        coeff = Xl @ g                                                # (N,)
        Xl_resid = Xl - np.outer(coeff, g)                           # remove global PC1
        for u_idx, u in enumerate(unit_names):
            idx = unit_groups[u]
            V_raw[l, u_idx]   = pc1_with_sign(Xl[idx])
            V_resid[l, u_idx] = pc1_with_sign(Xl_resid[idx])
        if (l + 1) % 6 == 0 or l == N_LAYERS - 1:
            print(f"  [proto] layer {l+1}/{N_LAYERS}  ({time.time()-t0:.1f}s)", flush=True)
    return V_raw, V_resid


# ─────────────────────────────────────────────────────────────────────────────
# Similarity matrices  (REF: nait_unit_similarity.layer_cos / agg_cos)
# ─────────────────────────────────────────────────────────────────────────────
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


def fig_dendrogram(S_agg, unit_names, path, tag):
    D = np.clip(1.0 - S_agg, 0, 2)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    try:
        Z = linkage(squareform(D, checks=False), method="average")
    except Exception as e:
        print(f"  [dendro skip] {e}"); return
    fig, ax = plt.subplots(figsize=(max(14, len(unit_names)*0.35), 6))
    dendrogram(Z, labels=[short(u) for u in unit_names], leaf_rotation=75,
               leaf_font_size=7, ax=ax, color_threshold=0.5 * max(Z[:, 2]))
    ax.set_ylabel("1 − cos_sim (avg over layers)")
    ax.set_title(f"Unit Prototype Dendrogram  [{tag}]")
    plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()


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

    lines = [f"\n## {tag}\n\n",
             f"- **Subject block** (avg cos): within = **{within_subj:.3f}**, across = **{across_subj:.3f}**, ratio = **{within_subj/(abs(across_subj)+1e-9):.2f}x**\n",
             f"- **Level   block** (avg cos): within = **{within_lvl:.3f}**, across = **{across_lvl:.3f}**, ratio = **{within_lvl/(abs(across_lvl)+1e-9):.2f}x**\n",
             "\n### Top 15 most similar unit pairs (aggregated cos)\n\n| cos | u1 | u2 | same subj? | same lvl? |\n|---|---|---|---|---|\n"]
    for c, a, b in pairs[:15]:
        lines.append(f"| {c:.3f} | {short(a)} | {short(b)} | {'✓' if subj_of(a)==subj_of(b) else '·'} | {'✓' if lvl_of(a)==lvl_of(b) else '·'} |\n")
    lines.append("\n### Bottom 10 most DIS-similar pairs\n\n| cos | u1 | u2 |\n|---|---|---|\n")
    for c, a, b in pairs[-10:][::-1]:
        lines.append(f"| {c:.3f} | {short(a)} | {short(b)} |\n")
    return "".join(lines), subj_n, subj_M, lvl_n, lvl_M


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Qwen3-8B Unit Prototype Similarity  (raw=과거 / resid=현재)")
    print("=" * 72, flush=True)

    df = load_metadata_indexed()
    delta = np.load(CACHE_PATH, mmap_mode="r")        # (N, L, D)
    assert delta.shape[0] == len(df), f"cache N={delta.shape[0]} vs df={len(df)}"
    assert delta.shape[1] == N_LAYERS and delta.shape[2] == D_HID, f"cache shape {delta.shape}"

    unit_groups, unit_names = unit_positions(df)
    U = len(unit_names)
    print(f"  N={delta.shape[0]}  L={N_LAYERS}  D={D_HID}  U={U} (excludeOther, MIN_N={MIN_N})", flush=True)
    print(f"  units: {[short(u) for u in unit_names]}", flush=True)

    V_raw, V_resid = build_prototypes(delta, unit_groups, unit_names)

    lbls = [short(u) for u in unit_names]

    # per-sample-layer heatmaps
    for tag, V in [("raw", V_raw), ("resid", V_resid)]:
        for l in SAMPLE_LAYERS:
            fig_heatmap(layer_cos(V[l]), lbls, f"Unit-prototype cos  layer {l}  [{tag}]",
                        FIG_DIR / f"unit_sim_layer_{tag}_L{l}.png", annot=(U <= 30))

    # aggregated
    S_raw   = agg_cos(V_raw)
    S_resid = agg_cos(V_resid)
    fig_heatmap(S_raw,   lbls, "Aggregated unit-prototype cos (mean over layers) [raw / 과거]",
                FIG_DIR / "unit_sim_agg_raw.png",   annot=(U <= 30))
    fig_heatmap(S_resid, lbls, "Aggregated unit-prototype cos (mean over layers) [resid / 현재]",
                FIG_DIR / "unit_sim_agg_resid.png", annot=(U <= 30))

    sec_raw,   sn_r, sM_r, ln_r, lM_r = make_section("raw Δ𝒜  (과거)",   S_raw,   unit_names)
    sec_resid, sn_d, sM_d, ln_d, lM_d = make_section("resid Δ𝒜 (현재)", S_resid, unit_names)

    fig_heatmap(sM_r, sn_r, "Subject×Subject block avg cos [raw]",
                FIG_DIR/"unit_sim_block_subject_raw.png",   vmin=-0.5, vmax=1, annot=True)
    fig_heatmap(sM_d, sn_d, "Subject×Subject block avg cos [resid]",
                FIG_DIR/"unit_sim_block_subject_resid.png", vmin=-0.5, vmax=1, annot=True)
    fig_heatmap(lM_r, ln_r, "Level×Level block avg cos [raw]",
                FIG_DIR/"unit_sim_block_level_raw.png",     vmin=-0.5, vmax=1, annot=True)
    fig_heatmap(lM_d, ln_d, "Level×Level block avg cos [resid]",
                FIG_DIR/"unit_sim_block_level_resid.png",   vmin=-0.5, vmax=1, annot=True)

    fig_dendrogram(S_raw,   unit_names, FIG_DIR/"unit_sim_dendrogram_raw.png",   "raw")
    fig_dendrogram(S_resid, unit_names, FIG_DIR/"unit_sim_dendrogram_resid.png", "resid")

    body = [
        "# Qwen3-8B Unit Prototype Similarity (raw=과거 / resid=현재)\n\n",
        "## Method (REF: 4.6_Task2 nait_unit_similarity)\n",
        "- v_u^l = sign-calibrated PC1 of {Δ𝒜_s^l : s∈u} (unit-norm, Eq.4).\n",
        "- Per-layer sim:  S_l[u,u'] = v_u^l · v_{u'}^l  (= cos).\n",
        "- Aggregated sim: S_agg[u,u'] = mean_l v_u^l · v_{u'}^l.\n",
        "- **raw (과거)**: prototypes on raw Δ𝒜.  **resid (현재)**: after removing per-layer global PC1.\n",
        "- PC1 via torch.pca_lowrank(q=6) for speed (equivalent leading component).\n",
        f"- U={U}, L={N_LAYERS}, N={delta.shape[0]}, D={D_HID}.\n",
    ]
    body.append(sec_raw); body.append(sec_resid)
    body.append("\n## Figures\n"
                "- `unit_sim_layer_{raw,resid}_L{3,18,30}.png`\n"
                "- `unit_sim_agg_{raw,resid}.png`\n"
                "- `unit_sim_block_subject_{raw,resid}.png`\n"
                "- `unit_sim_block_level_{raw,resid}.png`\n"
                "- `unit_sim_dendrogram_{raw,resid}.png`\n")
    REPORT.write_text("".join(body), encoding="utf-8")
    print(f"\n  [REPORT] {REPORT}\nDone.", flush=True)


if __name__ == "__main__":
    main()
