"""
supervised_direction.py
========================
Phase A — Supervised direction (LDA) + layer-window evaluation
on top of the cached ΔA tensor produced by direction_calibrated.py
(`delta_cache.npy` of shape (N=2666, L=36, D=12288), float32).

Why
---
Phase 0 (direction_calibrated.py) showed:
 * unsupervised PC1 (with sign calibration) aligns mostly with the **level / difficulty**
   axis (max |ρ(score, level)| = 0.808 at layer 21), and
 * **subject silhouette is negative** on PC1, yet a PCA-256 + logistic probe achieves
   macro-F1 ≈ 0.74 at layer 14 → subject information IS in the activations,
   but PC1 is not the right direction to view it.

This script extracts *supervised* directions per layer:
  1. **Subject LDA** (multi-class, 8 subjects → up to 7 LDA dimensions, "subject axis").
  2. **Level LDA-1** (binary: level ≤ median vs > median → 1-D "difficulty axis").
  3. **Level ridge direction** (continuous regression, "difficulty axis" v2).
  4. Compare each supervised direction with calibrated PC1 (cosine, ρ).
  5. **Layer-window decision**:
       - subject window  : layers maximising LDA macro-F1
       - level   window  : layers maximising ridge R² (and |ρ| of LDA-1)
       - report top-3 layers for each axis.

All metrics are 5-fold cross-validated.  Pure CPU, ~5-10 min.

Inputs
------
  outputs/delta_cache.npy       (N, L, D)  float32
  outputs/delta_cache_meta.parquet         id / subject / level / pass_rate

Outputs
-------
  outputs/lda_directions.npz                W_subj[L,D,7], v_lvl_lda[L,D], v_lvl_ridge[L,D]
  outputs/supervised_per_layer.csv          per-layer F1 / R² / cos / ρ
  outputs/REPORT_supervised.md              human-readable summary
  outputs/plots/layer_window_compare.png    F1(subj) & R²(level) vs layer
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

BASE = Path("/scratch/lami2026/personal/jimin_2782")
NAIT_DIR = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/nait"
OUT_DIR  = NAIT_DIR / "outputs"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CACHE = OUT_DIR / "delta_cache.npy"
META  = OUT_DIR / "delta_cache_meta.parquet"
DIRS_PHASE0 = OUT_DIR / "directions.npz"

PROBE_K = 256          # PCA dim for fast LDA/ridge fitting per layer
N_FOLDS = 5
SEED    = 0


def main() -> None:
    t0 = time.time()
    # ── load ────────────────────────────────────────────────────────────────
    delta = np.load(CACHE, mmap_mode="r")           # (N, L, D)
    meta  = pd.read_parquet(META).reset_index(drop=True)
    meta["id"] = meta["id"].astype(str)
    N, L, D = delta.shape
    print(f"[load] ΔA={delta.shape}  meta(cols={list(meta.columns)})={meta.shape}")

    # delta_cache_meta.parquet only stores `id` in row order — join subject/level
    # from the shifts_metadata.jsonl, and pass_rate from Track-A parquet.
    import json
    SHIFT_DIR = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts"
    PASS_PARQ = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet"
    rows = []
    with open(SHIFT_DIR / "shifts_metadata.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") not in {"ok", "ok (skipped)", "completed"}:
                continue
            rows.append({"id": str(r["id"]),
                         "subject": r.get("subject"),
                         "level":   r.get("level")})
    sm = pd.DataFrame(rows).drop_duplicates(subset="id")
    pdf = pd.read_parquet(PASS_PARQ)
    pdf["id"] = pdf["sample_id"].astype(str)
    sm = sm.merge(pdf[["id", "pass_rate"]], on="id", how="left")
    meta = meta.merge(sm, on="id", how="left")
    assert meta["subject"].notna().all(), "subject missing for some ids"
    assert meta["level"].notna().all(),   "level missing for some ids"
    print(f"[meta] after join: {meta.shape}  cols={list(meta.columns)}")

    # Phase-0 calibrated PC1 directions, for cos similarity comparison
    pc1 = np.load(DIRS_PHASE0)["v_cal"]              # (L, D) — Phase-0 calibrated PC1
    print(f"[load] PC1 calibrated shape = {pc1.shape}")

    subj_codes, subj_classes = pd.factorize(meta["subject"].astype(str))
    n_subj = len(subj_classes)
    level   = meta["level"].astype(float).values
    level_med = float(np.median(level))
    lvl_bin = (level > level_med).astype(int)         # binary difficulty
    print(f"[meta] subjects={list(subj_classes)}  level median={level_med}")

    # Storage
    W_subj       = np.zeros((L, D, min(n_subj - 1, 7)), dtype=np.float32)
    v_lvl_lda    = np.zeros((L, D), dtype=np.float32)
    v_lvl_ridge  = np.zeros((L, D), dtype=np.float32)
    rows = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    kf  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for l in range(L):
        t = time.time()
        X = delta[:, l, :].astype(np.float32)         # (N, D)

        # PCA-256 reduction is used only for CV F1 / R² estimation.  The final
        # LDA / ridge directions are still fit in the FULL D-dim space.
        pca = PCA(n_components=PROBE_K, svd_solver="randomized", random_state=SEED).fit(X)
        Xp = pca.transform(X)                         # (N, 256)

        # ── (1) Subject LDA ────────────────────────────────────────────────
        f1_folds = []
        for tr, va in skf.split(Xp, subj_codes):
            lda_cv = LinearDiscriminantAnalysis(solver="svd")
            lda_cv.fit(Xp[tr], subj_codes[tr])
            f1_folds.append(f1_score(subj_codes[va], lda_cv.predict(Xp[va]),
                                     average="macro"))
        subj_f1 = float(np.mean(f1_folds))

        # Fit final LDA on full D and store axes (back-projected through PCA basis).
        lda_full = LinearDiscriminantAnalysis(solver="svd").fit(Xp, subj_codes)
        # lda_full.scalings_ has shape (PROBE_K, n_components).  Back-project to D.
        # x_red = X @ pca.components_.T   ⇒  direction in D = pca.components_.T @ scaling
        scal = lda_full.scalings_[:, :W_subj.shape[2]]   # (256, 7)
        Wd = pca.components_.T @ scal                    # (D, 7)
        Wd /= (np.linalg.norm(Wd, axis=0, keepdims=True) + 1e-12)
        W_subj[l] = Wd.astype(np.float32)

        # ── (2) Level LDA-1 (binary) ───────────────────────────────────────
        f1_lvl_folds, rho_lvl_folds = [], []
        for tr, va in skf.split(Xp, lvl_bin):
            lda_b = LinearDiscriminantAnalysis(solver="svd")
            lda_b.fit(Xp[tr], lvl_bin[tr])
            pred = lda_b.predict(Xp[va])
            f1_lvl_folds.append(f1_score(lvl_bin[va], pred, average="macro"))
            score_va = Xp[va] @ lda_b.scalings_[:, 0]
            rho_lvl_folds.append(spearmanr(score_va, level[va]).statistic)
        lvl_lda_f1  = float(np.mean(f1_lvl_folds))
        lvl_lda_rho = float(np.mean(rho_lvl_folds))

        lda_b_full = LinearDiscriminantAnalysis(solver="svd").fit(Xp, lvl_bin)
        vd = pca.components_.T @ lda_b_full.scalings_[:, 0]
        vd /= (np.linalg.norm(vd) + 1e-12)
        v_lvl_lda[l] = vd.astype(np.float32)

        # ── (3) Level ridge (continuous regression) ───────────────────────
        r2_folds = []
        for tr, va in kf.split(Xp):
            rg = Ridge(alpha=1.0, random_state=SEED).fit(Xp[tr], level[tr])
            r2_folds.append(r2_score(level[va], rg.predict(Xp[va])))
        lvl_ridge_r2 = float(np.mean(r2_folds))

        rg_full = Ridge(alpha=1.0, random_state=SEED).fit(Xp, level)
        vd2 = pca.components_.T @ rg_full.coef_
        vd2 /= (np.linalg.norm(vd2) + 1e-12)
        v_lvl_ridge[l] = vd2.astype(np.float32)

        # ── (4) Cosine sim with Phase-0 PC1 ───────────────────────────────
        cos_pc1_lvl_lda   = float(np.abs(pc1[l] @ v_lvl_lda[l]))
        cos_pc1_lvl_ridge = float(np.abs(pc1[l] @ v_lvl_ridge[l]))
        # mean abs cos between PC1 and each subject LDA axis (7 axes)
        cos_pc1_subj      = float(np.mean(np.abs(pc1[l] @ W_subj[l])))

        rows.append(dict(
            layer=l,
            subj_F1=subj_f1,
            lvl_lda_F1=lvl_lda_f1,
            lvl_lda_rho=lvl_lda_rho,
            lvl_ridge_R2=lvl_ridge_r2,
            cos_pc1_lvl_lda=cos_pc1_lvl_lda,
            cos_pc1_lvl_ridge=cos_pc1_lvl_ridge,
            cos_pc1_subj_mean=cos_pc1_subj,
        ))
        print(f"  L{l:02d}  subjF1={subj_f1:.3f}  lvlLDA_F1={lvl_lda_f1:.3f} "
              f"lvlLDA_ρ={lvl_lda_rho:+.3f}  lvlRidge_R²={lvl_ridge_r2:+.3f} "
              f"cos(PC1,lvl_ridge)={cos_pc1_lvl_ridge:.3f}  ({time.time()-t:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "supervised_per_layer.csv", index=False)
    np.savez_compressed(OUT_DIR / "lda_directions.npz",
                        W_subj=W_subj, v_lvl_lda=v_lvl_lda, v_lvl_ridge=v_lvl_ridge,
                        subject_classes=np.array(list(subj_classes)),
                        level_median=np.float32(level_med))
    print(f"[save] supervised_per_layer.csv  +  lda_directions.npz")

    # ── plots ───────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(df["layer"], df["subj_F1"],       "o-", color="#1f77b4", label="subj macro-F1 (LDA)")
    ax1.plot(df["layer"], df["lvl_lda_F1"],    "s-", color="#9467bd", label="level LDA binary F1")
    ax2.plot(df["layer"], df["lvl_ridge_R2"],  "^-", color="#d62728", label="level ridge R²")
    ax2.plot(df["layer"], np.abs(df["lvl_lda_rho"]), "v--", color="#ff7f0e",
             label="|ρ(lvl_LDA, level)|")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("F1 (subject / level-binary)")
    ax2.set_ylabel("R² / |ρ|  (level)")
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(-0.1, 1.0)
    ax1.grid(alpha=0.3)
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center",
               ncol=2, fontsize=8)
    fig.suptitle("Phase A — Supervised direction quality per layer (Qwen3-8B, N=2666)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "layer_window_compare.png", dpi=140)
    plt.close(fig)

    # ── report ──────────────────────────────────────────────────────────────
    top_subj  = df.nlargest(5, "subj_F1")[["layer", "subj_F1"]].values.tolist()
    top_lvl_r2 = df.nlargest(5, "lvl_ridge_R2")[["layer", "lvl_ridge_R2"]].values.tolist()
    top_lvl_lda = df.iloc[np.argsort(-np.abs(df["lvl_lda_rho"].values))[:5]][
        ["layer", "lvl_lda_rho"]].values.tolist()

    lines = []
    lines.append("# Phase A — Supervised Direction Report\n")
    lines.append(f"- N samples : {N}")
    lines.append(f"- N layers  : {L}")
    lines.append(f"- D hidden  : {D}")
    lines.append(f"- PCA dim used for fitting : {PROBE_K}")
    lines.append(f"- CV folds  : {N_FOLDS}")
    lines.append(f"- Wall time : {(time.time()-t0)/60:.1f} min\n")

    lines.append("## Top layers — Subject (LDA macro-F1)\n")
    lines.append("| rank | layer | macro-F1 |")
    lines.append("|---:|---:|---:|")
    for i, (l, f1) in enumerate(top_subj, 1):
        lines.append(f"| {i} | {int(l)} | {f1:.3f} |")

    lines.append("\n## Top layers — Level (ridge R²)\n")
    lines.append("| rank | layer | R² |")
    lines.append("|---:|---:|---:|")
    for i, (l, r2) in enumerate(top_lvl_r2, 1):
        lines.append(f"| {i} | {int(l)} | {r2:+.3f} |")

    lines.append("\n## Top layers — Level (LDA-1 |ρ|)\n")
    lines.append("| rank | layer | ρ |")
    lines.append("|---:|---:|---:|")
    for i, (l, rho) in enumerate(top_lvl_lda, 1):
        lines.append(f"| {i} | {int(l)} | {rho:+.3f} |")

    # Layer-window recommendation
    subj_win = sorted(df.nlargest(5, "subj_F1")["layer"].astype(int).tolist())
    lvl_win  = sorted(df.nlargest(5, "lvl_ridge_R2")["layer"].astype(int).tolist())
    lines.append("\n## Recommended layer windows\n")
    lines.append(f"- **Subject window** (top-5 by macro-F1): `{subj_win}`")
    lines.append(f"- **Level window**   (top-5 by ridge R²): `{lvl_win}`\n")

    # Cosine sim summary with PC1
    cos_l = df.loc[df["lvl_ridge_R2"].idxmax()]
    lines.append("## Cosine similarity vs Phase-0 calibrated PC1 (at best-level layer)\n")
    lines.append(f"- layer = **{int(cos_l['layer'])}**")
    lines.append(f"- |cos(PC1, v_level_ridge)| = **{cos_l['cos_pc1_lvl_ridge']:.3f}**")
    lines.append(f"- |cos(PC1, v_level_LDA)|   = {cos_l['cos_pc1_lvl_lda']:.3f}")
    lines.append(f"- mean |cos(PC1, W_subject)| (7 axes) = {cos_l['cos_pc1_subj_mean']:.3f}\n")

    lines.append("## Files\n")
    lines.append("- `supervised_per_layer.csv` — per-layer metrics table")
    lines.append("- `lda_directions.npz`        — W_subj[L,D,7], v_lvl_lda[L,D], v_lvl_ridge[L,D]")
    lines.append("- `plots/layer_window_compare.png` — F1 / R² vs layer\n")
    lines.append("## Notes\n")
    lines.append("- LDA / ridge are fit on PCA-256 reduced features, then back-projected to D=12288.")
    lines.append("- Comparing the level-direction with calibrated PC1 quantifies how aligned the")
    lines.append("  unsupervised difficulty axis is with the *supervised* difficulty axis.")
    lines.append("- The subject window vs level window separation motivates the Phase-B feature design")
    lines.append("  (concat of subject-projected ΔA in subj_window, level-projected ΔA in lvl_window).")

    (OUT_DIR / "REPORT_supervised.md").write_text("\n".join(lines))
    print(f"[done] wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
