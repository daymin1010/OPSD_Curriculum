"""
analyze_critical.py
===================
Critical-analysis baseline: prompt-only activation vs ΔA.

Research question
-----------------
Phase A showed that subject identity is highly readable from ΔA
(macro-F1 = 0.779 at L14 with PCA-256 + LDA). But ΔA is
   ΔA = h(t_K) − h(t_1)
i.e. it depends on BOTH the prompt and the generated continuation. Maybe
the "subject signal" is just *prompt keywords* the model has already
encoded at t_1 (the last prompt token), and ΔA inherits this naturally.

This script answers the question with a like-for-like baseline:

  PROMPT-ONLY activation A_prompt(l) := MLP-down-proj input at t_1
                                        (the last prompt token, before generation)

We probe subject / level / pass_rate from A_prompt per layer with the
SAME PCA-256 + LDA / Ridge pipeline used in Phase A, then compare:

    Phase A (ΔA)        ─┐
                          ├── per-layer F1_subject, R²_level, R²_pass
    This (A_prompt)     ─┘

Interpretation rules of thumb
-----------------------------
  • If F1_subject(A_prompt) ≈ F1_subject(ΔA)  → ΔA subject signal is
                                                trivially inherited from the prompt
  • If F1_subject(ΔA) ≫ F1_subject(A_prompt)  → ΔA actually carries
                                                model-internal reasoning info
                                                beyond what's in the prompt
  • For level/pass: A_prompt cannot see truncation or generation outcome,
                    so these scores measure prompt-level difficulty cues
                    (problem length, vocabulary).

Inputs
------
  outputs/prompt_act/{id}.pt        : (36, 12288) bfloat16  ← Step 1 output
  outputs/prompt_act/prompt_activation_metadata.jsonl
  Phase A: outputs/supervised_per_layer.csv

Outputs
-------
  outputs/critical/prompt_act_cache.npy           (N, L, D) float32  ~4.7 GB
  outputs/critical/prompt_act_meta.parquet        id-aligned
  outputs/critical/prompt_per_layer.csv           same columns as Phase A's per-layer csv
  outputs/critical/critical_compare.csv           Phase A vs A_prompt per-layer joined
  outputs/critical/REPORT_critical.md             human report (TL;DR + tables)
  outputs/critical/plots/{f1_subj_compare,r2_level_compare,r2_pass_compare}.png

Wall: ~10-15 min on CPU (LDA over 36 layers × 5-fold CV).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/scratch/lami2026/personal/jimin_2782")
NAIT      = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/nait"
PROMPT_DIR = NAIT / "outputs/prompt_act"
OUT_DIR    = NAIT / "outputs/critical"
PLOT_DIR   = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PHASE_A_CSV = NAIT / "outputs/supervised_per_layer.csv"
PASS_PARQ   = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet"

CACHE   = OUT_DIR / "prompt_act_cache.npy"
META    = OUT_DIR / "prompt_act_meta.parquet"

PROBE_K  = 256
N_FOLDS  = 5
SEED     = 0


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build / load A_prompt cache
# ══════════════════════════════════════════════════════════════════════════════

def build_cache() -> tuple[np.ndarray, pd.DataFrame]:
    """Stack all prompt_act/{id}.pt into a single (N, L, D) float32 array."""
    if CACHE.exists() and META.exists():
        print(f"[load] cache hit: {CACHE}")
        arr = np.load(CACHE, mmap_mode="r")
        meta = pd.read_parquet(META).reset_index(drop=True)
        meta["id"] = meta["id"].astype(str)
        print(f"[load] arr={arr.shape}  meta={meta.shape}")
        return arr, meta

    pt_files = sorted(PROMPT_DIR.glob("*.pt"))
    print(f"[build] found {len(pt_files)} .pt files in {PROMPT_DIR}")
    if not pt_files:
        sys.exit("no .pt files — did Step 1 pilot extraction finish?")

    # peek shape
    d0 = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    L, D = d0["prompt_act"].shape
    print(f"[build] per-sample shape: ({L}, {D})")

    N = len(pt_files)
    arr = np.empty((N, L, D), dtype=np.float32)
    meta_rows = []

    t0 = time.time()
    for i, pf in enumerate(pt_files):
        d = torch.load(pf, map_location="cpu", weights_only=False)
        arr[i] = d["prompt_act"].float().numpy()
        meta_rows.append({
            "id":      str(d["id"]),
            "subject": d.get("subject", ""),
            "level":   int(d.get("level", -1)),
            "prompt_len": int(d.get("prompt_len", -1)),
        })
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{N}] elapsed={time.time()-t0:.0f}s")
    print(f"[build] done | wall={time.time()-t0:.0f}s")

    meta = pd.DataFrame(meta_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, arr)
    meta[["id", "subject", "level", "prompt_len"]].to_parquet(META, index=False)
    print(f"[save] {CACHE}  ({arr.nbytes/1e9:.2f} GB)")
    print(f"[save] {META}")
    return arr, meta


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Per-layer probe (mirror Phase A)
# ══════════════════════════════════════════════════════════════════════════════

def per_layer_probe(arr: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    N, L, D = arr.shape

    # attach pass_rate
    pdf = pd.read_parquet(PASS_PARQ)[["sample_id", "pass_rate"]]
    pdf["id"] = pdf["sample_id"].astype(str)
    meta = meta.merge(pdf[["id", "pass_rate"]], on="id", how="left")
    print(f"[probe] meta after merge: {meta.shape}  pass_NA={meta['pass_rate'].isna().sum()}")
    assert meta["subject"].notna().all() and (meta["subject"] != "").all(), \
        "missing subject in some rows"
    assert meta["level"].notna().all(), "missing level"

    # encode labels
    subj_classes = sorted(meta["subject"].unique())
    subj_to_idx  = {s: i for i, s in enumerate(subj_classes)}
    y_subj = meta["subject"].map(subj_to_idx).values
    y_lvl  = meta["level"].values.astype(np.float32)
    y_pass = meta["pass_rate"].values.astype(np.float32)
    lvl_med = float(np.median(y_lvl))
    y_lvl_bin = (y_lvl > lvl_med).astype(np.int64)

    print(f"[probe] subjects = {subj_classes}")
    print(f"[probe] level median = {lvl_med}  (used for binary LDA-1)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    kf  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    records = []
    t0 = time.time()
    for l in range(L):
        Xl = arr[:, l, :].astype(np.float32)  # (N, D)
        # PCA-256
        pca = PCA(n_components=PROBE_K, random_state=SEED)
        Xp  = pca.fit_transform(Xl)            # (N, 256)

        # Subject — multiclass LDA
        f1s = []
        for tr, te in skf.split(Xp, y_subj):
            try:
                lda = LinearDiscriminantAnalysis()
                lda.fit(Xp[tr], y_subj[tr])
                pred = lda.predict(Xp[te])
                f1s.append(f1_score(y_subj[te], pred, average="macro"))
            except Exception as e:
                warnings.warn(f"L{l} subj LDA failed: {e}")
                f1s.append(np.nan)
        subj_f1 = float(np.nanmean(f1s))

        # Level — binary LDA-1
        try:
            lda1 = LinearDiscriminantAnalysis(n_components=1)
            lda1.fit(Xp, y_lvl_bin)
            proj = lda1.transform(Xp).ravel()
            rho, _ = spearmanr(proj, y_lvl)
            lvl_lda_rho = float(abs(rho))
            lf1s = []
            for tr, te in skf.split(Xp, y_lvl_bin):
                lda1b = LinearDiscriminantAnalysis()
                lda1b.fit(Xp[tr], y_lvl_bin[tr])
                lf1s.append(f1_score(y_lvl_bin[te], lda1b.predict(Xp[te]), average="macro"))
            lvl_lda_f1 = float(np.nanmean(lf1s))
        except Exception as e:
            warnings.warn(f"L{l} lvl LDA failed: {e}")
            lvl_lda_f1 = lvl_lda_rho = np.nan

        # Level — Ridge regression
        try:
            r2s = []
            for tr, te in kf.split(Xp):
                ridge = Ridge(alpha=1.0, random_state=SEED)
                ridge.fit(Xp[tr], y_lvl[tr])
                r2s.append(r2_score(y_lvl[te], ridge.predict(Xp[te])))
            lvl_ridge_r2 = float(np.nanmean(r2s))
        except Exception as e:
            warnings.warn(f"L{l} lvl ridge failed: {e}")
            lvl_ridge_r2 = np.nan

        # Pass — Ridge regression
        try:
            r2s = []
            mask = ~np.isnan(y_pass)
            for tr, te in kf.split(Xp[mask]):
                Xm = Xp[mask]; ym = y_pass[mask]
                ridge = Ridge(alpha=1.0, random_state=SEED)
                ridge.fit(Xm[tr], ym[tr])
                r2s.append(r2_score(ym[te], ridge.predict(Xm[te])))
            pass_ridge_r2 = float(np.nanmean(r2s))
        except Exception as e:
            warnings.warn(f"L{l} pass ridge failed: {e}")
            pass_ridge_r2 = np.nan

        records.append({
            "layer":         l,
            "subj_F1":       subj_f1,
            "lvl_lda_F1":    lvl_lda_f1,
            "lvl_lda_rho":   lvl_lda_rho,
            "lvl_ridge_R2":  lvl_ridge_r2,
            "pass_ridge_R2": pass_ridge_r2,
        })
        print(f"  L{l:02d}  subjF1={subj_f1:.3f}  "
              f"lvlF1={lvl_lda_f1:.3f}  |ρ|lvl={lvl_lda_rho:.3f}  "
              f"R²lvl={lvl_ridge_r2:+.3f}  R²pass={pass_ridge_r2:+.3f}  "
              f"({time.time()-t0:.0f}s)")
    df = pd.DataFrame(records)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compare with Phase A
# ══════════════════════════════════════════════════════════════════════════════

def compare_and_plot(prompt_df: pd.DataFrame) -> pd.DataFrame:
    if not PHASE_A_CSV.exists():
        warnings.warn(f"[compare] Phase A csv not found: {PHASE_A_CSV}")
        return prompt_df

    a = pd.read_csv(PHASE_A_CSV)
    # keep canonical metric columns
    keep = ["layer", "subj_F1", "lvl_lda_F1", "lvl_lda_rho", "lvl_ridge_R2"]
    keep_existing = [c for c in keep if c in a.columns]
    a = a[keep_existing].rename(columns={c: f"deltaA_{c}" for c in keep_existing if c != "layer"})
    p = prompt_df.rename(columns={c: f"prompt_{c}" for c in prompt_df.columns if c != "layer"})

    cmp = a.merge(p, on="layer", how="outer").sort_values("layer")
    cmp.to_csv(OUT_DIR / "critical_compare.csv", index=False)
    print(f"[compare] saved {OUT_DIR/'critical_compare.csv'}")

    # plots
    def plot_pair(col_d, col_p, title, fname):
        plt.figure(figsize=(9, 4.5))
        if col_d in cmp.columns:
            plt.plot(cmp["layer"], cmp[col_d], "-o", label="ΔA  (Phase A)", color="C0")
        if col_p in cmp.columns:
            plt.plot(cmp["layer"], cmp[col_p], "-s", label="A_prompt (this)", color="C3")
        plt.xlabel("layer"); plt.ylabel(title)
        plt.title(f"{title}: ΔA vs prompt-only activation")
        plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / fname, dpi=120)
        plt.close()

    plot_pair("deltaA_subj_F1",      "prompt_subj_F1",      "subject macro-F1", "f1_subj_compare.png")
    plot_pair("deltaA_lvl_ridge_R2", "prompt_lvl_ridge_R2", "level Ridge R²",   "r2_level_compare.png")
    plot_pair("deltaA_lvl_lda_rho",  "prompt_lvl_lda_rho",  "|ρ|(LDA-1, level)", "rho_level_compare.png")
    if "prompt_pass_ridge_R2" in p.columns:
        plt.figure(figsize=(9, 4.5))
        plt.plot(cmp["layer"], p.set_index("layer").reindex(cmp["layer"])["prompt_pass_ridge_R2"],
                 "-s", label="A_prompt", color="C3")
        plt.xlabel("layer"); plt.ylabel("pass_rate Ridge R²")
        plt.title("pass_rate Ridge R² from prompt-only activation")
        plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "r2_pass_prompt.png", dpi=120)
        plt.close()

    return cmp


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Report
# ══════════════════════════════════════════════════════════════════════════════

def write_report(prompt_df: pd.DataFrame, cmp_df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Critical Analysis — prompt-only activation vs ΔA\n")
    lines.append("## Setup\n")
    lines.append("- Source: `outputs/prompt_act/{id}.pt`, 36 layers × 12288 dims (bfloat16, "
                 "MLP down_proj input @ last prompt token, no generation).\n")
    lines.append(f"- Pilot: N={len(prompt_df)} layers from prompt cache; reusing 8 MATH subjects.\n")
    lines.append("- Probe pipeline identical to Phase A: PCA-256 → 5-fold CV LDA/Ridge.\n\n")

    lines.append("## Headline numbers (prompt-only activation)\n")
    pi_subj  = prompt_df["subj_F1"].idxmax()
    pi_level = prompt_df["lvl_ridge_R2"].idxmax()
    pi_pass  = prompt_df["pass_ridge_R2"].idxmax()
    pi_rho   = prompt_df["lvl_lda_rho"].idxmax()
    lines.append("| signal | best layer | value |\n|---|---|---|\n")
    lines.append(f"| subject macro-F1                | L{int(prompt_df.loc[pi_subj, 'layer'])} "
                 f"| **{prompt_df.loc[pi_subj, 'subj_F1']:.3f}** |\n")
    lines.append(f"| level Ridge R²                  | L{int(prompt_df.loc[pi_level,'layer'])} "
                 f"| **{prompt_df.loc[pi_level,'lvl_ridge_R2']:+.3f}** |\n")
    lines.append(f"| level LDA-1 \\|ρ\\| vs level     | L{int(prompt_df.loc[pi_rho,  'layer'])} "
                 f"| **{prompt_df.loc[pi_rho,  'lvl_lda_rho']:.3f}** |\n")
    lines.append(f"| pass_rate Ridge R²              | L{int(prompt_df.loc[pi_pass, 'layer'])} "
                 f"| **{prompt_df.loc[pi_pass, 'pass_ridge_R2']:+.3f}** |\n\n")

    if "deltaA_subj_F1" in cmp_df.columns:
        lines.append("## ΔA vs A_prompt — head-to-head\n")
        head = cmp_df[["layer", "deltaA_subj_F1", "prompt_subj_F1",
                       "deltaA_lvl_ridge_R2", "prompt_lvl_ridge_R2",
                       "deltaA_lvl_lda_rho",  "prompt_lvl_lda_rho"]].copy()
        # mid block & late layers
        peeks = [11, 14, 17, 18, 21, 28]
        head = head[head["layer"].isin(peeks)]
        lines.append(head.to_markdown(index=False, floatfmt=".3f"))
        lines.append("\n\n")

        # global summary
        d_subj_best = cmp_df["deltaA_subj_F1"].max() if "deltaA_subj_F1" in cmp_df else np.nan
        p_subj_best = cmp_df["prompt_subj_F1"].max() if "prompt_subj_F1" in cmp_df else np.nan
        d_lvl_best  = cmp_df["deltaA_lvl_ridge_R2"].max() if "deltaA_lvl_ridge_R2" in cmp_df else np.nan
        p_lvl_best  = cmp_df["prompt_lvl_ridge_R2"].max() if "prompt_lvl_ridge_R2" in cmp_df else np.nan
        lines.append("### Best-layer comparison\n")
        lines.append("| signal | ΔA best | A_prompt best | gap |\n|---|---:|---:|---:|\n")
        lines.append(f"| subject F1 | {d_subj_best:.3f} | {p_subj_best:.3f} "
                     f"| {d_subj_best - p_subj_best:+.3f} |\n")
        lines.append(f"| level R²   | {d_lvl_best:+.3f} | {p_lvl_best:+.3f} "
                     f"| {d_lvl_best - p_lvl_best:+.3f} |\n\n")

        lines.append("### Interpretation guide\n")
        if not (np.isnan(d_subj_best) or np.isnan(p_subj_best)):
            gap_subj = d_subj_best - p_subj_best
            if gap_subj < 0.05:
                lines.append(f"- subject F1 gap = {gap_subj:+.3f}: ΔA's subject info "
                             f"is *essentially the prompt's keyword footprint*. "
                             f"The Phase-A claim of \"subject signal in ΔA\" should be "
                             f"reframed as \"subject is already linearly readable from the "
                             f"prompt activation at t_1\".\n")
            elif gap_subj < 0.15:
                lines.append(f"- subject F1 gap = {gap_subj:+.3f}: ΔA carries a modest "
                             f"amount of additional subject information beyond the prompt.\n")
            else:
                lines.append(f"- subject F1 gap = {gap_subj:+.3f}: ΔA carries substantially "
                             f"more subject information than the prompt alone — model-internal "
                             f"reasoning representation is contributing.\n")
        lines.append("- For level/pass: A_prompt cannot see generation outcome, so a high "
                     "A_prompt R²_pass would imply pass-rate is predictable from prompt "
                     "features alone (length, vocabulary), not model reasoning.\n")

    out_path = OUT_DIR / "REPORT_critical.md"
    with open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"[report] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Critical Analysis: prompt-only activation vs ΔA")
    print("=" * 70)
    arr, meta = build_cache()
    df = per_layer_probe(arr, meta)
    df.to_csv(OUT_DIR / "prompt_per_layer.csv", index=False)
    print(f"[save] {OUT_DIR / 'prompt_per_layer.csv'}")
    cmp = compare_and_plot(df)
    write_report(df, cmp if cmp is not None else df)
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
