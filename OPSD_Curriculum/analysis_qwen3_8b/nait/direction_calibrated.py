"""
direction_calibrated.py
=======================
Per-layer PC1 + sign-calibrated direction analysis on the Qwen3-8B pilot 2,666
activation shifts (ΔA = h_tK − h_t1, 36 layers, dim=12288).

Pipeline
--------
0. Load metadata (Track B) + pass_rate (Track A); align by id.
1. Stream-load all 2666 .pt files into a (N=2666, L=36, D=12288) float32 array.
2. For EACH layer l:
   (Step 1) v_l = first principal component of ΔA^(l)  (torch.pca_lowrank, q=8)
   (Step 2) μ_diff_l = mean over samples of ΔA^(l)   (= mean activation shift)
   (Step 3) sign calibration: if μ_diff_l · v_l < 0  →  v_l ← -v_l
3. Projection scores S[n, l] = ΔA^(l)[n] · v_l    (calibrated, signed)
4. Subject / level separation diagnostics:
     - Spearman ρ(score_l, level)
     - one-way ANOVA F(score_l ~ subject)
     - silhouette(subject)  on calibrated score (1D per layer) + top-K PCs (8D)
5. Linear probes (cheap baseline) on PCA-256 reduced features:
     - subject:  multinomial logistic regression (5-fold CV macro-F1)
     - level:    ridge regression       (5-fold CV R²)
     - pass:     ridge regression       (5-fold CV R²)
6. Plots + markdown report.

Runs on CPU.  ~15-25 min wall-time.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

BASE = Path("/scratch/lami2026/personal/jimin_2782")
SHIFT_DIR = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts"
PASS_PARQ = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet"
OUT_DIR   = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs"
PLOT_DIR  = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = OUT_DIR / "delta_cache.npy"     # shape (N, L, D) float32 ≈ 4.7 GB
META_PATH  = OUT_DIR / "delta_cache_meta.parquet"

N_LAYERS = 36
D_HID    = 12288
PCA_Q    = 8
PROBE_K  = 256


# ─────────────────────────────────────────────────────────────────────────────
# 0/1. Load metadata + cache ΔA tensor (one-time)
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata_df() -> pd.DataFrame:
    rows = []
    with open(SHIFT_DIR / "shifts_metadata.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") not in {"ok", "ok (skipped)", "completed"}:
                continue
            rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
    df["id"] = df["id"].astype(str)

    # join pass rate
    pdf = pd.read_parquet(PASS_PARQ)
    pdf["id"] = pdf["sample_id"].astype(str)
    df = df.merge(
        pdf[["id", "pass_rate", "pass_count", "truncation_count", "mean_response_length"]],
        on="id", how="left",
    )
    return df


def build_cache(df: pd.DataFrame) -> np.ndarray:
    if CACHE_PATH.exists() and META_PATH.exists():
        meta = pd.read_parquet(META_PATH)
        if len(meta) == len(df) and (meta["id"].values == df["id"].values).all():
            print(f"[cache] hit → {CACHE_PATH} ({CACHE_PATH.stat().st_size/1e9:.2f} GB)")
            return np.load(CACHE_PATH, mmap_mode="r")
        print("[cache] meta mismatch → rebuilding")

    N = len(df)
    print(f"[cache] building (N={N}, L={N_LAYERS}, D={D_HID}) ≈ {N*N_LAYERS*D_HID*4/1e9:.2f} GB")
    arr = np.empty((N, N_LAYERS, D_HID), dtype=np.float32)
    t0 = time.time()
    for i, sid in enumerate(df["id"].tolist()):
        path = SHIFT_DIR / f"{sid}.pt"
        s = torch.load(path, map_location="cpu", weights_only=False)
        sh = s["shifts"]
        for l in range(N_LAYERS):
            v = sh[l]
            arr[i, l] = v.float().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{N}  ({time.time()-t0:.1f}s)")
    np.save(CACHE_PATH, arr)
    df[["id"]].to_parquet(META_PATH)
    print(f"[cache] saved → {CACHE_PATH} ({time.time()-t0:.1f}s total)")
    return np.load(CACHE_PATH, mmap_mode="r")


# ─────────────────────────────────────────────────────────────────────────────
# 2/3. Per-layer PC1 + sign calibration + projection scores
# ─────────────────────────────────────────────────────────────────────────────

def per_layer_pca_and_calibrate(delta: np.ndarray):
    """
    delta : (N, L, D) float32 (mmap ok)
    returns
        V_top : (L, PCA_Q, D) float32      — top PCs (sign-uncalibrated for k>0)
        v_cal : (L, D)        float32      — calibrated PC1 (Step 3 applied)
        mu    : (L, D)        float32      — μ_diff_l = mean of ΔA^(l)
        evr   : (L, PCA_Q)    float32      — explained variance ratio
        S     : (N, L)        float32      — projection scores onto v_cal
        S_topk: (N, L, PCA_Q) float32      — projection scores onto V_top (k≥0)
    """
    N, L, D = delta.shape
    V_top  = np.zeros((L, PCA_Q, D),  dtype=np.float32)
    v_cal  = np.zeros((L, D),         dtype=np.float32)
    mu_all = np.zeros((L, D),         dtype=np.float32)
    evr    = np.zeros((L, PCA_Q),     dtype=np.float32)
    S      = np.zeros((N, L),         dtype=np.float32)
    S_topk = np.zeros((N, L, PCA_Q),  dtype=np.float32)

    for l in range(L):
        t0 = time.time()
        # Load this layer fully into RAM (N×D fp32 ≈ 130 MB)
        X = np.ascontiguousarray(delta[:, l, :]).astype(np.float32)   # (N, D)

        # ──── Step 1: PC1 (and a few more for diagnostics) ─────────────────
        # Center for PCA
        mean_X = X.mean(axis=0)                          # (D,)
        Xc = X - mean_X
        # torch.pca_lowrank works on tensors; we'll use it for speed
        Xt = torch.from_numpy(Xc)
        U, Sval, Vt = torch.pca_lowrank(Xt, q=PCA_Q, center=False, niter=4)
        # Vt : (D, q). Components are columns of Vt.
        comps = Vt.numpy().T.astype(np.float32)          # (q, D)
        # Normalize (should already be unit-norm but ensure)
        comps /= (np.linalg.norm(comps, axis=1, keepdims=True) + 1e-12)
        v1 = comps[0]                                    # (D,)  ← PC1

        # explained variance ratio
        total_var = float((Xc * Xc).sum() / max(N - 1, 1))
        sv = Sval.numpy().astype(np.float64)
        var_per_pc = (sv ** 2) / max(N - 1, 1)
        evr_l = (var_per_pc / max(total_var, 1e-12)).astype(np.float32)

        # ──── Step 2: μ_diff_l (mean activation shift) ─────────────────────
        # NB: ΔA already encodes (h_tK − h_t1) per sample; their mean is the
        # population mean shift; equivalently mean_X here.
        mu_l = mean_X.copy()

        # ──── Step 3: sign calibration ─────────────────────────────────────
        if float(mu_l @ v1) < 0:
            v1 = -v1

        # Projection scores
        # Score onto calibrated v1: dot with the raw (uncentered) ΔA per sample.
        # (Using uncentered X so that score absorbs the bulk shift along v1.)
        S[:, l]      = X @ v1
        S_topk[:, l] = X @ comps.T          # (N, q)

        V_top[l]  = comps
        v_cal[l]  = v1
        mu_all[l] = mu_l
        evr[l]    = evr_l
        print(f"  layer {l:02d}  EVR1={evr_l[0]*100:5.2f}%  "
              f"|mu|={np.linalg.norm(mu_l):.2f}  cos(mu,v1)={float(mu_l@v1)/(np.linalg.norm(mu_l)*np.linalg.norm(v1)+1e-12):+.3f}  "
              f"({time.time()-t0:.1f}s)")
    return V_top, v_cal, mu_all, evr, S, S_topk


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diagnostics: subject/level separation per layer
# ─────────────────────────────────────────────────────────────────────────────

def diagnostics(S: np.ndarray, S_topk: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr, f_oneway
    from sklearn.metrics import silhouette_score

    subjects = df["subject"].astype(str).values
    levels   = df["level"].astype(float).values
    passes   = df["pass_rate"].astype(float).values

    N, L = S.shape
    rows = []
    # Pre-encode subject ids for silhouette
    subj_uniq = sorted(set(subjects))
    subj_id = np.array([subj_uniq.index(s) for s in subjects])

    # Sample if very large (silhouette is O(n²)); 2666 is fine.
    for l in range(L):
        s = S[:, l]
        rho_lv, p_lv = spearmanr(s, levels, nan_policy="omit")
        rho_pa, p_pa = spearmanr(s, passes, nan_policy="omit")
        # ANOVA F across subjects on the 1D score
        groups = [s[subjects == sb] for sb in subj_uniq if (subjects == sb).sum() > 1]
        F, p_F = f_oneway(*groups)
        # silhouette: subject on calibrated 1D score
        try:
            sil_1d = silhouette_score(s.reshape(-1, 1), subj_id, metric="euclidean")
        except Exception:
            sil_1d = float("nan")
        # silhouette: subject on top-K PC subspace
        try:
            sil_kd = silhouette_score(S_topk[:, l, :], subj_id, metric="euclidean")
        except Exception:
            sil_kd = float("nan")
        rows.append(dict(
            layer=l,
            rho_score_level=rho_lv, p_level=p_lv,
            rho_score_pass=rho_pa,  p_pass=p_pa,
            anova_subject_F=F, anova_subject_p=p_F,
            silhouette_subject_1d=sil_1d,
            silhouette_subject_topK=sil_kd,
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Linear probes on PCA-256 features (per layer)
# ─────────────────────────────────────────────────────────────────────────────

def linear_probes(delta: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    subjects = df["subject"].astype(str).values
    levels   = df["level"].astype(float).values
    passes   = df["pass_rate"].astype(float).values

    N, L, D = delta.shape
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    kf  = KFold(n_splits=5, shuffle=True, random_state=0)
    for l in range(L):
        t0 = time.time()
        X = np.ascontiguousarray(delta[:, l, :]).astype(np.float32)
        # PCA-256 (no scaling; ΔA already comparable across dims)
        pca = PCA(n_components=PROBE_K, random_state=0, svd_solver="randomized")
        Z = pca.fit_transform(X)        # (N, 256)
        # subject classifier
        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=2000, multi_class="multinomial",
                                                  C=1.0, n_jobs=1))])
        f1 = cross_val_score(clf, Z, subjects, cv=skf, scoring="f1_macro", n_jobs=1).mean()
        # level regression
        reg_l = Pipeline([("sc", StandardScaler()), ("rd", Ridge(alpha=10.0))])
        r2_l  = cross_val_score(reg_l, Z, levels, cv=kf, scoring="r2", n_jobs=1).mean()
        # pass regression
        reg_p = Pipeline([("sc", StandardScaler()), ("rd", Ridge(alpha=10.0))])
        r2_p  = cross_val_score(reg_p, Z, passes, cv=kf, scoring="r2", n_jobs=1).mean()
        rows.append(dict(layer=l, subject_f1_macro=f1, level_r2=r2_l, pass_r2=r2_p,
                         pca256_evr_sum=float(pca.explained_variance_ratio_.sum())))
        print(f"  probe layer {l:02d}  subj_F1={f1:.3f}  level_R²={r2_l:+.3f}  pass_R²={r2_p:+.3f}  "
              f"PCA256 EVR={pca.explained_variance_ratio_.sum()*100:.1f}%  ({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plots + report
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagnostics(diag: pd.DataFrame, probe: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    L = diag["layer"].values
    axes[0,0].plot(L, diag["rho_score_level"].abs(), label="|ρ(score, level)|", marker="o")
    axes[0,0].plot(L, diag["rho_score_pass"].abs(),  label="|ρ(score, pass)|",  marker="x")
    axes[0,0].set_xlabel("layer"); axes[0,0].set_ylabel("|Spearman ρ|"); axes[0,0].legend(); axes[0,0].grid(alpha=.3)
    axes[0,0].set_title("Calibrated-direction score vs. level / pass_rate")

    axes[0,1].plot(L, diag["silhouette_subject_1d"], label="silhouette(subject, 1D)", marker="o")
    axes[0,1].plot(L, diag["silhouette_subject_topK"], label=f"silhouette(subject, top-{PCA_Q}D)", marker="x")
    axes[0,1].axhline(0, color="k", lw=.5)
    axes[0,1].set_xlabel("layer"); axes[0,1].set_ylabel("silhouette"); axes[0,1].legend(); axes[0,1].grid(alpha=.3)
    axes[0,1].set_title("Subject separation by layer")

    axes[1,0].plot(L, probe["subject_f1_macro"], label="subject F1 (logistic, PCA-256)", marker="o", color="C2")
    axes[1,0].axhline(1/7, color="k", lw=.5, ls="--", label="random (1/7)")
    axes[1,0].set_xlabel("layer"); axes[1,0].set_ylabel("macro-F1"); axes[1,0].legend(); axes[1,0].grid(alpha=.3)
    axes[1,0].set_title("Subject linear-probe accuracy")

    axes[1,1].plot(L, probe["level_r2"], label="level R² (ridge, PCA-256)", marker="o", color="C3")
    axes[1,1].plot(L, probe["pass_r2"],  label="pass R² (ridge, PCA-256)",  marker="x", color="C4")
    axes[1,1].axhline(0, color="k", lw=.5)
    axes[1,1].set_xlabel("layer"); axes[1,1].set_ylabel("R² (CV)"); axes[1,1].legend(); axes[1,1].grid(alpha=.3)
    axes[1,1].set_title("Level / pass linear-probe R²")

    plt.tight_layout()
    out = PLOT_DIR / "summary_per_layer.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"[plot] {out}")


def write_report(diag: pd.DataFrame, probe: pd.DataFrame, evr: np.ndarray):
    md = ["# Calibrated Direction & Linear Probe Report",
          "",
          f"- N samples: 2666",
          f"- N layers : {N_LAYERS}",
          f"- D hidden : {D_HID}",
          f"- PCA top-K: {PCA_Q}",
          f"- Probe PCA dim: {PROBE_K}",
          "",
          "## Per-layer summary (calibrated v_l = PC1 with sign calibration)",
          "",
          "| L | EVR(PC1)% | |ρ(score, level)| | |ρ(score, pass)| | sil(subj, 1D) | sil(subj, topK) | subj F1 | level R² | pass R² |",
          "|---|----------:|----------------:|---------------:|--------------:|---------------:|--------:|---------:|--------:|"]
    for i, r in diag.iterrows():
        l = int(r["layer"])
        pr = probe.iloc[l]
        md.append(f"| {l} | {evr[l,0]*100:5.2f} | {abs(r['rho_score_level']):.3f} | {abs(r['rho_score_pass']):.3f} "
                  f"| {r['silhouette_subject_1d']:+.3f} | {r['silhouette_subject_topK']:+.3f} "
                  f"| {pr['subject_f1_macro']:.3f} | {pr['level_r2']:+.3f} | {pr['pass_r2']:+.3f} |")
    md.append("")
    # Best layers
    bl_lv = int(diag["rho_score_level"].abs().idxmax())
    bl_pa = int(diag["rho_score_pass"].abs().idxmax())
    bl_sj = int(probe["subject_f1_macro"].idxmax())
    md += [
        "## Best layers",
        f"- |ρ(score, level)| max: layer **{bl_lv}**  ρ={diag.iloc[bl_lv]['rho_score_level']:+.3f}",
        f"- |ρ(score, pass)|  max: layer **{bl_pa}**  ρ={diag.iloc[bl_pa]['rho_score_pass']:+.3f}",
        f"- subject F1 max     : layer **{bl_sj}**  F1={probe.iloc[bl_sj]['subject_f1_macro']:.3f}",
        f"- level R² max       : layer **{int(probe['level_r2'].idxmax())}** R²={probe['level_r2'].max():+.3f}",
        f"- pass  R² max       : layer **{int(probe['pass_r2'].idxmax())}** R²={probe['pass_r2'].max():+.3f}",
        "",
        "## Notes",
        "- Calibrated direction is unsupervised PC1 with sign aligned to μ_diff (paper Eq. 4).",
        "- 1D silhouette ≈ 0 with weak |ρ(level)| would mean *unsupervised PC1 ≠ level/subject axis*.",
        "- Compare with the linear probe (PCA-256 + ridge/logreg): if probe is much stronger, the",
        "  signal exists but in directions other than PC1 — motivates LDA / supervised direction next.",
    ]
    out = OUT_DIR / "REPORT.md"
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"[report] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    df = load_metadata_df()
    df = df.sort_values("id").reset_index(drop=True)
    print(f"[meta] N={len(df)}  subjects={sorted(df['subject'].unique())}  levels={sorted(df['level'].unique())}")

    delta = build_cache(df)                         # mmap (N, L, D)

    print("\n[step] per-layer PC1 + sign calibration + projection scores")
    V_top, v_cal, mu_all, evr, S, S_topk = per_layer_pca_and_calibrate(delta)
    np.savez(OUT_DIR / "directions.npz",
             v_cal=v_cal, V_top=V_top, mu=mu_all, evr=evr)
    np.save(OUT_DIR / "scores_calibrated.npy", S)         # (N, L)
    np.save(OUT_DIR / "scores_topK.npy",       S_topk)    # (N, L, K)

    print("\n[step] diagnostics (silhouette / spearman / ANOVA)")
    diag = diagnostics(S, S_topk, df)
    diag.to_csv(OUT_DIR / "diagnostics_per_layer.csv", index=False)

    print("\n[step] linear probes (PCA-256 → logistic/ridge, 5-fold CV)")
    probe = linear_probes(delta, df)
    probe.to_csv(OUT_DIR / "linear_probe_per_layer.csv", index=False)

    plot_diagnostics(diag, probe)
    write_report(diag, probe, evr)

    print(f"\n[done] total wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
