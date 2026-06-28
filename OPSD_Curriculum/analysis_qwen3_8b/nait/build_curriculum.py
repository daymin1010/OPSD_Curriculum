"""
build_curriculum.py
====================
Phase B — Track C: NAIT-style cluster discovery + curriculum stage definition
on the Qwen3-8B pilot 2,666 samples.

What this does
--------------
0. Load ΔA cache (N, L, D), meta (id/subject/level/pass_rate), Phase-0 calibrated
   PC1 directions, and Phase-A supervised directions (W_subj, v_lvl_lda, v_lvl_ridge).

1. Choose two layer windows from Phase-A REPORT_supervised.md:
     - LEVEL_WINDOW = layers maximising ridge R² (difficulty axis)
     - SUBJ_WINDOW  = layers maximising LDA macro-F1 (subject axis)

2. Build a low-dim feature per sample by concatenating
     - level    : mean over LEVEL_WINDOW of  ΔA · v_lvl_ridge           (1 dim)
     - subject  : mean over SUBJ_WINDOW  of  ΔA · W_subj (7 axes)       (7 dims)
     - pass_rate (z-scored)                                              (1 dim)
                                                                ⇒ 9-D feature.

3. Cluster discovery on the 9-D feature:
     - KMeans for K = 3..7  +  silhouette  +  BIC (via GaussianMixture)
     - Pick K* by argmax silhouette (tie-break by BIC).
     - Final clustering = KMeans(K*).

4. **Stage definition (5 stages)** combining (a) pass_rate quartiles
   and (b) cluster identity into the same 5-stage layout used in the
   1.5B curriculum (`trivial / easy / mixed / challenging / unreachable`):
     - Stage 1 (trivial)        : pass_rate ≥ 0.875
     - Stage 5 (unreachable)    : pass_rate == 0       (cluster majority subject = informative)
     - Stage 2..4               : 3 equal-population bins of remaining samples,
                                  ordered by mean (level − pass_rate*scale).
   Each stage gets a manifest with id / subject / level / pass_rate / cluster.

5. **Sanity vs old 1.5B**
     - subject distribution per stage (chi² test of independence)
     - if the old C5 outliers JSON exists: overlap with our Stage-5 (unreachable)

Outputs
-------
  outputs/curriculum/features.parquet                 (id, 9-D feature, cluster, stage)
  outputs/curriculum/stage_{1..5}_manifest.parquet    one per stage
  outputs/curriculum/clustering_diagnostics.csv       K, silhouette, BIC
  outputs/curriculum/REPORT_curriculum.md             human summary
  outputs/curriculum/plots/{cluster_scatter,stage_subject_heatmap}.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

BASE = Path("/scratch/lami2026/personal/jimin_2782")
NAIT_DIR = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/nait"
OUT_DIR  = NAIT_DIR / "outputs"
CUR_DIR  = OUT_DIR / "curriculum"
PLOT_DIR = CUR_DIR / "plots"
CUR_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CACHE = OUT_DIR / "delta_cache.npy"
META  = OUT_DIR / "delta_cache_meta.parquet"
DIRS_LDA = OUT_DIR / "lda_directions.npz"
SUPER_CSV = OUT_DIR / "supervised_per_layer.csv"

OLD_C5 = BASE / "src/4.6_Task2/activation/analysis/full_final/C5_outlier_samples.json"


def load_data():
    delta = np.load(CACHE, mmap_mode="r")             # (N, L, D)

    # Re-join meta the same way as Phase A
    meta = pd.read_parquet(META).reset_index(drop=True)
    meta["id"] = meta["id"].astype(str)

    SHIFT_DIR = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts"
    PASS_PARQ = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet"
    rows = []
    with open(SHIFT_DIR / "shifts_metadata.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("status") not in {"ok", "ok (skipped)", "completed"}: continue
            rows.append({"id": str(r["id"]), "subject": r.get("subject"),
                         "level": r.get("level")})
    sm = pd.DataFrame(rows).drop_duplicates(subset="id")
    pdf = pd.read_parquet(PASS_PARQ)
    pdf["id"] = pdf["sample_id"].astype(str)
    sm = sm.merge(pdf[["id", "pass_rate", "truncation_count"]], on="id", how="left")
    meta = meta.merge(sm, on="id", how="left")

    lda = np.load(DIRS_LDA, allow_pickle=True)
    sup = pd.read_csv(SUPER_CSV)
    return delta, meta, lda, sup


def build_feature(delta, meta, lda, sup, top_k_layers=5):
    """Returns (feat, lvl_win, subj_win)."""
    lvl_win  = sorted(sup.nlargest(top_k_layers, "lvl_ridge_R2")["layer"].astype(int).tolist())
    subj_win = sorted(sup.nlargest(top_k_layers, "subj_F1")["layer"].astype(int).tolist())
    print(f"[window] level layers  = {lvl_win}")
    print(f"[window] subject layers= {subj_win}")

    v_lvl  = lda["v_lvl_ridge"]                   # (L, D)
    W_subj = lda["W_subj"]                        # (L, D, 7)

    # mean over windows of dot product
    N, L, D = delta.shape

    # level scalar (1 dim)
    lvl_scores = np.zeros(N, dtype=np.float32)
    for l in lvl_win:
        x = np.asarray(delta[:, l, :], dtype=np.float32)
        lvl_scores += x @ v_lvl[l]
    lvl_scores /= len(lvl_win)

    # subject vector (7 dim)
    K_subj = W_subj.shape[2]
    subj_scores = np.zeros((N, K_subj), dtype=np.float32)
    for l in subj_win:
        x = np.asarray(delta[:, l, :], dtype=np.float32)
        subj_scores += x @ W_subj[l]
    subj_scores /= len(subj_win)

    # pass rate (z-scored)
    pr = meta["pass_rate"].astype(np.float32).values
    pr_z = (pr - pr.mean()) / (pr.std() + 1e-8)

    feat = np.concatenate([
        lvl_scores[:, None],          # 1
        subj_scores,                  # 7
        pr_z[:, None],                # 1
    ], axis=1)                        # (N, 9)

    # standardise each column
    feat = (feat - feat.mean(0, keepdims=True)) / (feat.std(0, keepdims=True) + 1e-8)
    return feat, lvl_win, subj_win, lvl_scores, subj_scores


def choose_K(feat, K_range=range(3, 8), seed=0):
    rows = []
    best = None
    for K in K_range:
        km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(feat)
        sil = silhouette_score(feat, km.labels_)
        # BIC via GMM with same K
        gm = GaussianMixture(n_components=K, covariance_type="full",
                             random_state=seed, n_init=3).fit(feat)
        bic = gm.bic(feat)
        rows.append({"K": K, "silhouette": float(sil), "bic": float(bic)})
        if (best is None) or (sil > best["sil"]):
            best = {"K": K, "sil": sil, "bic": bic, "labels": km.labels_, "km": km}
        print(f"  K={K}  sil={sil:+.4f}  BIC={bic:.0f}")
    return best, pd.DataFrame(rows)


def assign_stages(meta, lvl_scores, cluster_labels):
    """
    5 stages, paper-style:
      Stage 1 (trivial)     : pass_rate ≥ 0.875
      Stage 5 (unreachable) : pass_rate == 0
      Stages 2/3/4          : remaining samples, equal-tertiles by difficulty score
                              = level_score - pass_rate.
    """
    pr = meta["pass_rate"].values
    stage = np.full(len(meta), -1, dtype=int)

    stage[pr >= 0.875] = 1
    stage[pr <= 0.0]   = 5

    mid_mask = (stage == -1)
    diff_score = lvl_scores - pr * float(np.std(lvl_scores))   # rough scale match
    mid_idx = np.where(mid_mask)[0]
    order = mid_idx[np.argsort(diff_score[mid_mask])]
    n = len(order)
    s2 = order[: n // 3]
    s3 = order[n // 3 : 2 * n // 3]
    s4 = order[2 * n // 3 :]
    stage[s2] = 2
    stage[s3] = 3
    stage[s4] = 4

    assert (stage != -1).all()
    return stage


def main() -> None:
    t0 = time.time()
    delta, meta, lda, sup = load_data()
    N = len(meta)
    print(f"[load] N={N}  delta={delta.shape}")

    feat, lvl_win, subj_win, lvl_scores, subj_scores = build_feature(
        delta, meta, lda, sup, top_k_layers=5)
    print(f"[feat] shape = {feat.shape}")

    print("[cluster] scanning K")
    best, diag = choose_K(feat)
    diag.to_csv(CUR_DIR / "clustering_diagnostics.csv", index=False)
    K = best["K"]
    labels = best["labels"]
    print(f"[cluster] picked K={K}  silhouette={best['sil']:.4f}")

    stages = assign_stages(meta, lvl_scores, labels)

    # ── assemble full feature dataframe ───────────────────────────────────
    cols = (["lvl_proj"]
            + [f"subj_{i}" for i in range(subj_scores.shape[1])]
            + ["pass_z"])
    fdf = pd.DataFrame(feat, columns=cols)
    fdf["id"] = meta["id"].values
    fdf["subject"] = meta["subject"].values
    fdf["level"]   = meta["level"].values
    fdf["pass_rate"] = meta["pass_rate"].values
    fdf["cluster"] = labels
    fdf["stage"]   = stages
    fdf.to_parquet(CUR_DIR / "features.parquet", index=False)
    print(f"[save] features.parquet")

    # ── stage manifests ──────────────────────────────────────────────────
    for s in range(1, 6):
        sub = fdf[fdf["stage"] == s][
            ["id", "subject", "level", "pass_rate", "cluster"]].reset_index(drop=True)
        sub.to_parquet(CUR_DIR / f"stage_{s}_manifest.parquet", index=False)
        print(f"  stage {s}: N={len(sub)}  mean pass={sub['pass_rate'].mean():.3f} "
              f"mean level={sub['level'].mean():.2f}")

    # ── sanity vs old 1.5B C5 outliers (Stage-5 overlap) ─────────────────
    overlap_line = "(old C5 outliers JSON not found — skipped)"
    if OLD_C5.exists():
        try:
            with open(OLD_C5) as f:
                old = json.load(f)
            old_ids = {str(x) if not isinstance(x, dict) else str(x.get("id", x))
                       for x in old}
            s5_ids = set(fdf.loc[fdf["stage"] == 5, "id"])
            inter = old_ids & s5_ids
            overlap_line = (f"old C5 outliers = {len(old_ids)}, "
                            f"Stage-5 = {len(s5_ids)}, overlap = {len(inter)}")
        except Exception as e:
            overlap_line = f"(failed to load C5 outliers: {e})"

    # ── subject × stage chi² ─────────────────────────────────────────────
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(fdf["subject"], fdf["stage"])
    chi2, p, dof, _ = chi2_contingency(ct.values)
    print(f"[chi²] subj×stage  χ²={chi2:.1f}  dof={dof}  p={p:.3e}")

    # ── plots ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    p2 = PCA(n_components=2, random_state=0).fit_transform(feat)
    scat = ax.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    ax.set_title(f"KMeans clusters (K={K}, silhouette={best['sil']:.3f})")
    ax.set_xlabel("PCA-1 of feature")
    ax.set_ylabel("PCA-2 of feature")
    fig.colorbar(scat, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cluster_scatter.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ct.values / ct.values.sum(axis=0, keepdims=True),
                   cmap="viridis", aspect="auto")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns.tolist())
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index.tolist())
    ax.set_xlabel("stage"); ax.set_ylabel("subject")
    ax.set_title("Subject composition per stage (column-normalised)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stage_subject_heatmap.png", dpi=140)
    plt.close(fig)

    # ── report ────────────────────────────────────────────────────────────
    lines = ["# Phase B — Curriculum Construction Report\n",
             f"- N samples              : {N}",
             f"- Feature dim            : {feat.shape[1]}  (1 lvl_proj + 7 subj + 1 pass_z)",
             f"- LEVEL window layers    : `{lvl_win}`",
             f"- SUBJECT window layers  : `{subj_win}`",
             f"- Chosen K (silhouette)  : **{K}**  (sil={best['sil']:.4f}, "
             f"BIC={best['bic']:.0f})",
             f"- Wall time              : {(time.time()-t0)/60:.1f} min\n"]
    lines += ["## K selection\n",
              diag.to_markdown(index=False, floatfmt=".4f"), ""]

    sizes = fdf.groupby("stage").agg(
        n=("id", "size"),
        mean_pass=("pass_rate", "mean"),
        mean_level=("level", "mean"))
    lines += ["## Stage sizes & means\n",
              sizes.reset_index().to_markdown(index=False, floatfmt=".3f"), ""]

    lines += ["## Subject × stage contingency (column %)\n",
              (ct / ct.sum(axis=0)).round(3).to_markdown(), "",
              f"- χ² test:  χ²={chi2:.2f}  dof={dof}  p={p:.3e}", ""]

    lines += ["## Sanity vs old 1.5B C5-outliers\n",
              "- " + overlap_line, ""]

    lines += ["## Files\n",
              "- `features.parquet`              : per-sample feature + cluster + stage",
              "- `stage_{1..5}_manifest.parquet` : per-stage manifests",
              "- `clustering_diagnostics.csv`    : K-scan silhouette + BIC",
              "- `plots/cluster_scatter.png`     : PCA-2D of feature, coloured by cluster",
              "- `plots/stage_subject_heatmap.png`: subject composition per stage", ""]

    lines += ["## Notes\n",
              "- Stage layout is fixed (1=trivial, 5=unreachable, 2/3/4=tertile of `level - pass`).",
              "- Cluster identity is *not* used directly to define stages, but is stored alongside",
              "  to allow stratified sampling within a stage in later experiments.",
              "- Subject window vs level window are deliberately taken from *different* layers",
              "  (Phase-A finding: subject info concentrates lower, level info higher)."]

    (CUR_DIR / "REPORT_curriculum.md").write_text("\n".join(lines))
    print(f"[done] wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
