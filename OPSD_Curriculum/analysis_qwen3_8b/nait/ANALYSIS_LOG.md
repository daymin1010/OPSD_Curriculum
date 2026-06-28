# NAIT Analysis Log — Qwen3-8B pilot (N=2666)

This document records every analysis we ran on the Qwen3-8B activation-shift
pilot (2,666 samples × 36 layers × 12288 dims), what method we used, **why**,
the inputs/outputs of each script, and the headline numbers.

> **Scope.** Activation analysis on top of Track-B activation shifts
> (ΔA = h_{tK} − h_{t1}, per layer, residual stream) for the Qwen3-8B base
> (non-reasoning) model, joined with Track-A pass-rate measurements.

---

## Directory layout

```
src/OPSD_Curriculum/analysis_qwen3_8b/nait/
├── ANALYSIS_LOG.md                   ← this file
├── direction_calibrated.py           ← Phase 0
├── supervised_direction.py           ← Phase A
├── build_curriculum.py               ← Phase B
└── outputs/
    ├── delta_cache.npy               (N=2666, L=36, D=12288)  float32  ≈ 4.7 GB
    ├── delta_cache_meta.parquet      (id only — joined at run-time)
    │
    ├── directions.npz                Phase-0 results
    ├── diagnostics_per_layer.csv     Phase-0 unsupervised diagnostics
    ├── linear_probe_per_layer.csv    Phase-0 probe diagnostics
    ├── scores_calibrated.npy         (N, L)   PC1 calibrated projection
    ├── scores_topK.npy               (N, L, 8) top-8 PC projections
    ├── REPORT.md                     Phase-0 human report
    ├── plots/summary_per_layer.png
    │
    ├── lda_directions.npz            Phase-A results
    ├── supervised_per_layer.csv      Phase-A per-layer metrics
    ├── REPORT_supervised.md          Phase-A human report
    ├── plots/layer_window_compare.png
    │
    └── curriculum/                   Phase-B results
        ├── features.parquet
        ├── stage_{1..5}_manifest.parquet
        ├── clustering_diagnostics.csv
        ├── REPORT_curriculum.md
        └── plots/{cluster_scatter,stage_subject_heatmap}.png
```

---

## Phase 0 — Unsupervised calibrated PC1 (`direction_calibrated.py`)

**Goal.** Implement the NAIT paper's "calibrated direction" recipe (Eq. 1–3) on
Qwen3-8B activation shifts and quantify per-layer signal for subject / level /
pass_rate.

**Method.**
For each layer $l$:

1. **PC1**: $v_l = \text{PCA}_1(\Delta\mathcal{A}^{(l)})$ via `torch.pca_lowrank(q=8)`.
2. **Mean shift**: $\mu_l = \frac{1}{N}\sum_n \Delta\mathcal{A}^{(l)}_n$.
3. **Sign calibration**: if $\mu_l \cdot v_l < 0$, flip $v_l \leftarrow -v_l$.
4. Project: $S[n,l] = \Delta\mathcal{A}^{(l)}_n \cdot v_l$.
5. **Diagnostics** on $S[:,l]$:
   - Spearman ρ vs `level` and vs `pass_rate`
   - one-way ANOVA F (`subject` factor)
   - silhouette of `subject` clusters in 1-D (PC1) and 8-D (top-K)
6. **Linear probe baseline** (PCA-256 reduced features, 5-fold CV):
   - `subject` → multinomial logistic regression, macro-F1
   - `level`  → ridge regression, R²
   - `pass`   → ridge regression, R²

**Why this design.** PC1 + sign calibration is the unsupervised baseline the
NAIT paper proposes. We compare against a *supervised* probe (PCA-256 + LDA
/ ridge) so we can tell whether "subject info is missing" or "subject info
exists but is not on PC1".

**Headline results** (from `outputs/REPORT.md`):

| Signal | Best layer | Value |
|---|---|---|
| \|ρ(PC1-score, level)\| | **L21** | **0.808** |
| \|ρ(PC1-score, pass_rate)\| | L11 | 0.561 |
| Subject silhouette on PC1 (1-D) | – | **−0.13 .. −0.30** (all layers) |
| Subject macro-F1 (PCA-256 + logistic) | **L14** | **0.737** |
| Level R² (PCA-256 + ridge) | **L18** | **+0.878** |
| Pass R²  (PCA-256 + ridge) | L16 | +0.293 |

**Conclusion.** Calibrated PC1 ≈ unsupervised *difficulty axis* (very strong
ρ with `level`). Subject information *exists* in ΔA (probe F1=0.74 ≫ random
0.14) but is **not** on PC1 (negative silhouette) → motivates Phase A.

---

## Phase A — Supervised directions (LDA + ridge) (`supervised_direction.py`)

**Goal.** Find the *actual* subject and level directions in ΔA, evaluate
their layer dependence, and decide layer windows for downstream curriculum
construction.

**Method.** For each layer $l$, on ΔA reduced to 256 PCs:

1. **Subject LDA** — 8 classes → up to 7 LDA dimensions
   ($W_{\text{subj}} \in \mathbb{R}^{D \times 7}$). Metric: 5-fold CV macro-F1.
2. **Level-binary LDA-1** — class = level above/below median (binary, threshold = median=4)
   → 1-D direction $v_{\text{lvl,lda}}$. Metric: F1 + Spearman ρ vs continuous level.
3. **Level ridge** — fit Ridge(α=1) regressing `level` on PCA-256 features
   → direction $v_{\text{lvl,ridge}}$. Metric: 5-fold CV R².
4. **Direction back-projection**: $v_D = \Pi^\top \beta$ where $\Pi$ is the
   PCA-256 components matrix, then unit-normalised. This brings supervised
   directions back to the full 12288-D space.
5. **Cosine similarity** vs Phase-0 calibrated PC1 (per layer) to quantify
   how the supervised difficulty axis differs from the unsupervised PC1.
6. **Layer-window recommendation**: top-5 layers by `subj_F1` (subject window)
   and top-5 by `lvl_ridge_R2` (level window).

**Why.** Phase 0 told us subject info isn't on PC1 — we need a supervised
direction. LDA is the natural choice (Fisher-optimal axis for class
separation). Ridge gives a *continuous* difficulty axis less sensitive to the
median-split. Comparing both vs PC1 tells us how well the unsupervised proxy
captures the supervised signal.

**Outputs**:
- `supervised_per_layer.csv`: layer, subj_F1, lvl_lda_F1, lvl_lda_rho,
  lvl_ridge_R2, cos(PC1, v_lvl_lda), cos(PC1, v_lvl_ridge), cos(PC1, W_subj)
- `lda_directions.npz`: $W_{\text{subj}}[L,D,7]$, $v_{\text{lvl,lda}}[L,D]$,
  $v_{\text{lvl,ridge}}[L,D]$, list of subject classes, level median
- `REPORT_supervised.md` + `plots/layer_window_compare.png`

**Headline result (DONE — wall = 14.6 min).**

| Signal | Best layer(s) | Value |
|---|---|---|
| Subject macro-F1 (LDA on PCA-256, 5-fold CV) | **L14** (also L12, L13, L15, L11) | **0.779** |
| Level R² (Ridge on PCA-256, 5-fold CV) | **L18** ≈ L17 | **+0.879** |
| Level LDA-1 \|ρ\| vs continuous level | L23, L18, L20, L16, L19 | up to **0.892** |
| \|cos(PC1, v_lvl_ridge)\| @ best-level layer | L18 | **0.107** |
| \|cos(PC1, v_lvl_lda)\| @ L18 | L18 | 0.075 |
| mean \|cos(PC1, W_subject)\| (7 axes) @ L14 | L14 | **0.0043** ← essentially orthogonal |

**Layer windows chosen for Phase B**:
- Subject window: layers **[11, 12, 13, 14, 15]**  (mid-block)
- Level   window: layers **[15, 16, 17, 18, 19]**  (mid–late)

**Interpretation.**
- Both subject (F1=0.78, vs chance=0.125) and level (R²=0.88) signals are
  strongly present in ΔA. The earlier worry that "subject/level info isn't
  in PC1" was correct — but the *information* is in ΔA, just not on the
  highest-variance axis.
- The **subject window** (L11–15) and **level window** (L15–19) overlap but
  are not identical. Subject identity is most readable a few layers earlier
  than fine-grained level.
- `cos(PC1, v_lvl_ridge) ≈ 0.11` and `cos(PC1, W_subj) ≈ 0.004`: PC1 is
  nearly orthogonal to the supervised label directions, yet still has
  \|ρ\|=0.81 with level. This is the classic "variance vs. discriminability"
  decoupling — PC1 finds the largest-variance ΔA direction, which happens to
  rank samples roughly by difficulty (long correct vs. short wrong), while
  the supervised axes pick up label-specific signal in lower-variance
  components.

### Why the id-join in Phase A is provably safe

`delta_cache_meta.parquet` only stores `id` (no subject/level/pass_rate).
This was a Phase-0 design slip (`df[["id"]].to_parquet(META_PATH)`).
Phase A re-joins subject/level/pass_rate at run-time. This is correct
because:

1. **Cache row i ↔ id\[i\]** by construction. In
   `direction_calibrated.py` the cache is filled inside a loop
   `for i, sid in enumerate(df["id"].tolist())`, so cache row `i`
   *is* sample `df.iloc[i].id` — by definition, not by sort order.
2. **META.parquet preserves that order**. The next line is
   `df[["id"]].to_parquet(META_PATH)`, so the on-disk parquet row
   `i` carries exactly the id used to write cache row `i`.
3. **Re-use is gated on order equality**. On every subsequent run,
   `direction_calibrated.py` rebuilds `df` from `shifts_metadata.jsonl`
   and only re-uses the cache if
   `(meta["id"].values == df["id"].values).all()`.
4. **Phase A merges on the id key**, not the index:
   `meta.merge(sm, on="id", how="left")`. Pandas left-merge is
   order-preserving for the left frame, so `meta.iloc[i]` still
   corresponds to cache row `i`, with subject/level/pass_rate
   correctly attached by id.
5. **Sanity assertions** catch any failure:
   ```python
   assert meta["subject"].notna().all(), "subject missing for some ids"
   assert meta["level"].notna().all(),   "level missing for some ids"
   ```
   The Phase-A run logged `meta after join: (2666, 4)` with all 8
   subjects present → join is intact.

> **TODO (fragility note).** Next time the cache is rebuilt from scratch,
> update `direction_calibrated.py` to write
> `df[["id","subject","level","pass_rate"]].to_parquet(META_PATH)`.
> The current code is correct but relies on an extra join step downstream.

---

## Phase B — Curriculum stage construction (`build_curriculum.py`)

**Goal.** Produce a 5-stage NAIT-style curriculum manifest for the Qwen3-8B
pilot, combining model-internal difficulty (pass_rate) with activation-space
clusters.

**Method.**

1. Read Phase-A `supervised_per_layer.csv`, pick
   - **LEVEL_WINDOW** = top-5 layers by `lvl_ridge_R2`
   - **SUBJECT_WINDOW** = top-5 layers by `subj_F1`
2. **Per-sample 9-D feature** = $\big[ \mathrm{lvl\_proj},\ \mathrm{subj\_proj}_{1..7},\ z(\text{pass\_rate}) \big]$
   where
   - $\mathrm{lvl\_proj} = \overline{\Delta\mathcal{A}^{(l)} \cdot v_{\text{lvl,ridge}}}$ averaged over LEVEL_WINDOW
   - $\mathrm{subj\_proj} = \overline{\Delta\mathcal{A}^{(l)} W_{\text{subj}}}$ averaged over SUBJECT_WINDOW (7 channels)
   Each column is standardised (zero mean, unit variance).
3. **Cluster discovery**: KMeans for $K \in \{3,4,5,6,7\}$, pick $K^*$ by max
   silhouette (BIC via GaussianMixture as tie-breaker / sanity check).
4. **Stage assignment** (independent of the cluster index, but stored
   alongside for later stratification):
   - **Stage 1 (trivial)**: pass_rate ≥ 0.875 (≥ 7/8 correct)
   - **Stage 5 (unreachable)**: pass_rate == 0
   - **Stages 2 / 3 / 4**: remaining samples ranked by
     $\mathrm{lvl\_proj} - \mathrm{pass\_rate}\cdot\sigma(\mathrm{lvl\_proj})$,
     split into 3 equal-population bins.
5. **Sanity**:
   - χ² test of independence on the `subject × stage` table (do stages
     respect subject balance?).
   - If `src/4.6_Task2/.../full_final/C5_outlier_samples.json` exists,
     compute overlap with our Stage 5 — checks whether the 1.5B-derived
     "unreachable" set agrees with the Qwen3-8B-derived one.

**Why this design.**

- The 5-stage layout mirrors the 1.5B curriculum so downstream training
  pipelines are drop-in compatible.
- We mix `pass_rate` (model-internal) and `lvl_proj` (label-side difficulty
  in activation space) because the validation report showed Pass and Level
  carry *different* signal (R² 0.29 vs 0.88).
- KMeans cluster id is *not* used to define stages — it is only stored as
  metadata, so that within a stage we can later subsample to balance subjects
  or other auxiliary axes.

**Outputs**:
- `curriculum/features.parquet` (id, lvl_proj, subj_0..6, pass_z, subject, level, pass_rate, cluster, stage)
- `curriculum/stage_{1..5}_manifest.parquet`
- `curriculum/clustering_diagnostics.csv` (K, silhouette, BIC)
- `curriculum/REPORT_curriculum.md`
- `curriculum/plots/{cluster_scatter,stage_subject_heatmap}.png`

**Headline results (DONE — wall = 0.2 min).**

K-scan picked **K\*=7** (silhouette=0.265, BIC=50821). Silhouette monotonically
increased from K=3 (0.145) to K=7, suggesting the 9-D feature space has rich
sub-structure beyond pure difficulty.

| stage | n | mean pass | mean level | semantics |
|---:|---:|---:|---:|---|
| 1 | **1063** (39.9%) | 0.973 | 2.96 | trivial — model already solves them |
| 2 | 332 (12.5%) | 0.537 | 2.93 | easy level, partial success (warm-up) |
| 3 | 332 (12.5%) | 0.453 | 5.04 | mid level, mixed success |
| 4 | 333 (12.5%) | 0.316 | 6.71 | hard level, mostly failing |
| 5 | **606** (22.7%) | 0.000 | 5.28 | unreachable — 0/8 across all rollouts |

**Subject × stage χ² test**: χ²=137.8, dof=28, **p ≈ 1.9e-16**.
Stages are *not* subject-uniform. Notable patterns from the column-% table:
- **Prealgebra** concentrates in Stage 2 (14.8%) and is nearly absent from Stage 4 (0.3%) — consistent with its label being "easy".
- **Precalculus** concentrates in Stages 3-4 (17.5% / 15.3%) and is sparse in Stage 5 (7.4%) — model gets these mostly right or partially right, rarely 0/8.
- **Counting & Probability** is overrepresented in Stage 5 (20.1%) — a subject the model fails on more than average.
- This χ² result is *informative*, not a defect: stages reflect both difficulty and per-subject competence, which is exactly what a curriculum should capture.

**Comparison vs 1.5B C5-outliers**: old C5 outlier set has only 4 samples in
this pilot (most outliers were defined on the full 40K population) — overlap
with Qwen3-8B Stage 5 = 0. **Not** evidence of disagreement; just that the
overlap set is too small. A meaningful comparison requires the 40K extension.

---

## How to reproduce

CPU only.  All scripts mmap-load the 4.7 GB cache from disk — no GPU.

```bash
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
ROOT=/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b/nait

# Phase 0 (~15 min the first time, since it builds delta_cache.npy)
$PY $ROOT/direction_calibrated.py

# Phase A (~8-12 min)
$PY $ROOT/supervised_direction.py

# Phase B (~3-5 min)
$PY $ROOT/build_curriculum.py
```

---

## Decisions to revisit later

- **Per-layer windows vs single layer**: we average over top-5 layers; could
  swap to "best single layer" or "all 36 layers concatenated". The top-5
  average is the cheapest robust choice but loses inter-layer structure.
- **Subject "Other" class**: 8th subject is a catch-all → can hurt LDA. If
  Phase-A macro-F1 plateaus at ≈0.7, dropping the "Other" rows during LDA
  fitting may push it higher.
- **Pass=0 stage**: 22.7% of the pilot has pass_rate==0. The current rule
  assigns *all* of them to Stage 5. We might want to split pass=0 by
  truncation-vs-failure (Track-A `truncation_count`) — truncated solves are
  not necessarily "unreachable", they were just cut off at max_tokens.
- **40K extension**: same Phase-A directions can be applied to a larger
  population once Track-A pass_rate is computed for the full 40K. The
  per-sample feature is O(D)·O(top_k_layers) — trivial to recompute.
