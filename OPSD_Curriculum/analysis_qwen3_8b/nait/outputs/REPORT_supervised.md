# Phase A — Supervised Direction Report

- N samples : 2666
- N layers  : 36
- D hidden  : 12288
- PCA dim used for fitting : 256
- CV folds  : 5
- Wall time : 14.6 min

## Top layers — Subject (LDA macro-F1)

| rank | layer | macro-F1 |
|---:|---:|---:|
| 1 | 14 | 0.779 |
| 2 | 12 | 0.776 |
| 3 | 15 | 0.766 |
| 4 | 13 | 0.764 |
| 5 | 11 | 0.757 |

## Top layers — Level (ridge R²)

| rank | layer | R² |
|---:|---:|---:|
| 1 | 18 | +0.879 |
| 2 | 17 | +0.879 |
| 3 | 19 | +0.876 |
| 4 | 16 | +0.873 |
| 5 | 15 | +0.865 |

## Top layers — Level (LDA-1 |ρ|)

| rank | layer | ρ |
|---:|---:|---:|
| 1 | 23 | +0.892 |
| 2 | 18 | +0.891 |
| 3 | 20 | +0.890 |
| 4 | 16 | +0.889 |
| 5 | 19 | +0.889 |

## Recommended layer windows

- **Subject window** (top-5 by macro-F1): `[11, 12, 13, 14, 15]`
- **Level window**   (top-5 by ridge R²): `[15, 16, 17, 18, 19]`

## Cosine similarity vs Phase-0 calibrated PC1 (at best-level layer)

- layer = **18**
- |cos(PC1, v_level_ridge)| = **0.107**
- |cos(PC1, v_level_LDA)|   = 0.075
- mean |cos(PC1, W_subject)| (7 axes) = 0.004

## Files

- `supervised_per_layer.csv` — per-layer metrics table
- `lda_directions.npz`        — W_subj[L,D,7], v_lvl_lda[L,D], v_lvl_ridge[L,D]
- `plots/layer_window_compare.png` — F1 / R² vs layer

## Notes

- LDA / ridge are fit on PCA-256 reduced features, then back-projected to D=12288.
- Comparing the level-direction with calibrated PC1 quantifies how aligned the
  unsupervised difficulty axis is with the *supervised* difficulty axis.
- The subject window vs level window separation motivates the Phase-B feature design
  (concat of subject-projected ΔA in subj_window, level-projected ΔA in lvl_window).