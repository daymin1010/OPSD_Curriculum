# Phase B — Curriculum Construction Report

- N samples              : 2666
- Feature dim            : 9  (1 lvl_proj + 7 subj + 1 pass_z)
- LEVEL window layers    : `[15, 16, 17, 18, 19]`
- SUBJECT window layers  : `[11, 12, 13, 14, 15]`
- Chosen K (silhouette)  : **7**  (sil=0.2645, BIC=50821)
- Wall time              : 0.2 min

## K selection

|      K |   silhouette |        bic |
|-------:|-------------:|-----------:|
| 3.0000 |       0.1448 | 54128.6244 |
| 4.0000 |       0.1676 | 52491.1370 |
| 5.0000 |       0.1925 | 47289.5739 |
| 6.0000 |       0.2354 | 46589.8454 |
| 7.0000 |       0.2645 | 50821.2618 |

## Stage sizes & means

|   stage |        n |   mean_pass |   mean_level |
|--------:|---------:|------------:|-------------:|
|   1.000 | 1063.000 |       0.973 |        2.958 |
|   2.000 |  332.000 |       0.537 |        2.934 |
|   3.000 |  332.000 |       0.453 |        5.036 |
|   4.000 |  333.000 |       0.316 |        6.706 |
|   5.000 |  606.000 |       0.000 |        5.284 |

## Subject × stage contingency (column %)

| subject                |     1 |     2 |     3 |     4 |     5 |
|:-----------------------|------:|------:|------:|------:|------:|
| Algebra                | 0.151 | 0.12  | 0.13  | 0.138 | 0.099 |
| Counting & Probability | 0.121 | 0.154 | 0.123 | 0.132 | 0.201 |
| Geometry               | 0.127 | 0.12  | 0.142 | 0.147 | 0.157 |
| Intermediate Algebra   | 0.146 | 0.096 | 0.117 | 0.138 | 0.107 |
| Number Theory          | 0.16  | 0.099 | 0.133 | 0.165 | 0.162 |
| Other                  | 0.078 | 0.136 | 0.105 | 0.123 | 0.144 |
| Prealgebra             | 0.086 | 0.148 | 0.075 | 0.003 | 0.056 |
| Precalculus            | 0.131 | 0.127 | 0.175 | 0.153 | 0.074 |

- χ² test:  χ²=137.76  dof=28  p=1.890e-16

## Sanity vs old 1.5B C5-outliers

- old C5 outliers = 4, Stage-5 = 606, overlap = 0

## Files

- `features.parquet`              : per-sample feature + cluster + stage
- `stage_{1..5}_manifest.parquet` : per-stage manifests
- `clustering_diagnostics.csv`    : K-scan silhouette + BIC
- `plots/cluster_scatter.png`     : PCA-2D of feature, coloured by cluster
- `plots/stage_subject_heatmap.png`: subject composition per stage

## Notes

- Stage layout is fixed (1=trivial, 5=unreachable, 2/3/4=tertile of `level - pass`).
- Cluster identity is *not* used directly to define stages, but is stored alongside
  to allow stratified sampling within a stage in later experiments.
- Subject window vs level window are deliberately taken from *different* layers
  (Phase-A finding: subject info concentrates lower, level info higher).