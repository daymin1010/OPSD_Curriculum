# ΔA Residualization Ablation (PCA top-K removal) — pilot

- N = **1608**, K values tested = [0, 1, 2]

## METHOD
residualize_analysis.py
=======================
ABLATION on the common-component removal step of similarity_analysis.py.

The primary analysis used GLOBAL-MEAN centering (subtract μ = mean_i ΔA_i).
Here we test a more aggressive removal: strip the top-K PRINCIPAL COMPONENTS
of ΔA (per layer) instead of just the mean. The hypothesis: a single shared
"reasoning-shift" axis dominates ΔA; removing 1-2 PCs should make the
subject/level separability gap *larger and cleaner* than mean-centering.

Per layer l (36) and a chosen ΔA (THINKING / FAITHFUL):
  X = ΔA[:, l, :]                      (N, 12288)
  X0 = X - mean(X)                     (also remove mean, like centering)
  V  = top-K right singular vectors of X0   (K, 12288)
  X_res = X0 - (X0 @ V^T) @ V          (project out the top-K shared axes)
We then run the SAME subject/level groupings (centroid cosine, within/between
gap, permutation p, LEVEL ordinality) as similarity_analysis.py, reusing its
helpers, and tabulate gap/p/ordinality for K ∈ {0(mean-only), 1, 2}.

K=0 reproduces the primary mean-centered numbers (sanity cross-check).

CPU only. Reuses load + grouping from similarity_analysis.py (same dir).
Outputs:
  - REPORT_residualize_<tag>.md
  - residualize_summary_<tag>.csv

## ===== THINKING :: residualize K=0 (mean-only) =====

### [THINKING_resid0] SUBJECT grouping  (groups=8, MIN_N=1)
group sizes: Algebra:210, Counting & Probability:220, Geometry:219, Intermediate Algebra:207, Number Theory:233, Other:217, Prealgebra:107, Precalculus:195
- within_mean cos = +0.226 | between_mean cos = -0.127 | gap = +0.353
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
                        AlgebraCounting & ProbabilityGeometryIntermediate AlgebraNumber Theory   OtherPrealgebraPrecalculus
                Algebra   1.000  -0.389  -0.134   0.286  -0.117  -0.298   0.278   0.035
 Counting & Probability  -0.389   1.000   0.117  -0.599   0.498  -0.370   0.079  -0.677
               Geometry  -0.134   0.117   1.000  -0.230  -0.112  -0.425   0.066  -0.006
   Intermediate Algebra   0.286  -0.599  -0.230   1.000  -0.318   0.032  -0.449   0.636
          Number Theory  -0.117   0.498  -0.112  -0.318   1.000  -0.439   0.015  -0.647
                  Other  -0.298  -0.370  -0.425   0.032  -0.439   1.000  -0.300   0.253
             Prealgebra   0.278   0.079   0.066  -0.449   0.015  -0.300   1.000  -0.351
            Precalculus   0.035  -0.677  -0.006   0.636  -0.647   0.253  -0.351   1.000
```

### [THINKING_resid0] LEVEL grouping  (groups=8, MIN_N=1)
group sizes: 1:208, 2:240, 3:240, 4:227, 5:210, 6:210, 7:207, 8:66
- within_mean cos = +0.324 | between_mean cos = -0.108 | gap = +0.432
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
               1       2       3       4       5       6       7       8
       1   1.000   0.906   0.570  -0.323  -0.864  -0.928  -0.808  -0.674
       2   0.906   1.000   0.772  -0.104  -0.772  -0.953  -0.904  -0.822
       3   0.570   0.772   1.000   0.378  -0.370  -0.688  -0.902  -0.889
       4  -0.323  -0.104   0.378   1.000   0.483   0.159  -0.196  -0.317
       5  -0.864  -0.772  -0.370   0.483   1.000   0.829   0.546   0.375
       6  -0.928  -0.953  -0.688   0.159   0.829   1.000   0.830   0.713
       7  -0.808  -0.904  -0.902  -0.196   0.546   0.830   1.000   0.923
       8  -0.674  -0.822  -0.889  -0.317   0.375   0.713   0.923   1.000
```
- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = +0.862  (positive => adjacent levels more similar)

## ===== THINKING :: residualize K=1 (mean + top-1 PCs removed) =====

### [THINKING_resid1] SUBJECT grouping  (groups=8, MIN_N=1)
group sizes: Algebra:210, Counting & Probability:220, Geometry:219, Intermediate Algebra:207, Number Theory:233, Other:217, Prealgebra:107, Precalculus:195
- within_mean cos = +0.234 | between_mean cos = -0.126 | gap = +0.359
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
                        AlgebraCounting & ProbabilityGeometryIntermediate AlgebraNumber Theory   OtherPrealgebraPrecalculus
                Algebra   1.000  -0.187  -0.083   0.253   0.083  -0.367   0.068  -0.069
 Counting & Probability  -0.187   1.000   0.074  -0.619   0.438  -0.416   0.344  -0.689
               Geometry  -0.083   0.074   1.000  -0.221  -0.162  -0.429   0.133   0.013
   Intermediate Algebra   0.253  -0.619  -0.221   1.000  -0.284   0.028  -0.567   0.633
          Number Theory   0.083   0.438  -0.162  -0.284   1.000  -0.470   0.171  -0.640
                  Other  -0.367  -0.416  -0.429   0.028  -0.470   1.000  -0.316   0.261
             Prealgebra   0.068   0.344   0.133  -0.567   0.171  -0.316   1.000  -0.495
            Precalculus  -0.069  -0.689   0.013   0.633  -0.640   0.261  -0.495   1.000
```

### [THINKING_resid1] LEVEL grouping  (groups=8, MIN_N=1)
group sizes: 1:208, 2:240, 3:240, 4:227, 5:210, 6:210, 7:207, 8:66
- within_mean cos = +0.250 | between_mean cos = -0.100 | gap = +0.350
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
               1       2       3       4       5       6       7       8
       1   1.000   0.812  -0.002  -0.599  -0.846  -0.848  -0.616  -0.266
       2   0.812   1.000   0.316  -0.388  -0.775  -0.894  -0.723  -0.475
       3  -0.002   0.316   1.000   0.408  -0.056  -0.238  -0.613  -0.596
       4  -0.599  -0.388   0.408   1.000   0.592   0.391  -0.045  -0.267
       5  -0.846  -0.775  -0.056   0.592   1.000   0.807   0.440   0.078
       6  -0.848  -0.894  -0.238   0.391   0.807   1.000   0.620   0.304
       7  -0.616  -0.723  -0.613  -0.045   0.440   0.620   1.000   0.676
       8  -0.266  -0.475  -0.596  -0.267   0.078   0.304   0.676   1.000
```
- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = +0.826  (positive => adjacent levels more similar)

## ===== THINKING :: residualize K=2 (mean + top-2 PCs removed) =====

### [THINKING_resid2] SUBJECT grouping  (groups=8, MIN_N=1)
group sizes: Algebra:210, Counting & Probability:220, Geometry:219, Intermediate Algebra:207, Number Theory:233, Other:217, Prealgebra:107, Precalculus:195
- within_mean cos = +0.238 | between_mean cos = -0.125 | gap = +0.363
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
                        AlgebraCounting & ProbabilityGeometryIntermediate AlgebraNumber Theory   OtherPrealgebraPrecalculus
                Algebra   1.000  -0.196  -0.023   0.217   0.074  -0.416   0.155  -0.044
 Counting & Probability  -0.196   1.000   0.027  -0.608   0.404  -0.377   0.364  -0.682
               Geometry  -0.023   0.027   1.000  -0.193  -0.191  -0.396   0.106  -0.001
   Intermediate Algebra   0.217  -0.608  -0.193   1.000  -0.267  -0.012  -0.575   0.641
          Number Theory   0.074   0.404  -0.191  -0.267   1.000  -0.455   0.173  -0.629
                  Other  -0.416  -0.377  -0.396  -0.012  -0.455   1.000  -0.308   0.255
             Prealgebra   0.155   0.364   0.106  -0.575   0.173  -0.308   1.000  -0.547
            Precalculus  -0.044  -0.682  -0.001   0.641  -0.629   0.255  -0.547   1.000
```

### [THINKING_resid2] LEVEL grouping  (groups=8, MIN_N=1)
group sizes: 1:208, 2:240, 3:240, 4:227, 5:210, 6:210, 7:207, 8:66
- within_mean cos = +0.186 | between_mean cos = -0.122 | gap = +0.308
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
               1       2       3       4       5       6       7       8
       1   1.000   0.381  -0.255  -0.513  -0.698  -0.534  -0.316   0.029
       2   0.381   1.000   0.334  -0.026  -0.496  -0.725  -0.585  -0.341
       3  -0.255   0.334   1.000   0.549   0.067  -0.236  -0.617  -0.541
       4  -0.513  -0.026   0.549   1.000   0.439   0.035  -0.392  -0.506
       5  -0.698  -0.496   0.067   0.439   1.000   0.571   0.116  -0.237
       6  -0.534  -0.725  -0.236   0.035   0.571   1.000   0.417   0.080
       7  -0.316  -0.585  -0.617  -0.392   0.116   0.417   1.000   0.582
       8   0.029  -0.341  -0.541  -0.506  -0.237   0.080   0.582   1.000
```
- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = +0.808  (positive => adjacent levels more similar)

## ===== FAITHFUL :: residualize K=0 (mean-only) =====

### [FAITHFUL_resid0] SUBJECT grouping  (groups=8, MIN_N=1)
group sizes: Algebra:210, Counting & Probability:220, Geometry:219, Intermediate Algebra:207, Number Theory:233, Other:217, Prealgebra:107, Precalculus:195
- within_mean cos = +0.170 | between_mean cos = -0.125 | gap = +0.295
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
                        AlgebraCounting & ProbabilityGeometryIntermediate AlgebraNumber Theory   OtherPrealgebraPrecalculus
                Algebra   1.000  -0.664  -0.710   0.648  -0.637   0.158   0.626   0.349
 Counting & Probability  -0.664   1.000   0.433  -0.669   0.621  -0.422  -0.404  -0.630
               Geometry  -0.710   0.433   1.000  -0.640   0.503  -0.372  -0.553  -0.392
   Intermediate Algebra   0.648  -0.669  -0.640   1.000  -0.629   0.225   0.360   0.489
          Number Theory  -0.637   0.621   0.503  -0.629   1.000  -0.559  -0.514  -0.622
                  Other   0.158  -0.422  -0.372   0.225  -0.559   1.000   0.020   0.383
             Prealgebra   0.626  -0.404  -0.553   0.360  -0.514   0.020   1.000   0.115
            Precalculus   0.349  -0.630  -0.392   0.489  -0.622   0.383   0.115   1.000
```

### [FAITHFUL_resid0] LEVEL grouping  (groups=8, MIN_N=1)
group sizes: 1:208, 2:240, 3:240, 4:227, 5:210, 6:210, 7:207, 8:66
- within_mean cos = +0.285 | between_mean cos = -0.122 | gap = +0.406
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
               1       2       3       4       5       6       7       8
       1   1.000   0.964   0.885   0.291  -0.743  -0.946  -0.926  -0.888
       2   0.964   1.000   0.959   0.445  -0.653  -0.970  -0.977  -0.949
       3   0.885   0.959   1.000   0.562  -0.529  -0.940  -0.974  -0.959
       4   0.291   0.445   0.562   1.000   0.038  -0.452  -0.546  -0.573
       5  -0.743  -0.653  -0.529   0.038   1.000   0.616   0.546   0.507
       6  -0.946  -0.970  -0.940  -0.452   0.616   1.000   0.941   0.913
       7  -0.926  -0.977  -0.974  -0.546   0.546   0.941   1.000   0.953
       8  -0.888  -0.949  -0.959  -0.573   0.507   0.913   0.953   1.000
```
- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = +0.860  (positive => adjacent levels more similar)

## ===== FAITHFUL :: residualize K=1 (mean + top-1 PCs removed) =====

### [FAITHFUL_resid1] SUBJECT grouping  (groups=8, MIN_N=1)
group sizes: Algebra:210, Counting & Probability:220, Geometry:219, Intermediate Algebra:207, Number Theory:233, Other:217, Prealgebra:107, Precalculus:195
- within_mean cos = +0.162 | between_mean cos = -0.135 | gap = +0.296
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
                        AlgebraCounting & ProbabilityGeometryIntermediate AlgebraNumber Theory   OtherPrealgebraPrecalculus
                Algebra   1.000  -0.300  -0.139   0.183  -0.031  -0.243   0.003  -0.051
 Counting & Probability  -0.300   1.000  -0.184  -0.411   0.253  -0.286   0.148  -0.499
               Geometry  -0.139  -0.184   1.000  -0.218  -0.250  -0.176   0.100  -0.071
   Intermediate Algebra   0.183  -0.411  -0.218   1.000  -0.240   0.004  -0.324   0.278
          Number Theory  -0.031   0.253  -0.250  -0.240   1.000  -0.492   0.101  -0.479
                  Other  -0.243  -0.286  -0.176   0.004  -0.492   1.000  -0.370   0.297
             Prealgebra   0.003   0.148   0.100  -0.324   0.101  -0.370   1.000  -0.378
            Precalculus  -0.051  -0.499  -0.071   0.278  -0.479   0.297  -0.378   1.000
```

### [FAITHFUL_resid1] LEVEL grouping  (groups=8, MIN_N=1)
group sizes: 1:208, 2:240, 3:240, 4:227, 5:210, 6:210, 7:207, 8:66
- within_mean cos = +0.203 | between_mean cos = -0.116 | gap = +0.319
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
               1       2       3       4       5       6       7       8
       1   1.000   0.868   0.169  -0.619  -0.758  -0.715  -0.609  -0.283
       2   0.868   1.000   0.321  -0.516  -0.723  -0.724  -0.667  -0.348
       3   0.169   0.321   1.000  -0.033  -0.196  -0.321  -0.423  -0.288
       4  -0.619  -0.516  -0.033   1.000   0.465   0.333   0.157  -0.042
       5  -0.758  -0.723  -0.196   0.465   1.000   0.498   0.343   0.126
       6  -0.715  -0.724  -0.321   0.333   0.498   1.000   0.380   0.152
       7  -0.609  -0.667  -0.423   0.157   0.343   0.380   1.000   0.208
       8  -0.283  -0.348  -0.288  -0.042   0.126   0.152   0.208   1.000
```
- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = +0.744  (positive => adjacent levels more similar)

## ===== FAITHFUL :: residualize K=2 (mean + top-2 PCs removed) =====

### [FAITHFUL_resid2] SUBJECT grouping  (groups=8, MIN_N=1)
group sizes: Algebra:210, Counting & Probability:220, Geometry:219, Intermediate Algebra:207, Number Theory:233, Other:217, Prealgebra:107, Precalculus:195
- within_mean cos = +0.157 | between_mean cos = -0.137 | gap = +0.294
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
                        AlgebraCounting & ProbabilityGeometryIntermediate AlgebraNumber Theory   OtherPrealgebraPrecalculus
                Algebra   1.000  -0.307  -0.091   0.158  -0.070  -0.208  -0.058  -0.053
 Counting & Probability  -0.307   1.000  -0.214  -0.397   0.192  -0.245   0.131  -0.464
               Geometry  -0.091  -0.214   1.000  -0.180  -0.278  -0.207   0.111  -0.050
   Intermediate Algebra   0.158  -0.397  -0.180   1.000  -0.227  -0.030  -0.345   0.249
          Number Theory  -0.070   0.192  -0.278  -0.227   1.000  -0.409   0.075  -0.443
                  Other  -0.208  -0.245  -0.207  -0.030  -0.409   1.000  -0.328   0.223
             Prealgebra  -0.058   0.131   0.111  -0.345   0.075  -0.328   1.000  -0.358
            Precalculus  -0.053  -0.464  -0.050   0.249  -0.443   0.223  -0.358   1.000
```

### [FAITHFUL_resid2] LEVEL grouping  (groups=8, MIN_N=1)
group sizes: 1:208, 2:240, 3:240, 4:227, 5:210, 6:210, 7:207, 8:66
- within_mean cos = +0.174 | between_mean cos = -0.125 | gap = +0.298
- permutation p(gap >= obs) = 0.0050  (N_PERM=200, label shuffle)
- centroid cosine matrix:
```
               1       2       3       4       5       6       7       8
       1   1.000   0.791  -0.035  -0.523  -0.659  -0.588  -0.493  -0.212
       2   0.791   1.000   0.205  -0.410  -0.627  -0.617  -0.572  -0.287
       3  -0.035   0.205   1.000   0.075  -0.065  -0.225  -0.346  -0.230
       4  -0.523  -0.410   0.075   1.000   0.331   0.170   0.007  -0.093
       5  -0.659  -0.627  -0.065   0.331   1.000   0.333   0.178   0.021
       6  -0.588  -0.617  -0.225   0.170   0.333   1.000   0.210   0.055
       7  -0.493  -0.572  -0.346   0.007   0.178   0.210   1.000   0.116
       8  -0.212  -0.287  -0.230  -0.093   0.021   0.055   0.116   1.000
```
- ORDINALITY: ρ( centroid_cos , -|level_a-level_b| ) = +0.776  (positive => adjacent levels more similar)

## ===== SUMMARY (gap & significance vs K) =====
```
dA         K    group   within  between      gap        p  ord_rho
THINKING   0  subject    0.226   -0.127    0.353   0.0050     -   
THINKING   0    level    0.324   -0.108    0.432   0.0050   +0.862
THINKING   1  subject    0.234   -0.126    0.359   0.0050     -   
THINKING   1    level    0.250   -0.100    0.350   0.0050   +0.826
THINKING   2  subject    0.238   -0.125    0.363   0.0050     -   
THINKING   2    level    0.186   -0.122    0.308   0.0050   +0.808
FAITHFUL   0  subject    0.170   -0.125    0.295   0.0050     -   
FAITHFUL   0    level    0.285   -0.122    0.406   0.0050   +0.860
FAITHFUL   1  subject    0.162   -0.135    0.296   0.0050     -   
FAITHFUL   1    level    0.203   -0.116    0.319   0.0050   +0.744
FAITHFUL   2  subject    0.157   -0.137    0.294   0.0050     -   
FAITHFUL   2    level    0.174   -0.125    0.298   0.0050   +0.776
```