# OPSD stage build — 2026-06-22

CPU-only pure stage construction. Existing training/OPSD/extraction/prior npz artifacts were not modified.

## 0. Parameters / universe
- include_other: `False`
- n_stages: `5`
- subject_run_r: `2`
- seeds: `[0, 1, 2]`
- W_SUBJ: `[9, 10, 11, 12, 14]`
- MIN_UNIT_N: `30`
- clusters: `{'C_alg': ['Algebra', 'Intermediate Algebra', 'Precalculus'], 'C_geo': ['Geometry'], 'C_disc': ['Counting & Probability', 'Number Theory', 'Prealgebra']}`
- relax_events: `2`
- elapsed_start_date: `2026-06-22`
- N problems: **28,771**
- units: **50**
- residual cluster pilot ARI(k=4): **+0.412**

## 1. Gate: ②↔③ problem-level stage rank Spearman
- Spearman ρ(stage_cond2, stage_cond3) = **-0.6172** (p=0)
- Interpretation: too close to +1 means C2 remains near pure difficulty order; consider increasing `r` or transition pressure.

## 2. ③ ours_C2: stage×subject / stage×level
### stage × subject
```
subject  Algebra  Counting & Probability  Geometry  Intermediate Algebra  Number Theory  Prealgebra  Precalculus
stage                                                                                                           
0            244                     798      2614                   267           1621           0          195
1              0                     720      2259                   517           1271           0          518
2           4894                     713         0                     0           1101           0            0
3           1190                     961       393                   725              0        2365           27
4            495                     667         0                   780           1443         538         1455
```
### stage × level
```
level    1     2     3     4     5     6     7   8
stage                                             
0      181     0   797  1317     0  1708  1683  53
1        0     0     0     0  3898  1387     0   0
2        0     0  1805  3538  1365     0     0   0
3      925  3006   961    17   725     0    27   0
4      213  1224  2227  1219     0   495     0   0
```
### stage mean level
| stage | mean_level | n |
|---|---:|---:|
| 0 | 5.278 | 5739 |
| 1 | 5.262 | 5285 |
| 2 | 3.934 | 6708 |
| 3 | 2.420 | 5661 |
| 4 | 3.196 | 5378 |

## 2. ② difficulty: stage×subject / stage×level
### stage × subject
```
subject  Algebra  Counting & Probability  Geometry  Intermediate Algebra  Number Theory  Prealgebra  Precalculus
stage                                                                                                           
0           1561                     667       393                   101            534        2348          151
1           1769                     961       797                   231            909         538          549
2           1554                     713      1317                   464           1101          17          588
3           1365                     720      1389                   725           1271           0          284
4            574                     798      1370                   768           1621           0          623
```
### stage × level
```
level     1     2     3     4     5     6     7   8
stage                                              
0      1319  4230   206     0     0     0     0   0
1         0     0  5584   170     0     0     0   0
2         0     0     0  5754     0     0     0   0
3         0     0     0   167  5587     0     0   0
4         0     0     0     0   401  3590  1710  53
```
### stage mean level
| stage | mean_level | n |
|---|---:|---:|
| 0 | 1.807 | 5755 |
| 1 | 3.030 | 5754 |
| 2 | 4.000 | 5754 |
| 3 | 4.971 | 5754 |
| 4 | 6.246 | 5754 |

## 3. ③ residual stage-to-stage distance
- consecutive stage residual distances: `[0.2535, 0.6935, 0.44, 1.0962]`
- mean consecutive distance = **0.6208**
- random unit-order mean±sd over 200 reps = **0.7211 ± 0.1204**

## 4. ③ stage-level distribution table (⑤ fixed spec)
```
level    1     2     3     4     5     6     7   8
stage                                             
0      181     0   797  1317     0  1708  1683  53
1        0     0     0     0  3898  1387     0   0
2        0     0  1805  3538  1365     0     0   0
3      925  3006   961    17   725     0    27   0
4      213  1224  2227  1219     0   495     0   0
```

## 5. Missing/no-centroid unit assignment
| unit | assigned_centroid_unit | method |
|---|---|---|
| Intermediate Algebra|L1 | Algebra|L1 | same_cluster_level_nearest |
| Precalculus|L1 | Algebra|L1 | same_cluster_level_nearest |
| Prealgebra|L4 | Counting & Probability|L4 | same_cluster_level_nearest |
| Precalculus|L7 | Algebra|L7 | same_cluster_level_nearest |
| Counting & Probability|L8 | Counting & Probability|L7 | same_cluster_nearest |
| Geometry|L8 | Geometry|L7 | same_cluster_nearest |
| Intermediate Algebra|L8 | Algebra|L7 | same_cluster_nearest |
| Number Theory|L8 | Counting & Probability|L7 | same_cluster_nearest |

## 6. Output files
- `stages_cond1_random_seed0.json`
- `stages_cond1_random_seed1.json`
- `stages_cond1_random_seed2.json`
- `stages_cond2_diff.json`
- `stages_cond3_ours_C2.json`
- `stages_cond4_shuffle_seed0.json`
- `stages_cond4_shuffle_seed1.json`
- `stages_cond4_shuffle_seed2.json`
- `stages_cond5_diffmatched_seed0.json`
- `stages_cond5_diffmatched_seed1.json`
- `stages_cond5_diffmatched_seed2.json`
- `manifest.json`
- `stagebuild_artifacts.npz`
