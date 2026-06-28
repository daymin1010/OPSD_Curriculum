# OPSD stage build — 2026-06-22

CPU-only pure stage construction. Existing training/OPSD/extraction/prior npz artifacts were not modified.

## 0. Parameters / universe
- include_other: `False`
- n_stages: `5`
- construction: `tiered_difficulty_backbone_residual_within_tier`
- n_tiers_default: `3`
- tier_candidates: `[1, 2, 3, 4, 5, 6, 7, 8]`
- selected_n_tiers: `2`
- selected_rho_cond2_cond3: `0.6926605048858947`
- selected_stage_mean_levels: `[2.903414450103754, 2.7526041666666665, 3.9090119812467443, 5.168043148491488, 5.308839190628328]`
- selected_stage_mean_level_diffs: `[-0.15081028343708747, 1.1564078145800778, 1.259031167244744, 0.14079604213683972]`
- tier_gate_target_rho: `+0.4..+0.7`
- tier_gate_record: `{'n_tiers': 2, 'variant': 't0=Geometry|L3;t1=high', 'rho': 0.6926605048858947, 'p': 0.0, 'stage_mean_levels': [2.903414450103754, 2.7526041666666665, 3.9090119812467443, 5.168043148491488, 5.308839190628328], 'stage_mean_level_diffs': [-0.15081028343708747, 1.1564078145800778, 1.259031167244744, 0.14079604213683972], 'n_dips': 1, 'min_diff': -0.15081028343708747, 'rho_gate_ok': True, 'mean_level_gate_ok': True, 'gate_pass': True, 'penalty': 0.15266050488589467}`
- seeds: `[0, 1, 2]`
- W_SUBJ: `[9, 10, 11, 12, 14]`
- MIN_UNIT_N: `30`
- clusters: `{'C_alg': ['Algebra', 'Intermediate Algebra', 'Precalculus'], 'C_geo': ['Geometry'], 'C_disc': ['Counting & Probability', 'Number Theory', 'Prealgebra']}`
- relax_events: `0`
- elapsed_start_date: `2026-06-22`
- N problems: **28,771**
- units: **50**
- residual cluster pilot ARI(k=4): **+0.412**

## 1. Decision gate: tiered ③ C2
- selected n_tiers = **2**
- Spearman ρ(stage_cond2, stage_cond3) = **+0.6927** (target +0.4~+0.7; p=0)
- ③ stage mean levels = **[2.903, 2.753, 3.909, 5.168, 5.309]**
- mean-level diffs = `[-0.151, 1.156, 1.259, 0.141]`; dips=1; min_diff=-0.151
- gate_pass = **True**
- Interpretation: ③ should preserve an ascending difficulty trend while remaining clearly distinct from pure ② difficulty order.

## 2. ③ ours_C2: stage×subject / stage×level
### stage × subject
```
subject  Algebra  Counting & Probability  Geometry  Intermediate Algebra  Number Theory  Prealgebra  Precalculus
stage                                                                                                           
0           3079                       0      1190                   332              0           0          700
1           1805                    2341         0                     0              0        1998            0
2           1444                       0         0                  1957           1443         888           27
3            495                       0         0                     0           3970           0         1468
4              0                    1518      4076                     0             23          17            0
```
### stage × level
```
level    1     2     3     4     5     6    7   8
stage                                            
0      236  1764  1577  1724     0     0    0   0
1      106  2021  3304   713     0     0    0   0
2      977   445   909   464  2090   517  346  11
3        0     0     0  1856  1789  1723  565   0
4        0     0     0  1334  2109  1350  799  42
```
### stage mean level
| stage | mean_level | n |
|---|---:|---:|
| 0 | 2.903 | 5301 |
| 1 | 2.753 | 6144 |
| 2 | 3.909 | 5759 |
| 3 | 5.168 | 5933 |
| 4 | 5.309 | 5634 |

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
- consecutive stage residual distances: `[1.1296, 0.953, 0.3468, 1.3334]`
- mean consecutive distance = **0.9407**
- random unit-order mean±sd over 200 reps = **0.7211 ± 0.1204**

## 4. ③ stage-level distribution table (⑤ fixed spec)
```
level    1     2     3     4     5     6    7   8
stage                                            
0      236  1764  1577  1724     0     0    0   0
1      106  2021  3304   713     0     0    0   0
2      977   445   909   464  2090   517  346  11
3        0     0     0  1856  1789  1723  565   0
4        0     0     0  1334  2109  1350  799  42
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
