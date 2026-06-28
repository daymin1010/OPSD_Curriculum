# Qwen3-8B Unit Prototype Similarity (raw=과거 / resid=현재)

## Method (REF: 4.6_Task2 nait_unit_similarity)
- v_u^l = sign-calibrated PC1 of {Δ𝒜_s^l : s∈u} (unit-norm, Eq.4).
- Per-layer sim:  S_l[u,u'] = v_u^l · v_{u'}^l  (= cos).
- Aggregated sim: S_agg[u,u'] = mean_l v_u^l · v_{u'}^l.
- **raw (과거)**: prototypes on raw Δ𝒜.  **resid (현재)**: after removing per-layer global PC1.
- PC1 via torch.pca_lowrank(q=6) for speed (equivalent leading component).
- U=51, L=36, N=2666, D=12288.

## raw Δ𝒜  (과거)

- **Subject block** (avg cos): within = **0.408**, across = **0.425**, ratio = **0.96x**
- **Level   block** (avg cos): within = **0.637**, across = **0.427**, ratio = **1.49x**

### Top 15 most similar unit pairs (aggregated cos)

| cos | u1 | u2 | same subj? | same lvl? |
|---|---|---|---|---|
| 0.942 | Algebra_L7 | NT_L8 | · | · |
| 0.937 | C&P_L6 | NT_L8 | · | · |
| 0.936 | NT_L7 | NT_L8 | ✓ | · |
| 0.934 | C&P_L7 | NT_L8 | · | · |
| 0.927 | Algebra_L7 | C&P_L6 | · | · |
| 0.926 | C&P_L6 | C&P_L7 | ✓ | · |
| 0.924 | Algebra_L7 | C&P_L7 | · | ✓ |
| 0.923 | Geom_L6 | IA_L7 | · | · |
| 0.923 | Algebra_L6 | IA_L7 | · | · |
| 0.922 | Algebra_L6 | NT_L8 | · | · |
| 0.921 | Algebra_L6 | NT_L5 | · | · |
| 0.921 | IA_L7 | NT_L5 | · | · |
| 0.919 | C&P_L7 | NT_L7 | · | ✓ |
| 0.918 | Algebra_L6 | NT_L7 | · | · |
| 0.918 | NT_L7 | Pcalc_L7 | · | ✓ |

### Bottom 10 most DIS-similar pairs

| cos | u1 | u2 |
|---|---|---|
| -0.032 | C&P_L7 | Geom_L1 |
| -0.025 | IA_L2 | Prealg_L2 |
| -0.021 | IA_L2 | IA_L4 |
| -0.021 | Geom_L1 | Geom_L8 |
| -0.021 | Algebra_L7 | Geom_L1 |
| -0.019 | C&P_L8 | Geom_L1 |
| -0.019 | IA_L2 | Pcalc_L7 |
| -0.018 | C&P_L7 | NT_L2 |
| -0.015 | IA_L2 | IA_L8 |
| -0.015 | Geom_L1 | Geom_L6 |

## resid Δ𝒜 (현재)

- **Subject block** (avg cos): within = **0.163**, across = **0.156**, ratio = **1.05x**
- **Level   block** (avg cos): within = **0.280**, across = **0.151**, ratio = **1.86x**

### Top 15 most similar unit pairs (aggregated cos)

| cos | u1 | u2 | same subj? | same lvl? |
|---|---|---|---|---|
| 0.785 | Geom_L2 | NT_L1 | · | · |
| 0.777 | NT_L1 | Pcalc_L1 | · | ✓ |
| 0.776 | C&P_L1 | Geom_L2 | · | · |
| 0.773 | IA_L1 | Pcalc_L1 | · | ✓ |
| 0.741 | NT_L2 | Pcalc_L2 | · | ✓ |
| 0.727 | Geom_L2 | IA_L1 | · | · |
| 0.724 | C&P_L1 | NT_L1 | · | ✓ |
| 0.701 | Geom_L2 | Pcalc_L1 | · | · |
| 0.681 | IA_L1 | NT_L1 | · | ✓ |
| 0.673 | IA_L1 | NT_L2 | · | · |
| 0.667 | Geom_L1 | Geom_L2 | ✓ | · |
| 0.649 | Geom_L1 | NT_L1 | · | ✓ |
| 0.649 | IA_L1 | Pcalc_L2 | · | · |
| 0.642 | C&P_L1 | IA_L1 | · | ✓ |
| 0.635 | C&P_L1 | Pcalc_L1 | · | ✓ |

### Bottom 10 most DIS-similar pairs

| cos | u1 | u2 |
|---|---|---|
| -0.167 | Prealg_L1 | Prealg_L4 |
| -0.115 | Algebra_L1 | Prealg_L4 |
| -0.113 | Prealg_L1 | Prealg_L2 |
| -0.106 | NT_L7 | Prealg_L1 |
| -0.103 | Algebra_L1 | Pcalc_L5 |
| -0.102 | Prealg_L1 | Pcalc_L5 |
| -0.101 | C&P_L6 | Prealg_L1 |
| -0.100 | C&P_L7 | Prealg_L1 |
| -0.100 | Algebra_L1 | NT_L7 |
| -0.098 | IA_L2 | IA_L8 |

## Figures
- `unit_sim_layer_{raw,resid}_L{3,18,30}.png`
- `unit_sim_agg_{raw,resid}.png`
- `unit_sim_block_subject_{raw,resid}.png`
- `unit_sim_block_level_{raw,resid}.png`
- `unit_sim_dendrogram_{raw,resid}.png`
