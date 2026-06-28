# Track A — Replication Comparison: pilot1 vs pilot2 (RECOMPUTED)

> **Framing.** group-centroid cosine, global-mean centering, level ordinality, and the label-permutation test are OUR diagnostic for "does ΔA carry (subject/level) structure?" — NOT the NAIT paper's method. NAIT's PCA-direction scoring is applied later in the curriculum-direction stage. Track C (supervised difficulty direction) is NAIT-inspired but supervised.

**Replication design:** each pilot centered by its OWN mean (μ_pilot1 / μ_pilot2); permutations shuffle labels WITHIN each pilot. No pooling / no cross-centering (that is Track C only).

**N_PERM (per grouping)** = subject:1000, level:1000, unit:200 (unit has ~50 groups => costlier per perm; 200 gives p-resolution 0.005, ample for a descriptive grouping). metric source = recomputed via similarity_analysis.py functions (identical method).

## Population (filters applied identically to both)

| | pilot1 | pilot2 |
|---|---|---|
| shifts dir | `outputs/pilot/shifts` | `outputs/pilot2/shifts` |
| .pt loaded | 1608 | 1417 |
| non-finite ΔA dropped | 0 | 0 |
| outside L1–L7 dropped | 66 | 0 |
| **final N (L1–L7)** | **1542** | **1417** |
| levels | [1, 2, 3, 4, 5, 6, 7] | [1, 2, 3, 4, 5, 6, 7] |
| subjects | 8 | 8 |
| units (n≥10) | 53 | 47 |

> Note: the older pilot1 report quoted N=1541 because it was run on an earlier 1541-file snapshot; the loader applies no content filter, so current N = loadable .pt count. This recompute supersedes it.

## Main comparison (within / between / gap / perm-p)

| MODE | group | G(p1/p2) | within p1/p2 | between p1/p2 | **gap** p1/p2 (Δ) | perm-p p1/p2 | offdiag p1/p2 | levelρ p1/p2 |
|---|---|---|---|---|---|---|---|---|
| THINKING | subject | 8/8 | +0.229 / +0.251 | -0.127 / -0.113 | **+0.356 / +0.364** (Δ+0.008) | 0.0010 / 0.0010 | -0.127 / -0.113 | — |
| THINKING | level | 7/7 | +0.305 / +0.323 | -0.127 / -0.130 | **+0.433 / +0.453** (Δ+0.020) | 0.0010 / 0.0010 | -0.127 / -0.130 | +0.896 / +0.879 |
| THINKING | unit | 53/47 | +0.435 / +0.428 | -0.013 / -0.005 | **+0.447 / +0.433** (Δ-0.014) | 0.0050 / 0.0050 | -0.013 / -0.005 | — |
| FAITHFUL | subject | 8/8 | +0.165 / +0.154 | -0.127 / -0.118 | **+0.292 / +0.272** (Δ-0.020) | 0.0010 / 0.0010 | -0.127 / -0.118 | — |
| FAITHFUL | level | 7/7 | +0.275 / +0.311 | -0.134 / -0.134 | **+0.409 / +0.445** (Δ+0.037) | 0.0010 / 0.0010 | -0.134 / -0.134 | +0.871 / +0.847 |
| FAITHFUL | unit | 53/47 | +0.374 / +0.360 | +0.011 / +0.011 | **+0.363 / +0.349** (Δ-0.013) | 0.0050 / 0.0050 | +0.011 / +0.011 | — |

## Reference heuristic (NOT a conclusion — judge from raw numbers above)

Screening rule (loose): gap same sign + |Δgap|/max(|gap|) < 0.5 + both perm-p < 0.05; level: ordinality ρ same sign. `|Δgap|/max<0.5` is permissive — final call is human, from the raw within/between/gap/perm-p columns.

- [THINKING/subject] PASS(참고): gap同부호, 크기근접, 둘다유의
- [THINKING/level] PASS(참고): gap同부호, 크기근접, 둘다유의 | ordinalityρ 同부호 (+0.90/+0.88)
- [THINKING/unit] PASS(참고): gap同부호, 크기근접, 둘다유의
- [FAITHFUL/subject] PASS(참고): gap同부호, 크기근접, 둘다유의
- [FAITHFUL/level] PASS(참고): gap同부호, 크기근접, 둘다유의 | ordinalityρ 同부호 (+0.87/+0.85)
- [FAITHFUL/unit] PASS(참고): gap同부호, 크기근접, 둘다유의
