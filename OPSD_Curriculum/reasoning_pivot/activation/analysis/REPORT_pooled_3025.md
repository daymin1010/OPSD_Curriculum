# Phase 1 — POOLED (pilot1 + pilot2) ΔA Group-Similarity (canonical)

> **Framing.** group-centroid layer-averaged cosine, pooled-mean centering, level ordinality, and the label-permutation test are OUR diagnostic for "does ΔA carry (subject/level/unit) structure?" — NOT the NAIT paper's PCA-direction scoring (applied later in the curriculum-direction stage; Track C supervised direction is NAIT-inspired but supervised).

**Pooled design (OPPOSITE of the Track-A replication).** Both pilots are MERGED into one population and centered by a SINGLE pooled global mean (μ_pooled, per layer). Permutations shuffle labels across the WHOLE pooled set. This is the canonical / main analysis. (Per-pilot self-centering + within-pilot permutation is the replication track in `REPORT_pilot2_comparison.md` — do not confuse the two handlings.)

**N_PERM (per grouping):** subject:1000, level:1000, unit:200. metric source = recomputed via similarity_analysis.py functions (identical method to the per-pilot reports).

## Population (canonical N)

- raw .pt loaded (pilot1+pilot2) = **3025**
- non-finite ΔA dropped = **0** (expected 0)
- **finite pooled N = 3025** (canonical; '3000' is a nickname only — always report raw / finite / per-filter analysis-N)
- provenance: pilot1=1608, pilot2=1417

level counts (pooled finite):

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| n | 335 | 480 | 480 | 437 | 420 | 420 | 387 | 66 |

### L8 caveat (read before any L8 reading)

- L8 total = **66**; provenance: pilot1=66 (i.e. **L8 is entirely from pilot1**).
- L8 exists in only 5/8 subjects: Number Theory:23, Other:13, Intermediate Algebra:11, Counting & Probability:10, Geometry:9.
- Algebra / Prealgebra / Precalculus have L8 = 0 (handoff quirk #4).
- => **NO standalone L8 conclusions** (n small + subject imbalance). We report level grouping BOTH with (L1–L8) and without (L1–L7) the L8 cell so its effect is isolated.

## Main results (within / between / gap / perm-p)

| MODE | grouping | G | analysis-N | within | between | **gap** | perm-p | offdiag | level ρ |
|---|---|---|---|---|---|---|---|---|---|
| THINKING | subject | 8 | 3025 | +0.232 | -0.121 | **+0.353** | 0.0010 | -0.121 | — |
| THINKING | level (L1-L8, incl. n=66 L8) | 8 | 3025 | +0.324 | -0.110 | **+0.434** | 0.0010 | -0.110 | +0.841 |
| THINKING | level (L1-L7, L8 dropped) | 7 | 2959 | +0.312 | -0.121 | **+0.434** | 0.0010 | -0.121 | +0.896 |
| THINKING | unit (subject x level, n>=MIN_N) | 57 | 3025 | +0.430 | -0.010 | **+0.440** | 0.0050 | -0.010 | — |
| FAITHFUL | subject | 8 | 3025 | +0.141 | -0.121 | **+0.263** | 0.0010 | -0.121 | — |
| FAITHFUL | level (L1-L8, incl. n=66 L8) | 8 | 3025 | +0.282 | -0.125 | **+0.407** | 0.0010 | -0.125 | +0.826 |
| FAITHFUL | level (L1-L7, L8 dropped) | 7 | 2959 | +0.301 | -0.130 | **+0.430** | 0.0010 | -0.130 | +0.859 |
| FAITHFUL | unit (subject x level, n>=MIN_N) | 57 | 3025 | +0.359 | +0.003 | **+0.356** | 0.0050 | +0.003 | — |

Interpretation guide: gap = within − between (higher => groups internally coherent & mutually distinct); perm-p = P(gap_perm ≥ obs) under label shuffle; level ρ>0 => adjacent levels more similar (ordinality). Compare the L1–L8 vs L1–L7 level rows to see how the small L8 cell moves the level structure.

## Group sizes (per grouping, THINKING; identical to FAITHFUL)

- **subject** (G=8, N=3025): Algebra:420, Counting & Probability:430, Geometry:406, Intermediate Algebra:387, Number Theory:443, Other:397, Prealgebra:197, Precalculus:345
- **level (L1-L8, incl. n=66 L8)** (G=8, N=3025): 1:335, 2:480, 3:480, 4:437, 5:420, 6:420, 7:387, 8:66
- **level (L1-L7, L8 dropped)** (G=7, N=2959): 1:335, 2:480, 3:480, 4:437, 5:420, 6:420, 7:387
- **unit (subject x level, n>=MIN_N)** (G=57, N=3025): Algebra|L1:60, Algebra|L2:60, Algebra|L3:60, Algebra|L4:60, Algebra|L5:60, Algebra|L6:60, Algebra|L7:60, Counting & Probability|L1:60, Counting & Probability|L2:60, Counting & Probability|L3:60, Counting & Probability|L4:60, Counting & Probability|L5:60, Counting & Probability|L6:60, Counting & Probability|L7:60, Counting & Probability|L8:10, Geometry|L1:37, Geometry|L2:60, Geometry|L3:60, Geometry|L4:60, Geometry|L5:60, Geometry|L6:60, Geometry|L7:60, Intermediate Algebra|L1:16, Intermediate Algebra|L2:60, Intermediate Algebra|L3:60, Intermediate Algebra|L4:60, Intermediate Algebra|L5:60, Intermediate Algebra|L6:60, Intermediate Algebra|L7:60, Intermediate Algebra|L8:11, Number Theory|L1:60, Number Theory|L2:60, Number Theory|L3:60, Number Theory|L4:60, Number Theory|L5:60, Number Theory|L6:60, Number Theory|L7:60, Number Theory|L8:23, Other|L1:24, Other|L2:60, Other|L3:60, Other|L4:60, Other|L5:60, Other|L6:60, Other|L7:60, Other|L8:13, Prealgebra|L1:60, Prealgebra|L2:60, Prealgebra|L3:60, Prealgebra|L4:17, Precalculus|L1:18, Precalculus|L2:60, Precalculus|L3:60, Precalculus|L4:60, Precalculus|L5:60, Precalculus|L6:60, Precalculus|L7:27
