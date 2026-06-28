# Pilot 2,666 Validation Report
Generated: 2026-05-22T14:07:15.814319
Track A source: /scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet
Track B source: /scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts

## TL;DR
- Track A: ✅ Passed
- Track B: ✅ Passed
- Cross-Track: ✅ Passed
- **Overall**: `GO` ✅ | Track C=WAIT (사람 검토) | 40K=GO

---

## Detailed Sections


### A1. Integrity

| Check | Expected | Actual | Status |
|---|---|---|---|
| Row count | 2666 | 2666 | ✓ |
| ID set match (diff=0) | 0 | 0 | ✓ |
| pass_count range 0-8 | 0-8 | 0-8 | ✓ |
| pass_rate == pass_count/8 | all equal | check | ✓ |
| raw_responses len==8 | all 8 | check | ✓ |
| Null rows (excl. len_correct/incorrect NaN) | 0 | 0 | ✓ |
| NaN len_correct (pass=0 expected) | ~606 | 606 | ✓ |
**Status**: ✅ PASS

**Findings**: All integrity checks passed.

### A2. Pass Rate Distribution (§5.7 Hybrid input)

- pass=0 ratio: **0.227** (606/2666)
- pass=1.0 ratio: **0.312** (831/2666)
- mean pass_rate: 0.551, median: 0.625

| Bucket | Count | % |
|---|---|---|
| 0 | 606 | 22.7% |
| (0, 0.125] | 196 | 7.4% |
| (0.125, 0.25] | 162 | 6.1% |
| (0.25, 0.5] | 290 | 10.9% |
| (0.5, 0.875) | 349 | 13.1% |
| [0.875, 1.0) | 232 | 8.7% |
| 1.0 | 831 | 31.2% |

**§5.7 Auto-Decision**: `HYBRID=LAYERED` (pass=0 = 22.7%)

**Plots**: `plots/A2_pass_rate_histogram.png`
**Status**: ✅ PASS

### A3. Signal Alignment (pass_rate ↔ level)

- Spearman ρ(pass_rate, level) = **-0.5066** (p=7.845e-174)

**Pass rate by subject** (top 5 by mean):

| Subject | Mean | Std | N |
|---|---|---|---|
| Prealgebra | 0.627 | 0.395 | 200 |
| Algebra | 0.620 | 0.394 | 350 |
| Precalculus | 0.601 | 0.373 | 335 |
| Intermediate Algebra | 0.592 | 0.407 | 337 |
| Number Theory | 0.549 | 0.418 | 400 |

**Plots**: `plots/A3_pass_rate_vs_level_heatmap.png`

**Auto-Decision**: `ACCEPTABLE_A3=YES (ρ=-0.507, expected negative)`
**Status**: ✅ PASS

### A4. Response Quality

- Truncation rate: 0.051 (1082/21328 rollouts)
- Mean resp len (all): 1415.8 tokens
- Mean resp len (correct): 1216.0 tokens
- Mean resp len (incorrect): 1870.5 tokens

**Plots**: `plots/A4_response_length_correct_vs_incorrect.png`

**Spot check**: `spot_check/A4_pass_0_samples.md`, `A4_pass_full_samples.md`
**Status**: ✅ PASS

### A5. Stratification

- N units: 59
- Mean per unit: 45.2 (expected ~2666/59)
- Min/Max per unit: 5 / 50
- Actual total: 2666 (expected ≈2600 at 50/unit × 52 units)
**Status**: ✅ PASS

### A6. Resource (Track A wallclock)

- Track B sbatch log: `slurm-nait_pilot_qwen3.66378.iREMB-C-07.out`
- Start: Date:     Fri May 22 12:33:26 AM KST 2026
- End:   Date: Fri May 22 11:40:49 AM KST 2026

(Track A SLURM log not found in standard location; wallclock not extracted)
**Status**: ✅ PASS

### B1. Integrity

| Check | Expected | Actual | Status |
|---|---|---|---|
| .pt file count | 2666 | 2666 | ✓ |
| metadata row count | 2666 | 2666 | ✓ |
| metadata ID == pilot_ids | ∅ diff | 0 | ✓ |
| status==error rate | <1% | 0/2666 | ✓ |
| checkpoint union card | 2666 | 2666 | ✓ |
**Status**: ✅ PASS

**Findings**: All B1 checks passed.

### B2. Tensor Shape (random 10 .pt)

- n_layers expected: 36
- intermediate_size: 12288 (Qwen3-8B)
- dtype: bfloat16
- 10 random spot checks: all OK
**Status**: ✅ PASS

**Findings**: Tensor shape/dtype/sanity OK.

### B3. Activation Signal Sanity (streaming all 2,666)

- Samples processed: 2666
- Layer norm range: min_mean=2.747, max_mean=326.601
- Sample mean norm: mean=50.577, p5=45.851, p95=56.388
- Truncation rate (is_trunc): 0.137 (365/2666)
- Tokens generated: median=1036, p95=4096, max=4096
- <think> tags found: **0** (should be 0)

**Plots**: `plots/B3_layer_norm_boxplot.png`, `B3_sample_norm_histogram.png`
**Status**: ⚠️ WARN

### B4. Response Quality (Track B)

- is_trunc rate: 0.137
- think-tag count: **0 ✓**

**Spot check**: `spot_check/B4_generated_text_samples.md`, `B4_truncated_samples.md`
**Status**: ✅ PASS

### B5. Cross-Chunk Consistency

- Chunk sizes: [667, 667, 666, 666]
- Union cardinality: 2666 (expected 2666)
- Pairwise intersections: [0, 0, 0, 0, 0, 0] (should all be 0)
**Status**: ✅ PASS

### B6. Storage

- Directory size: 2.3G	/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts
- Avg .pt size: 875.5 KB
**Status**: ✅ PASS

### X1. ID Set Match (3-way)

- A △ ref = 0, B △ ref = 0, A △ B = 0
**Status**: ✅ PASS

### X2. Pass Rate ↔ Activation Norm (Pilot Diagnostic #1)

| Pass Bucket | Mean Norm | Std | N |
|---|---|---|---|
| 0 | 51.073 | 4.603 | 606 |
| (0,0.125] | 51.096 | 3.941 | 196 |
| (0.125,0.25] | 51.580 | 4.781 | 162 |
| (0.25,0.5] | 50.987 | 4.089 | 290 |
| (0.5,0.875) | 50.736 | 3.997 | 349 |
| [0.875,1) | 50.530 | 2.959 | 232 |
| 1.0 | 49.702 | 2.980 | 831 |

- Spearman ρ(pass_rate, mean_norm) = **-0.0910** (p=2.537e-06)
- ANOVA F=11.95, p=2.817e-13

**Plots**: `plots/X2_pass_rate_vs_activation_norm.png`

**Auto-Decision**: `X2_SIGNAL=STRONG`
**Status**: ✅ PASS

### X3. Truncation by Pass Bucket

| Bucket | Trunc Rate (Track A) |
|---|---|
| 0 | 0.322 |
| (0,0.125] | 0.352 |
| (0.125,0.25] | 0.370 |
| (0.25,0.5] | 0.269 |
| (0.5,0.875) | 0.140 |
| [0.875,1) | 0.052 |
| 1.0 | 0.001 |

**Plots**: `plots/X3_truncation_by_pass_bucket.png`
**Status**: ✅ PASS

### X4. Subject PCA (§8.A decision)

- PCA explained variance (PC1+PC2): 35.4%
- Silhouette (subject, euclidean): **-0.0154**

**Plots**: `plots/X4_subject_pca_scatter.png`

**Auto-Decision**: `SUBJECT=GPT_LABEL`
**Status**: ✅ PASS

---

## Decision Sheet

| Decision | Trigger | Result | Recommendation |
|---|---|---|---|
| §5.7 Hybrid difficulty | pass=0 ratio | HYBRID=LAYERED | **LAYERED** |
| Pilot Acceptable (A3) | ρ(pass, level) | ACCEPTABLE_A3=YES (ρ=-0.507, expected negative) | **YES (ρ=-0.507, expected negative)** |
| Cross-track signal (X2) | ρ(pass, norm) + ANOVA | X2_SIGNAL=STRONG | **STRONG** |
| §8.A Subject axis | silhouette(subject) | SUBJECT=GPT_LABEL | **GPT_LABEL** |
| Track C 진입 | B3 sanity + X2 signal | B3=FAIL, X2=X2_SIGNAL=STRONG | **WAIT (사람 검토)** |
| 40K 확장 trigger | P0 PASS + A3 + X2 STRONG | P0=OK | **GO** |

**Overall**: `GO` | FAILs: none | WARNs: ['B3']
