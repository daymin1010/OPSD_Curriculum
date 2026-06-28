# Phase-0 — OPSD↔labels join + Set-A coverage (REPORT_join_setA)

작성: curriculum_schedule.py run_phase0 (CPU, deterministic)
- OPSD dataset: `siyanzhao/Openthoughts_math_30k_opsd` (train split)
- OPSD columns: `['source', 'problem', 'solution', 'messages', 'system', 'conversations', 'generated_token_count', 'correct', 'Question', 'COT_Reason', 'Answer']`
- labels parquet: `openthoughts_30k_labels_final.parquet` (29434 rows)

## 0. Join key auto-detect
- **key used: `sha1(problem)[:16] == labels.problem_id`**
- **match rate = 1.0000** (29434/29434)
- coverage gate (>= 0.95): **PASS**
- candidate keys:
    - sha1_problem_id: matched=29434 rate=1.0000

## 1. Universe accounting
- OPSD train rows           : 29434
- unmatched (no label)      : 0
- matched but subject=Other : 663
- **Set-A trainable total   : 28771**
- labels-parquet Set-A ref  : ~28,771 (computed on parquet alone)
- gap vs 28,771             : 0  (OK / small)

## 2. Main (③-A) per-cell counts — difficulty × subject_cluster (Set-A)

| difficulty | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| D1[1, 2] | 252 | 1201 | 3703 | 393 |
| D2[3, 4] | 1999 | 3684 | 4084 | 2114 |
| D3[5, 6] | 1955 | 3504 | 1860 | 2259 |
| D4[7, 8] | 278 | 906 | 79 | 500 |

- clusters: C1{Intermediate Algebra, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- min cell = 79; #empty = 0

## 3. Diff-only (②-A) per-stage counts (Set-A)

| stage | difficulty | n |
|---|---|---|
| 0 | D1[1, 2] | 5549 |
| 1 | D2[3, 4] | 11881 |
| 2 | D3[5, 6] | 9578 |
| 3 | D4[7, 8] | 1763 |

## 4. Outputs
- per-row table: `outputs/join_setA_rows.parquet` (29434 rows)
- diff-only stages JSON: `stages/stages_diffonly_setA.json`
- main stages JSON (local copy): `stages/stages_arm3_excludeOther.json`
