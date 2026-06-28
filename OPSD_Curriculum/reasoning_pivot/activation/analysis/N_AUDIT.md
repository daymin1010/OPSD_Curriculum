# Phase 0 — N / L8 Audit (pooled pilot1+pilot2)

CPU-only. Reuses `similarity_analysis.load_pilot` (identical arrays to the analysis). Non-finite = any NaN/Inf in DAF or DAT.

## Q1 — N reconciliation (raw / finite)

| | .pt loaded | non-finite (DAF) | non-finite (DAT) | non-finite (either) | **finite N** |
|---|---|---|---|---|---|
| pilot1 | 1608 | 0 | 0 | 0 | **1608** |
| pilot2 | 1417 | 0 | 0 | 0 | **1417** |
| **pooled** | 3025 | 0 | 0 | 0 | **3025** |

**pilot1 file mtime span:** {'n': 1608, 'earliest': '2026-06-03 00:57', 'latest': '2026-06-04 12:12', 'span_hours': 35.25}

**1541 vs 1608 call:** pilot1 non-finite(either) = **0**, loaded−1541 = **67**.
→ **(b) population-growth hypothesis likely**: almost all 1608 files are finite, so 1541 was an earlier snapshot; current finite N supersedes it. Check mtime span above for a late file cluster.

## Q2 — L8 availability (drives conditional L8 extraction)

- L8 total (pooled finite): **66**  (pilot1=66, pilot2=0)
- usability threshold = 20
- **DECISION: L8 sufficient — proceed straight to pooled analysis**

L8 by subject:
  - Number Theory: 23
  - Other: 13
  - Intermediate Algebra: 11
  - Counting & Probability: 10
  - Geometry: 9

## Q3 — pooled level distribution + subject×level

level counts (pooled finite):

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| n | 335 | 480 | 480 | 437 | 420 | 420 | 387 | 66 |

subject × level crosstab (pooled finite):

```
level                    1   2   3   4   5   6   7   8
subject                                               
Algebra                 60  60  60  60  60  60  60   0
Counting & Probability  60  60  60  60  60  60  60  10
Geometry                37  60  60  60  60  60  60   9
Intermediate Algebra    16  60  60  60  60  60  60  11
Number Theory           60  60  60  60  60  60  60  23
Other                   24  60  60  60  60  60  60  13
Prealgebra              60  60  60  17   0   0   0   0
Precalculus             18  60  60  60  60  60  27   0
```

## Canonical N convention

- raw .pt = 3025  /  finite pooled N = **3025**.
- Always report raw / finite / analysis-N. '3000' is a nickname only.
