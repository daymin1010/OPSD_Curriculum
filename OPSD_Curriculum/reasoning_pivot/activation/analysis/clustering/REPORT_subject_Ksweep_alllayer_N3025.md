# Subject K-selection SWEEP — all-36-layer pooled THINKING ΔA (N=3025) + 29K feasibility

작성: subject_Ksweep_alllayer_pooled.py / pooled(pilot1+pilot2) THINKING ΔA, CPU, seed=42

## 0. Why this run
- Prior auto-K picked **K=2** (silhouette-best). K=2 makes subject ORDERING vacuous (③ σ-order ≡ ④ shuffle: one transition, the maximal jump). We sweep K and default to the **finest FEASIBLE K** (finer ⇒ smaller jumps, homogeneous stages, sharper ③-vs-④).
- **Hard floor = empty cell (n=0) on full 29K.** MIN_CELL=300 is a conservative diversity threshold only; 150 band also reported (OPSD oversamples small cells).
- **Recommended K = a feasibility CEILING (cell-count bound).** Budget T not fixed; with 4×K stages, per-stage step=T/(4K) shrinks → harness may finalize K BELOW this ceiling.

## 0b. Setup / assertions
- pooled N=3025 (p1=1608, p2=1417); 29K labels N=29434
- subjects (8 canonical): ['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Other', 'Prealgebra', 'Precalculus']
- representation: per-layer pooled-μ-centered ΔA; S=layer-avg cosine of 36-layer L2-normed centroids; D=1−S. difficulty FIXED {'D1': [1, 2], 'D2': [3, 4], 'D3': [5, 6], 'D4': [7, 8]}.
- consistency: recomputed pooled subject S vs saved npz → max|Δ|=0.00e+00 (atol 1e-3) → PASS
- 29K label file: `openthoughts_30k_labels_final.parquet` (problem_id, subject, level)

## 1. Subject similarity S (8×8) and distance D
```
                          Algebra Counting   Geometry Intermedi Number Th     Other Prealgebr Precalcul
                Algebra     1.000    -0.419    -0.062     0.291     0.049    -0.483     0.330     0.133
 Counting & Probability    -0.419     1.000     0.044    -0.657     0.448    -0.231     0.074    -0.728
               Geometry    -0.062     0.044     1.000    -0.196    -0.176    -0.331    -0.085     0.016
   Intermediate Algebra     0.291    -0.657    -0.196     1.000    -0.292     0.104    -0.445     0.666
          Number Theory     0.049     0.448    -0.176    -0.292     1.000    -0.514     0.126    -0.653
                  Other    -0.483    -0.231    -0.331     0.104    -0.514     1.000    -0.387     0.277
             Prealgebra     0.330     0.074    -0.085    -0.445     0.126    -0.387     1.000    -0.293
            Precalculus     0.133    -0.728     0.016     0.666    -0.653     0.277    -0.293     1.000
```
```
                          Algebra Counting   Geometry Intermedi Number Th     Other Prealgebr Precalcul
                Algebra     0.000     1.419     1.062     0.709     0.951     1.483     0.670     0.867
 Counting & Probability     1.419     0.000     0.956     1.657     0.552     1.231     0.926     1.728
               Geometry     1.062     0.956     0.000     1.196     1.176     1.331     1.085     0.984
   Intermediate Algebra     0.709     1.657     1.196     0.000     1.292     0.896     1.445     0.334
          Number Theory     0.951     0.552     1.176     1.292     0.000     1.514     0.874     1.653
                  Other     1.483     1.231     1.331     0.896     1.514     0.000     1.387     0.723
             Prealgebra     0.670     0.926     1.085     1.445     0.874     1.387     0.000     1.293
            Precalculus     0.867     1.728     0.984     0.334     1.653     0.723     1.293     0.000
```

## 1b. 29K subject × level (reference; binding constraint preview)
```
level                     1     2     3     4     5     6    7   8
subject                                                           
Algebra                 165  1190  1805  1724  1365   495   79   0
Counting & Probability  106   561   961   713   720   480  308  10
Geometry                 37   356   797  1317  1389   870  491   9
Intermediate Algebra     16    85   231   464   725   517  240  11
Number Theory            89   445   909  1101  1271  1033  565  23
Other                    24    98   124    81   122    92  109  13
Prealgebra              888  1460   538    17     0     0    0   0
Precalculus              18   133   549   755   518   195   27   0
```
- D4{7,8} per-subject totals: Algebra=79, Counting & Probability=318, Geometry=500, Intermediate Algebra=251, Number Theory=588, Other=122, Prealgebra=0, Precalculus=27
  → **Prealgebra D4 = 0** (empty): any K that isolates Prealgebra in D4 hits the hard floor.

## 2. K-sweep (K=2..8 + individual-subject limit)
PRIMARY = average linkage. headroom_mean = mean_offdiag(Dc) − σ_cost/(K−1) (③ beats ④ per transition; **=0 at K=2 by construction**, grows with K).

### Decision table (PRIMARY average linkage)
| K | n_stages | within_spread | σ_mean | σ_max | rand_mean | **headroom_mean** | silhouette | min_29K | #empty | #<300 | #<150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 0.408 | 1.956 | 1.956 | 1.956 | **0.000** | +0.342 | 374 | 0 | 0 | 0 |
| 3 | 12 | 0.238 | 1.191 | 1.269 | 1.418 | **0.226** | +0.238 | 374 | 0 | 0 | 0 |
| 4 | 16 | 0.160 | 1.105 | 1.269 | 1.277 | **0.172** | +0.338 | 79 | 0 | 1 | 1 |
| 5 | 20 | 0.092 | 0.981 | 1.093 | 1.229 | **0.248** | +0.299 | 79 | 0 | 7 | 3 |
| 6 | 24 | 0.040 | 0.835 | 1.064 | 1.164 | **0.329** | +0.230 | 0 | 2 | 9 | 5 |
| 7 | 28 | 0.013 | 0.769 | 0.956 | 1.141 | **0.372** | +0.134 | 0 | 2 | 9 | 5 |
| 8 | 32 | 0.000 | 0.688 | 0.956 | 1.121 | **0.433** | +nan | 0 | 2 | 11 | 7 |
| indiv(8) | 32 | 0.000 | 0.688 | 0.956 | 1.121 | **0.433** | n/a | 0 | 2 | 11 | 7 |

### Memberships per K (average linkage)
- K=2: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}
- K=3: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Number Theory, Prealgebra}; C3{Geometry}
- K=4: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- K=5: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra, Prealgebra}; C5{Geometry}
- K=6: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra}; C5{Prealgebra}; C6{Geometry}
- K=7: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability}; C4{Number Theory}; C5{Algebra}; C6{Prealgebra}; C7{Geometry}
- K=8: C1{Algebra}; C2{Counting & Probability}; C3{Geometry}; C4{Intermediate Algebra}; C5{Number Theory}; C6{Other}; C7{Prealgebra}; C8{Precalculus}
- indiv(8): each subject singleton; σ path = Geometry → Counting & Probability → Number Theory → Prealgebra → Algebra → Intermediate Algebra → Precalculus → Other (cost 4.818)

### Complete-linkage memberships (robustness)
- K=2: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}
- K=3: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Geometry, Prealgebra}
- K=4: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- K=5: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra, Prealgebra}; C5{Geometry}
- K=6: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra}; C5{Prealgebra}; C6{Geometry}
- K=7: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability}; C4{Number Theory}; C5{Algebra}; C6{Prealgebra}; C7{Geometry}
- K=8: C1{Algebra}; C2{Counting & Probability}; C3{Geometry}; C4{Intermediate Algebra}; C5{Number Theory}; C6{Other}; C7{Prealgebra}; C8{Precalculus}

## 3. Recommendation
- feasible K (no empty 29K cell): [2, 3, 4, 5]
- **feasibility CEILING K = 5** (finest K with zero empty cells). This is a cell-count ceiling, NOT the final K — budget T may pull K lower.
- **recommended K = 5** (finest feasible (empty=0) → ceiling K=5); human-confirmable, `--K` to override.
- note: at K=5, #cells<300=7, #cells<150=3 (diversity eyeball; OPSD oversamples small cells).

## 4. 29K feasibility — 4×K cell counts per K

### K=2  (min=374, empty=0, <300=0, <150=0)
| difficulty | C1 | C2 |
|---|---|---|
| D1[1, 2] | 374 | 5297 |
| D2[3, 4] | 2204 | 9882 |
| D3[5, 6] | 2169 | 7623 |
| D4[7, 8] | 400 | 1485 |
- clusters: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}

### K=3  (min=374, empty=0, <300=0, <150=0)
| difficulty | C1 | C2 | C3 |
|---|---|---|---|
| D1[1, 2] | 374 | 4904 | 393 |
| D2[3, 4] | 2204 | 7768 | 2114 |
| D3[5, 6] | 2169 | 5364 | 2259 |
| D4[7, 8] | 400 | 985 | 500 |
- clusters: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Number Theory, Prealgebra}; C3{Geometry}

### K=4  (min=79, empty=0, <300=1, <150=1)
| difficulty | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| D1[1, 2] | 374 | 1201 | 3703 | 393 |
| D2[3, 4] | 2204 | 3684 | 4084 | 2114 |
| D3[5, 6] | 2169 | 3504 | 1860 | 2259 |
| D4[7, 8] | 400 | 906 | 79 | 500 |
- clusters: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}

### K=5  (min=79, empty=0, <300=7, <150=3)
| difficulty | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| D1[1, 2] | 252 | 122 | 1201 | 3703 | 393 |
| D2[3, 4] | 1999 | 205 | 3684 | 4084 | 2114 |
| D3[5, 6] | 1955 | 214 | 3504 | 1860 | 2259 |
| D4[7, 8] | 278 | 122 | 906 | 79 | 500 |
- clusters: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra, Prealgebra}; C5{Geometry}

### K=6  (min=0, empty=2, <300=9, <150=5)
| difficulty | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| D1[1, 2] | 252 | 122 | 1201 | 1355 | 2348 | 393 |
| D2[3, 4] | 1999 | 205 | 3684 | 3529 | 555 | 2114 |
| D3[5, 6] | 1955 | 214 | 3504 | 1860 | 0 | 2259 |
| D4[7, 8] | 278 | 122 | 906 | 79 | 0 | 500 |
- clusters: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra}; C5{Prealgebra}; C6{Geometry}
- EMPTY cells: D3|C5{Prealgebra}, D4|C5{Prealgebra}

### K=7  (min=0, empty=2, <300=9, <150=5)
| difficulty | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| D1[1, 2] | 252 | 122 | 667 | 534 | 1355 | 2348 | 393 |
| D2[3, 4] | 1999 | 205 | 1674 | 2010 | 3529 | 555 | 2114 |
| D3[5, 6] | 1955 | 214 | 1200 | 2304 | 1860 | 0 | 2259 |
| D4[7, 8] | 278 | 122 | 318 | 588 | 79 | 0 | 500 |
- clusters: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability}; C4{Number Theory}; C5{Algebra}; C6{Prealgebra}; C7{Geometry}
- EMPTY cells: D3|C6{Prealgebra}, D4|C6{Prealgebra}

### K=8  (min=0, empty=2, <300=11, <150=7)
| difficulty | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| D1[1, 2] | 1355 | 667 | 393 | 101 | 534 | 122 | 2348 | 151 |
| D2[3, 4] | 3529 | 1674 | 2114 | 695 | 2010 | 205 | 555 | 1304 |
| D3[5, 6] | 1860 | 1200 | 2259 | 1242 | 2304 | 214 | 0 | 713 |
| D4[7, 8] | 79 | 318 | 500 | 251 | 588 | 122 | 0 | 27 |
- clusters: C1{Algebra}; C2{Counting & Probability}; C3{Geometry}; C4{Intermediate Algebra}; C5{Number Theory}; C6{Other}; C7{Prealgebra}; C8{Precalculus}
- EMPTY cells: D3|C7{Prealgebra}, D4|C7{Prealgebra}

## 5. Geometry handling at recommended K (report both; human decides)
Geometry is ~orthogonal to all subjects (nearest distance ≈0.95), forcing the largest σ-transition at any K; its path position is weakly determined.

### (a) Geometry as its own singleton cluster
- clusters: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory}; C4{Algebra, Prealgebra}; C6{Geometry}
- σ order: C2 → C1 → C6 → C3 → C4  (σ_mean=0.981, σ_max=1.093, headroom_mean=0.248)
- 29K feasibility: min=79, empty=0, <300=7, <150=3

### (b) Geometry merged into nearest cluster C3 (dist=1.064)
- clusters: C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Counting & Probability, Number Theory, Geometry}; C4{Algebra, Prealgebra}
- σ order: C2 → C1 → C4 → C3  (σ_mean=1.029, σ_max=1.261, headroom_mean=0.285)
- 29K feasibility: min=79, empty=0, <300=7, <150=3

## 6. Stage layout artifacts (snake / σ-reversal; generalizes to any K)
Rule: difficulty monotone; within each difficulty traverse clusters in σ order, reversing σ at each difficulty transition (subject cluster held constant across every difficulty boundary). Total stages = 4×K. arm⑤ random parts per difficulty = K.

### K=5 → `stages_arm3_K5.json`  (σ = C2 → C1 → C5 → C3 → C4)
- inter-cluster cosine S:
```
                  C1        C2        C3        C4        C5
        C1     1.000     0.214    -0.774    -0.261    -0.093
        C2     0.214     1.000    -0.417    -0.486    -0.331
        C3    -0.774    -0.417     1.000     0.017    -0.064
        C4    -0.261    -0.486     0.017     1.000    -0.095
        C5    -0.093    -0.331    -0.064    -0.095     1.000
```
| stage | difficulty | levels | subject_cluster | subject_members |
|---|---|---|---|---|
| 0 | D1 | [1, 2] | C2 | Other |
| 1 | D1 | [1, 2] | C1 | Intermediate Algebra, Precalculus |
| 2 | D1 | [1, 2] | C5 | Geometry |
| 3 | D1 | [1, 2] | C3 | Counting & Probability, Number Theory |
| 4 | D1 | [1, 2] | C4 | Algebra, Prealgebra |
| 5 | D2 | [3, 4] | C4 | Algebra, Prealgebra |
| 6 | D2 | [3, 4] | C3 | Counting & Probability, Number Theory |
| 7 | D2 | [3, 4] | C5 | Geometry |
| 8 | D2 | [3, 4] | C1 | Intermediate Algebra, Precalculus |
| 9 | D2 | [3, 4] | C2 | Other |
| 10 | D3 | [5, 6] | C2 | Other |
| 11 | D3 | [5, 6] | C1 | Intermediate Algebra, Precalculus |
| 12 | D3 | [5, 6] | C5 | Geometry |
| 13 | D3 | [5, 6] | C3 | Counting & Probability, Number Theory |
| 14 | D3 | [5, 6] | C4 | Algebra, Prealgebra |
| 15 | D4 | [7, 8] | C4 | Algebra, Prealgebra |
| 16 | D4 | [7, 8] | C3 | Counting & Probability, Number Theory |
| 17 | D4 | [7, 8] | C5 | Geometry |
| 18 | D4 | [7, 8] | C1 | Intermediate Algebra, Precalculus |
| 19 | D4 | [7, 8] | C2 | Other |
- difficulty-boundary subject-continuity: OK

### K=4 → `stages_arm3_K4.json`  (σ = C1 → C4 → C2 → C3)
- inter-cluster cosine S:
```
                  C1        C2        C3        C4
        C1     1.000    -0.774    -0.476    -0.269
        C2    -0.774     1.000     0.017    -0.064
        C3    -0.476     0.017     1.000    -0.095
        C4    -0.269    -0.064    -0.095     1.000
```
| stage | difficulty | levels | subject_cluster | subject_members |
|---|---|---|---|---|
| 0 | D1 | [1, 2] | C1 | Intermediate Algebra, Other, Precalculus |
| 1 | D1 | [1, 2] | C4 | Geometry |
| 2 | D1 | [1, 2] | C2 | Counting & Probability, Number Theory |
| 3 | D1 | [1, 2] | C3 | Algebra, Prealgebra |
| 4 | D2 | [3, 4] | C3 | Algebra, Prealgebra |
| 5 | D2 | [3, 4] | C2 | Counting & Probability, Number Theory |
| 6 | D2 | [3, 4] | C4 | Geometry |
| 7 | D2 | [3, 4] | C1 | Intermediate Algebra, Other, Precalculus |
| 8 | D3 | [5, 6] | C1 | Intermediate Algebra, Other, Precalculus |
| 9 | D3 | [5, 6] | C4 | Geometry |
| 10 | D3 | [5, 6] | C2 | Counting & Probability, Number Theory |
| 11 | D3 | [5, 6] | C3 | Algebra, Prealgebra |
| 12 | D4 | [7, 8] | C3 | Algebra, Prealgebra |
| 13 | D4 | [7, 8] | C2 | Counting & Probability, Number Theory |
| 14 | D4 | [7, 8] | C4 | Geometry |
| 15 | D4 | [7, 8] | C1 | Intermediate Algebra, Other, Precalculus |
- difficulty-boundary subject-continuity: OK

## 7. Robustness (report, NOT a gate) — recommended K and K=4
Expectation: S-matrix correlation stays HIGH (continuous geometry reliable) even where hard-cut membership wobbles (Geometry/Algebra/Prealgebra). Membership wobble is NOT a failure — it is the argument for finer/continuous placement.

### K=4
- pilot1: S_corr_vs_pooled=+0.972, membership_same=YES → C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Prealgebra}; C3{Counting & Probability, Number Theory}; C4{Geometry}
- pilot2: S_corr_vs_pooled=+0.973, membership_same=YES → C1{Algebra, Prealgebra}; C2{Counting & Probability, Number Theory}; C3{Intermediate Algebra, Other, Precalculus}; C4{Geometry}
- balanced, N=1200: S_corr_vs_pooled=+0.927, membership_same=NO → C1{Counting & Probability, Number Theory, Prealgebra}; C2{Algebra, Intermediate Algebra, Precalculus}; C3{Geometry}; C4{Other}

### K=5
- pilot1: S_corr_vs_pooled=+0.972, membership_same=YES → C1{Intermediate Algebra, Precalculus}; C2{Other}; C3{Algebra, Prealgebra}; C4{Counting & Probability, Number Theory}; C5{Geometry}
- pilot2: S_corr_vs_pooled=+0.973, membership_same=YES → C1{Algebra, Prealgebra}; C2{Counting & Probability, Number Theory}; C3{Intermediate Algebra, Precalculus}; C4{Other}; C5{Geometry}
- balanced, N=1200: S_corr_vs_pooled=+0.936, membership_same=NO → C1{Intermediate Algebra, Precalculus}; C2{Algebra}; C3{Other}; C4{Counting & Probability, Number Theory, Prealgebra}; C5{Geometry}

## 8. All-layer vs mid-L11-15 (view-choice documentation; carried over)
- ALL-36-layer average adopted as reviewer-defensible default (no principled criterion to single out mid layers). Invalidates old G1/G2/Other grouping. Grouping derived solely from all-36-layer S; no cross-view membership-agreement claim.