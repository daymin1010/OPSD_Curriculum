# Subject Grouping — EXCLUDE 'Other' (arm ③-main, FINAL MAIN), all-36-layer pooled THINKING ΔA (N=3025)

작성: subject_grouping_excludeOther_pooled.py / pooled(pilot1+pilot2), THINKING ΔA, CPU, seed=42

## 0. Setup / assertions
- pooled N = **3025** (pilot1=1608, pilot2=1417)
- 7 subjects (drop 'Other'): ['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Prealgebra', 'Precalculus']
- representation: per-layer pooled-μ-centered ΔA; S=layer-avg cosine of 36-layer L2-normed centroids; D=1−S.
- difficulty axis (FIXED): {'D1': [1, 2], 'D2': [3, 4], 'D3': [5, 6], 'D4': [7, 8]}
- 8×8 pooled S vs saved npz: max|Δ|=0.00e+00 (atol 1e-3) → PASS
- **7×7 S == 8×8 sub-block** (slice vs recompute): max|Δ|=0.00e+00 (atol 1e-6) → PASS

## 1. 7×7 subject similarity S (exclude Other)
```
                          Algebra Counting   Geometry Intermedi Number Th Prealgebr Precalcul
                Algebra     1.000    -0.419    -0.062     0.291     0.049     0.330     0.133
 Counting & Probability    -0.419     1.000     0.044    -0.657     0.448     0.074    -0.728
               Geometry    -0.062     0.044     1.000    -0.196    -0.176    -0.085     0.016
   Intermediate Algebra     0.291    -0.657    -0.196     1.000    -0.292    -0.445     0.666
          Number Theory     0.049     0.448    -0.176    -0.292     1.000     0.126    -0.653
             Prealgebra     0.330     0.074    -0.085    -0.445     0.126     1.000    -0.293
            Precalculus     0.133    -0.728     0.016     0.666    -0.653    -0.293     1.000
```

### Distance D = 1 − S
```
                          Algebra Counting   Geometry Intermedi Number Th Prealgebr Precalcul
                Algebra     0.000     1.419     1.062     0.709     0.951     0.670     0.867
 Counting & Probability     1.419     0.000     0.956     1.657     0.552     0.926     1.728
               Geometry     1.062     0.956     0.000     1.196     1.176     1.085     0.984
   Intermediate Algebra     0.709     1.657     1.196     0.000     1.292     1.445     0.334
          Number Theory     0.951     0.552     1.176     1.292     0.000     0.874     1.653
             Prealgebra     0.670     0.926     1.085     1.445     0.874     0.000     1.293
            Precalculus     0.867     1.728     0.984     0.334     1.653     1.293     0.000
```

## 2. Clustering — K=4 (average linkage primary; complete robustness)
- average cophenetic corr = +0.720; complete cophenetic corr = +0.697
- **K=4 partition (average linkage):** C1{Intermediate Algebra, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- K=4 partition (complete linkage, robustness): C1{Intermediate Algebra, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- **target partition match (average): YES** (target = {IntAlg,Precalc},{C&P,NumberTheory},{Algebra,Prealgebra},{Geometry})
- rationale: K=5 would split the +0.666 {IntAlg,Precalc} pair (wrong); K=3 remakes an incoherent bucket. K=4 is the principled granularity for 7 subjects.

## 3. σ ordering + smoothness (4 cluster centroids)
- inter-cluster centroid cosine S (4×4):
```
                  C1        C2        C3        C4
        C1     1.000    -0.774    -0.261    -0.093
        C2    -0.774     1.000     0.017    -0.064
        C3    -0.261     0.017     1.000    -0.095
        C4    -0.093    -0.064    -0.095     1.000
```
- **σ (optimal open path, total cost=3.139)** = C1{Intermediate Algebra, Precalculus} → C4{Geometry} → C2{Counting & Probability, Number Theory} → C3{Algebra, Prealgebra}
- mean per-transition distance = 1.046; max per-transition = 1.093
- random-order mean (mean off-diag Dc) = 1.212
- **headroom_mean = rand_mean − σ_mean = 0.165** (modest, dragged by Geometry's orthogonality — the honest smoothness for the real subjects)
- next 3 near-optimal σ paths (Geometry ~orthogonal → its σ position floats):
    cost=3.170: C1 → C4 → C3 → C2
    cost=3.307: C1 → C3 → C2 → C4
    cost=3.336: C2 → C3 → C1 → C4

## 4. Stage layout (4 difficulty × 4 cluster = 16 stages, snake / σ-reversal)
- subject clusters in σ order: C1 → C4 → C2 → C3
- rule: difficulty monotone; within each difficulty traverse clusters in σ order, reversing σ at each difficulty transition (subject cluster held constant across every D-boundary).

| stage | difficulty | levels | subject_cluster | subject_members |
|---|---|---|---|---|
| 0 | D1 | [1, 2] | C1 | Intermediate Algebra, Precalculus |
| 1 | D1 | [1, 2] | C4 | Geometry |
| 2 | D1 | [1, 2] | C2 | Counting & Probability, Number Theory |
| 3 | D1 | [1, 2] | C3 | Algebra, Prealgebra |
| 4 | D2 | [3, 4] | C3 | Algebra, Prealgebra |
| 5 | D2 | [3, 4] | C2 | Counting & Probability, Number Theory |
| 6 | D2 | [3, 4] | C4 | Geometry |
| 7 | D2 | [3, 4] | C1 | Intermediate Algebra, Precalculus |
| 8 | D3 | [5, 6] | C1 | Intermediate Algebra, Precalculus |
| 9 | D3 | [5, 6] | C4 | Geometry |
| 10 | D3 | [5, 6] | C2 | Counting & Probability, Number Theory |
| 11 | D3 | [5, 6] | C3 | Algebra, Prealgebra |
| 12 | D4 | [7, 8] | C3 | Algebra, Prealgebra |
| 13 | D4 | [7, 8] | C2 | Counting & Probability, Number Theory |
| 14 | D4 | [7, 8] | C4 | Geometry |
| 15 | D4 | [7, 8] | C1 | Intermediate Algebra, Precalculus |

- difficulty-boundary subject-continuity (snake): OK
- harness: arm ④ uses these SAME 16 cells (shuffle subject-cluster visiting order); arm ⑤ = 4 subject-agnostic random parts per difficulty.

## 5. Feasibility on the non-Other rows (Set-A)
- 29K total = 29434; N(Other) = 663; **Set-A total (non-Other) = 28771**
- 4×4 cell counts: min_cell=79, #empty=0, #<300=3, #<150=1

| difficulty | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| D1[1, 2] | 252 | 1201 | 3703 | 393 |
| D2[3, 4] | 1999 | 3684 | 4084 | 2114 |
| D3[5, 6] | 1955 | 3504 | 1860 | 2259 |
| D4[7, 8] | 278 | 906 | 79 | 500 |
- clusters: C1{Intermediate Algebra, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}

- **CONFOUND FLAG (documented, not a gate):** cluster C3{Algebra,Prealgebra} is Prealgebra-dominated at D1 and Algebra-ONLY at D4 (Prealgebra empty above L4). Cluster composition shifts across difficulty: D1=3703, D2=4084, D3=1860, D4=79.

## 6. Robustness (report, NOT a gate)
### pilot1-only (re-centered, N=1608)
- C1{Intermediate Algebra, Precalculus}; C2{Algebra, Prealgebra}; C3{Counting & Probability, Number Theory}; C4{Geometry}
- S-corr vs pooled = +0.981; target partition match: YES
### pilot2-only (re-centered, N=1417)
- C1{Algebra, Prealgebra}; C2{Counting & Probability, Number Theory}; C3{Intermediate Algebra, Precalculus}; C4{Geometry}
- S-corr vs pooled = +0.977; target partition match: YES

### gen_len-quintile-balanced subsample (re-centered)
- balanced N=1260
- C1{Counting & Probability, Number Theory, Prealgebra}; C2{Intermediate Algebra, Precalculus}; C3{Algebra}; C4{Geometry}
- S-corr vs pooled = +0.915; target partition match: NO
- note: balanced subsample may split {Algebra,Prealgebra} via Prealgebra's length/difficulty confound — documented, not a gate.
- dendrogram [average]: dendro_subjgroup_excludeOther_average.png
- dendrogram [complete]: dendro_subjgroup_excludeOther_complete.png
- reordered 7×7 S heatmap: heatmap_subjS7_reordered_excludeOther.png (leaf order: ['Intermediate Algebra', 'Precalculus', 'Geometry', 'Counting & Probability', 'Number Theory', 'Algebra', 'Prealgebra'])