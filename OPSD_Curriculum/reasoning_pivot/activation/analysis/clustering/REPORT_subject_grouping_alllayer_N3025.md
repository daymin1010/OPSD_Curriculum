# Subject Grouping — ALL-36-layer pooled THINKING ΔA (N=3025)

작성: subject_grouping_alllayer_pooled.py / pooled(pilot1+pilot2), THINKING ΔA, CPU, seed=42

## 0. Setup / assertions
- pooled N = **3025** (pilot1=1608, pilot2=1417)
- subjects (8 canonical, exact strings): ['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Other', 'Prealgebra', 'Precalculus']
- representation: per-layer pooled-μ-centered ΔA; S[g,h]=layer-avg cosine of 36-layer L2-normed centroids; D=1−S.
- difficulty axis (FIXED, unchanged): {'D1': [1, 2], 'D2': [3, 4], 'D3': [5, 6], 'D4': [7, 8]}
- **consistency check**: recomputed pooled subject S vs saved `sim_matrices_pooled3025_levsubj.npz` → max|Δ|=0.00e+00 (atol 1e-3) → PASS

## 1. Subject similarity S (8×8, all-36-layer, centered)
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

## 1b. Distance D = 1 − S
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

## 2. Clustering — K decided from data (NOT fixed to 3)
PRIMARY = average linkage on precomputed D; ROBUSTNESS = complete linkage. Ward/centroid/median NOT used on D (valid only for Euclidean).

### [average] linkage
- cophenetic correlation = +0.761
- merge heights (ascending) = [0.334, 0.552, 0.67, 0.809, 1.043, 1.07, 1.318]
- merge gaps = [0.218, 0.118, 0.139, 0.233, 0.027, 0.248]
- largest-gap recommendation: K=2 (gap=0.248)

| K | silhouette(precomputed) | clusters |
|---|---|---|
| 2 | +0.342 | {Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}; {Intermediate Algebra, Other, Precalculus} |
| 3 | +0.238 | {Algebra, Counting & Probability, Number Theory, Prealgebra}; {Geometry}; {Intermediate Algebra, Other, Precalculus} |
| 4 | +0.338 | {Algebra, Prealgebra}; {Counting & Probability, Number Theory}; {Geometry}; {Intermediate Algebra, Other, Precalculus} |
| 5 | +0.299 | {Algebra, Prealgebra}; {Counting & Probability, Number Theory}; {Geometry}; {Intermediate Algebra, Precalculus}; {Other} |
| 6 | +0.230 | {Algebra}; {Counting & Probability, Number Theory}; {Geometry}; {Intermediate Algebra, Precalculus}; {Other}; {Prealgebra} |

### [complete] linkage
- cophenetic correlation = +0.742
- merge heights (ascending) = [0.334, 0.552, 0.67, 0.896, 1.085, 1.419, 1.728]
- merge gaps = [0.218, 0.118, 0.226, 0.189, 0.334, 0.309]
- largest-gap recommendation: K=3 (gap=0.334)

| K | silhouette(precomputed) | clusters |
|---|---|---|
| 2 | +0.342 | {Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}; {Intermediate Algebra, Other, Precalculus} |
| 3 | +0.309 | {Algebra, Geometry, Prealgebra}; {Counting & Probability, Number Theory}; {Intermediate Algebra, Other, Precalculus} |
| 4 | +0.338 | {Algebra, Prealgebra}; {Counting & Probability, Number Theory}; {Geometry}; {Intermediate Algebra, Other, Precalculus} |
| 5 | +0.299 | {Algebra, Prealgebra}; {Counting & Probability, Number Theory}; {Geometry}; {Intermediate Algebra, Precalculus}; {Other} |
| 6 | +0.230 | {Algebra}; {Counting & Probability, Number Theory}; {Geometry}; {Intermediate Algebra, Precalculus}; {Other}; {Prealgebra} |

### Recommended K (PRIMARY=average linkage)
- silhouette peak at K=2 (sil=+0.342)
- largest merge-gap implies K=2
- **agree → RECOMMEND K=2**

#### side-by-side memberships (k=2,3,4)
- k=2: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}
- k=3: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Number Theory, Prealgebra}; C3{Geometry}
- k=4: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- dendrogram [average]: dendro_subjgroup_average_alllayer.png
- dendrogram [complete]: dendro_subjgroup_complete_alllayer.png

### [robustness, NON-PRIMARY] Ward on Euclidean of flattened per-layer-L2-normed centroids
Ward minimizes Euclidean variance; this reintroduces high-norm-layer dominance (exactly what layer-averaged cosine avoids). Expected to differ from the primary cosine clustering — shown only for transparency.
- ward cophenetic corr (Euclidean) = +0.758
- ward k=2: C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}
- ward k=3: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Geometry, Prealgebra}
- ward k=4: C1{Intermediate Algebra, Other, Precalculus}; C2{Counting & Probability, Number Theory}; C3{Algebra, Prealgebra}; C4{Geometry}
- reordered S heatmap: heatmap_subjS_reordered_alllayer.png (leaf order: ['Other', 'Intermediate Algebra', 'Precalculus', 'Geometry', 'Counting & Probability', 'Number Theory', 'Algebra', 'Prealgebra'])

## 3. Continuous subject ordering — exact open Hamiltonian path (min Σ consecutive D, 8!/2=20160)
- **optimal path** (cost=4.818): Geometry → Counting & Probability → Number Theory → Prealgebra → Algebra → Intermediate Algebra → Precalculus → Other
- near-optimal:
    cost=5.029: Number Theory → Counting & Probability → Geometry → Prealgebra → Algebra → Intermediate Algebra → Precalculus → Other
    cost=5.089: Geometry → Number Theory → Counting & Probability → Prealgebra → Algebra → Intermediate Algebra → Precalculus → Other
    cost=5.149: Geometry → Counting & Probability → Number Theory → Prealgebra → Algebra → Precalculus → Intermediate Algebra → Other
    cost=5.193: Counting & Probability → Number Theory → Prealgebra → Algebra → Intermediate Algebra → Precalculus → Other → Geometry

### Contiguity check (recommended K=2 clusters vs optimal path)
- path cluster runs: C2×5 | C1×3
- **clusters contiguous along optimal path? YES** (clustering & ordering agree)

## 4. σ ordering of the recommended-K clusters
- inter-cluster centroid cosine (analog of old G1-G2):
```
                  C1        C2
        C1     1.000    -0.956
        C2    -0.956     1.000
```
- **σ (optimal cluster path, cost=1.956)** = C1 → C2

## 5. Stage layout (difficulty × subject, snake / σ-reversal)
- difficulty axis (fixed): D1{1,2} D2{3,4} D3{5,6} D4{7,8}
- subject clusters in σ order: C1 → C2
- layout rule: difficulty advances monotonically; within each difficulty traverse clusters in σ order, REVERSING σ at each difficulty transition (subject cluster held constant across every difficulty boundary). Total stages = 4 × 2 = 8.

| stage | difficulty | levels | subject_cluster | subject_members |
|---|---|---|---|---|
| 0 | D1 | [1, 2] | C1 | Intermediate Algebra, Other, Precalculus |
| 1 | D1 | [1, 2] | C2 | Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra |
| 2 | D2 | [3, 4] | C2 | Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra |
| 3 | D2 | [3, 4] | C1 | Intermediate Algebra, Other, Precalculus |
| 4 | D3 | [5, 6] | C1 | Intermediate Algebra, Other, Precalculus |
| 5 | D3 | [5, 6] | C2 | Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra |
| 6 | D4 | [7, 8] | C2 | Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra |
| 7 | D4 | [7, 8] | C1 | Intermediate Algebra, Other, Precalculus |

- difficulty-boundary subject-continuity (snake): OK (subject cluster identical across every D-boundary)

### Downstream harness changes
- arm ④ (subject-order-shuffled): uses the SAME 4×2=8 cells; only the σ visiting order of subject clusters is shuffled within each difficulty.
- arm ⑤ (subject-agnostic random split): must split each difficulty into **K=2** random parts (NOT 3) to match the new granularity.

## 6. Robustness (report, NOT a gate)
### pilot1-only (re-centered, N=1608)
- C1{Intermediate Algebra, Other, Precalculus}; C2{Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra}
- S matrix correlation vs pooled = +0.972
### pilot2-only (re-centered, N=1417)
- C1{Algebra, Counting & Probability, Number Theory, Prealgebra}; C2{Geometry, Intermediate Algebra, Other, Precalculus}
- S matrix correlation vs pooled = +0.973
- recommended-K membership stable: pilot1=YES, pilot2=NO

### gen_len-quintile-balanced subsample
- balanced N=1200
- C1{Counting & Probability, Number Theory, Prealgebra}; C2{Algebra, Geometry, Intermediate Algebra, Other, Precalculus}
- S matrix correlation vs pooled = +0.927
- recommended-K membership survives length balancing: NO
- note: prior subject separability gap retained only ~57% under balancing → grouping length-robustness explicitly checked here.

## 7. All-layer vs mid-L11-15 (documentation of view choice; NOT a cross-view agreement claim)
- We adopt the ALL-36-layer average as the reviewer-defensible default (no principled criterion to single out mid layers).
- Documented difference: all-layer isolates Geometry and moves Algebra to the Intermediate-Algebra/Precalculus/Prealgebra side; mid-L11-15 grouped Geometry with IntAlg/Precalc and isolated 'Other'.
- This invalidates the OLD grouping G1{Algebra, Counting & Probability, Number Theory, Prealgebra} / G2{Geometry, Intermediate Algebra, Precalculus} / Other.
- The recommended grouping above is derived solely from the all-36-layer S; we do NOT claim membership agreement with the mid-layer view.