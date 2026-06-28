# Curriculum Materials — pooled THINKING ΔA (tag=currmat)

- pooled N = **3025** (pilot1=1608, pilot2=1417); subjects=8, levels=[1, 2, 3, 4, 5, 6, 7, 8]
- is_correct non-null: pilot1=1607, pilot2=1417, total=3024 (axis comparison uses pilot2 test).
- overall 1-shot correct rate (non-null) = 0.818

_Method: group-similarity / centering / perm = OUR diagnostic, not the NAIT PCA-scoring. CPU only. THINKING primary._

## TASK 1 — Difficulty axis (compared on pilot2 TEST)
- PCA fit on pilot1 (n=1608), projected pilot2 (n=1417); EVR=[0.42660000920295715, 0.09669999778270721, 0.053700000047683716] (3s)

### unsupervised PC candidates (ρ on pilot2 test)
| PC | EVR | ρ(level) | ρ(is_correct) | ρ(gen_len) |
|----|-----|----------|---------------|------------|
| PC1 | 0.4266 | +0.709 | -0.378 | +0.729 |
| PC2 | 0.0967 | +0.616 | +0.051 | +0.429 |
| PC3 | 0.0537 | +0.413 | +0.001 | +0.290 |

### supervised ridge (dual; α=10000 via pilot1 5-fold CV, cv ρ=+0.941; 6s)
| axis | ρ(level) | ρ(is_correct) | ρ(gen_len) |
|------|----------|---------------|------------|
| ridge_level | +0.937 | -0.280 | +0.745 |

**ADOPTED difficulty axis (pilot2 test, |ρ(level)| max): `ridge_level` (ρ(level)=+0.937).**
- ridge = honest out-of-sample (pilot1→pilot2); PCs centered by μ_train(pilot1) then projected to pilot2 for the SAME-sample fair comparison. PC1 considered first; PC2/PC3 reported in case PC1 tracks a non-difficulty common component (see ρ(gen_len)).
- [side sanity] pooled-all PC1 ρ(level)=+0.707, ρ(gen_len)=+0.739 (different sample than the comparison above; sanity only).

## TASK 2 — Two-axis (subject / level) decomposition
- units (n≥10): 57

### unit-centroid cosine by pair type (lower = more separated)
- same-level / diff-subject : mean cos = +0.456 (n_pairs=181)  ← subject가 같은 난이도 안에서 가르는 정도
- same-subject / diff-level (by Δlevel):
    Δ=1: mean cos = +0.734 (n=49)
    Δ=2: mean cos = +0.401 (n=41)
    Δ=3: mean cos = +0.019 (n=33)
    Δ=4: mean cos = -0.300 (n=25)
    Δ=5: mean cos = -0.467 (n=18)
    Δ=6: mean cos = -0.485 (n=11)
    Δ=7: mean cos = -0.387 (n=4)
    ordinality ρ(cos, -Δ) = +0.893 (positive => 가까운 level이 더 유사)
- both-diff (baseline)      : mean cos = -0.104 (n_pairs=1234)

### conditional separability (sample-level, block-restricted perm N=200)
- within-level / between-SUBJECT gap = -0.0418 (p=0.0050, blocks_used=200) → 같은 level 안에서 subject 분리도
- within-subject / between-LEVEL gap = +0.2274 (p=0.0050, blocks_used=200) → 같은 subject 안에서 level 분리도

## TASK 3+4 — Unit joint clustering & subject branching

### [layeravg] Ward clustering (U=57 units, feat dim=12288)
| K | silhouette(cosine) |
|---|--------------------|
| 4 | 0.411 |
| 5 | 0.438 |
| 6 | 0.470 |
| 7 | 0.487 |
| 8 | 0.465 |
- best K by silhouette = **7** (sil=0.487)
- dendrogram: dendro_layeravg_currmat.png

#### [layeravg] cluster composition (K=7)
- C1 (n_units=18): levels=[4, 5, 6, 7, 8] | subjects={'Counting & Probability': 4, 'Number Theory': 4, 'Other': 3, 'Intermediate Algebra': 3, 'Geometry': 1, 'Algebra': 1, 'Prealgebra': 1, 'Precalculus': 1} | subj_H=2.75 level_H=2.08 | mean difficulty score=+2.337
- C2 (n_units=9): levels=[4, 5, 6] | subjects={'Geometry': 3, 'Algebra': 2, 'Precalculus': 2, 'Intermediate Algebra': 1, 'Number Theory': 1} | subj_H=2.20 level_H=1.53 | mean difficulty score=+0.964
- C3 (n_units=2): levels=[3, 4] | subjects={'Counting & Probability': 2} | subj_H=-0.00 level_H=1.00 | mean difficulty score=-0.441
- C4 (n_units=4): levels=[3, 4] | subjects={'Algebra': 2, 'Number Theory': 1, 'Prealgebra': 1} | subj_H=1.50 level_H=0.81 | mean difficulty score=-0.752
- C5 (n_units=15): levels=[1, 2, 3] | subjects={'Geometry': 3, 'Algebra': 2, 'Counting & Probability': 2, 'Intermediate Algebra': 2, 'Number Theory': 2, 'Prealgebra': 2, 'Precalculus': 2} | subj_H=2.79 level_H=1.29 | mean difficulty score=-2.446
- C6 (n_units=4): levels=[3, 4] | subjects={'Intermediate Algebra': 2, 'Precalculus': 2} | subj_H=1.00 level_H=1.00 | mean difficulty score=-0.546
- C7 (n_units=5): levels=[1, 2, 3, 4, 5] | subjects={'Other': 5} | subj_H=-0.00 level_H=2.32 | mean difficulty score=-0.852

#### [layeravg] cluster × level
```
level    1  2  3  4  5  6  7  8
cluster                        
1        0  0  0  1  2  4  7  4
2        0  0  0  2  4  3  0  0
3        0  0  1  1  0  0  0  0
4        0  0  3  1  0  0  0  0
5        7  7  1  0  0  0  0  0
6        0  0  2  2  0  0  0  0
7        1  1  1  1  1  0  0  0
```
#### [layeravg] cluster × subject
```
subject  Algebra  Counting & Probability  Geometry  Intermediate Algebra  Number Theory  Other  Prealgebra  Precalculus
cluster                                                                                                                
1              1                       4         1                     3              4      3           1            1
2              2                       0         3                     1              1      0           0            2
3              0                       2         0                     0              0      0           0            0
4              2                       0         0                     0              1      0           1            0
5              2                       2         3                     2              2      0           2            2
6              0                       0         0                     2              0      0           0            2
7              0                       0         0                     0              0      5           0            0
```

- **subject-branching index** = same-level unit-pairs in DIFFERENT clusters = 90/181 = 0.497 (높을수록 같은 난이도가 subject별로 갈라짐 = joint 구조)
- mean per-cluster subject entropy = 1.46 bits, level entropy = 1.43 bits (subjectH 높고 levelH 낮으면 cluster가 주로 level띠)

### [midL11-15] Ward clustering (U=57 units, feat dim=61440)
| K | silhouette(cosine) |
|---|--------------------|
| 4 | 0.369 |
| 5 | 0.375 |
| 6 | 0.391 |
| 7 | 0.385 |
| 8 | 0.362 |
- best K by silhouette = **6** (sil=0.391)
- dendrogram: dendro_midL11-15_currmat.png

#### [midL11-15] cluster composition (K=6)
- C1 (n_units=7): levels=[4, 5, 6] | subjects={'Geometry': 3, 'Precalculus': 2, 'Algebra': 1, 'Intermediate Algebra': 1} | subj_H=1.84 level_H=1.38 | mean difficulty score=+0.975
- C2 (n_units=17): levels=[5, 6, 7, 8] | subjects={'Number Theory': 4, 'Intermediate Algebra': 3, 'Counting & Probability': 3, 'Other': 3, 'Algebra': 2, 'Geometry': 1, 'Precalculus': 1} | subj_H=2.66 level_H=1.78 | mean difficulty score=+2.515
- C3 (n_units=5): levels=[1, 2, 3, 4, 5] | subjects={'Other': 5} | subj_H=-0.00 level_H=2.32 | mean difficulty score=-0.852
- C4 (n_units=4): levels=[3, 4] | subjects={'Intermediate Algebra': 2, 'Precalculus': 2} | subj_H=1.00 level_H=1.00 | mean difficulty score=-0.546
- C5 (n_units=16): levels=[1, 2, 3] | subjects={'Counting & Probability': 3, 'Geometry': 3, 'Algebra': 2, 'Intermediate Algebra': 2, 'Number Theory': 2, 'Prealgebra': 2, 'Precalculus': 2} | subj_H=2.78 level_H=1.42 | mean difficulty score=-2.349
- C6 (n_units=8): levels=[3, 4, 5] | subjects={'Algebra': 2, 'Counting & Probability': 2, 'Number Theory': 2, 'Prealgebra': 2} | subj_H=2.00 level_H=1.41 | mean difficulty score=-0.228

#### [midL11-15] cluster × level
```
level    1  2  3  4  5  6  7  8
cluster                        
1        0  0  0  1  4  2  0  0
2        0  0  0  0  1  5  7  4
3        1  1  1  1  1  0  0  0
4        0  0  2  2  0  0  0  0
5        7  7  2  0  0  0  0  0
6        0  0  3  4  1  0  0  0
```
#### [midL11-15] cluster × subject
```
subject  Algebra  Counting & Probability  Geometry  Intermediate Algebra  Number Theory  Other  Prealgebra  Precalculus
cluster                                                                                                                
1              1                       0         3                     1              0      0           0            2
2              2                       3         1                     3              4      3           0            1
3              0                       0         0                     0              0      5           0            0
4              0                       0         0                     2              0      0           0            2
5              2                       3         3                     2              2      0           2            2
6              2                       2         0                     0              2      0           2            0
```

- **subject-branching index** = same-level unit-pairs in DIFFERENT clusters = 83/181 = 0.459 (높을수록 같은 난이도가 subject별로 갈라짐 = joint 구조)
- mean per-cluster subject entropy = 1.71 bits, level entropy = 1.55 bits (subjectH 높고 levelH 낮으면 cluster가 주로 level띠)

### subject-branching verdict
- branching index: layeravg=0.497 vs mid-layer=0.459
- layeravg view shows stronger same-level splitting. 판정 가이드: branching index가 충분히 크고 cluster×subject가 비대각이면 joint(subject 분기) 구조 = novelty; 거의 level 띠(낮은 branching, level별 정렬)이면 정직하게 'level-driven'으로 보고.

## TASK 5 — Cluster difficulty ordering (MATERIALS, NOT final stages)

### [layeravg] candidate difficulty order (easy→hard by axis `ridge_level`)
  1. C5: score=-2.446, levels=[1, 2, 3], n_units=15
  2. C7: score=-0.852, levels=[1, 2, 3, 4, 5], n_units=5
  3. C4: score=-0.752, levels=[3, 4], n_units=4
  4. C6: score=-0.546, levels=[3, 4], n_units=4
  5. C3: score=-0.441, levels=[3, 4], n_units=2
  6. C2: score=+0.964, levels=[4, 5, 6], n_units=9
  7. C1: score=+2.337, levels=[4, 5, 6, 7, 8], n_units=18

### [midL11-15] candidate difficulty order (easy→hard by axis `ridge_level`)
  1. C5: score=-2.349, levels=[1, 2, 3], n_units=16
  2. C3: score=-0.852, levels=[1, 2, 3, 4, 5], n_units=5
  3. C4: score=-0.546, levels=[3, 4], n_units=4
  4. C6: score=-0.228, levels=[3, 4, 5], n_units=8
  5. C1: score=+0.975, levels=[4, 5, 6], n_units=7
  6. C2: score=+2.515, levels=[5, 6, 7, 8], n_units=17

**stage 개수·경계·schedule은 본 재료를 사용자 review 후 확정.**

## Sparse units (n<10, EXCLUDED from clustering)
- 1 units: Geometry|L8(9)
- 단독 결론 금지; 추후 nearest-cluster 흡수 대상으로만 표기.