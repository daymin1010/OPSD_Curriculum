# SUBJECT 유사도 구조 게이트 (level 통제) — tag=subjsim

> **목적.** 난이도축=GPT level stage 확정. subject 는 내부표현 유사 subject 를
> 같은/인접 stage 에 배치(novelty)하려는 것. 이 리포트는 그 *선결 게이트* —
> "subject 유사도 구조가 LEVEL 오염을 빼고도 비자명하고 안정적인가".
> subject 라벨=GPT, subject 간 관계(유사도)=activation → circularity 약함.

**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. metric = group-centroid layer-averaged cosine (sa.* 재사용). CPU only.

## Population (canonical N)
- raw .pt = **3025**, non-finite drop = **0**, **finite N = 3025** ('3000' 은 별칭).
- provenance: pilot1=1608, pilot2=1417
- subjects (8-canonical): ['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Other', 'Prealgebra', 'Precalculus']
- 레이어 뷰: layeravg(36) 와 mid-L11–15 를 **동등 후보** 로 각각 판정.
- within-level 최소 subject 셀 = 5. (표본 적은 셀 제외/명시.)

**게이트 임계값** (경계 사례는 보류 서술): 구조 std(off-diag)≥0.05; 3방식 평균 행렬상관≥0.6; pilot1-2 행렬상관≥0.6.


## ===== VIEW: layeravg (layers=0..35 (36)) =====

### 작업1 — subject 유사도 행렬 (level 통제, 3 방식)

**(A) within-level (level 안 subject centroid cosine, level 가중평균)**
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   1.000   0.470   0.679   0.704   0.659   0.264   0.678   0.584
 Counting & Probability   0.470   1.000   0.479   0.209   0.670   0.169   0.669   0.125
               Geometry   0.679   0.479   1.000   0.525   0.536   0.237   0.552   0.573
   Intermediate Algebra   0.704   0.209   0.525   1.000   0.450   0.397   0.159   0.765
          Number Theory   0.659   0.670   0.536   0.450   1.000   0.093   0.642   0.251
                  Other   0.264   0.169   0.237   0.397   0.093   1.000   0.045   0.472
             Prealgebra   0.678   0.669   0.552   0.159   0.642   0.045   1.000   0.145
            Precalculus   0.584   0.125   0.573   0.765   0.251   0.472   0.145   1.000
```
per-level 사용 현황: L1(n=335):used 8 subj; L2(n=480):used 8 subj; L3(n=480):used 8 subj; L4(n=437):used 8 subj; L5(n=420):used 7 subj; L6(n=420):used 7 subj; L7(n=387):used 7 subj; L8(n=66):used 5 subj

**(B-main) GPT-level centroid 차감 후 subject cosine**
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   1.000  -0.395  -0.009   0.386   0.126  -0.445   0.121   0.048
 Counting & Probability  -0.395   1.000   0.009  -0.732   0.403  -0.285   0.389  -0.705
               Geometry  -0.009   0.009   1.000  -0.221  -0.167  -0.388   0.050   0.013
   Intermediate Algebra   0.386  -0.732  -0.221   1.000  -0.264   0.051  -0.525   0.689
          Number Theory   0.126   0.403  -0.167  -0.264   1.000  -0.564   0.196  -0.632
                  Other  -0.445  -0.285  -0.388   0.051  -0.564   1.000  -0.292   0.343
             Prealgebra   0.121   0.389   0.050  -0.525   0.196  -0.292   1.000  -0.594
            Precalculus   0.048  -0.705   0.013   0.689  -0.632   0.343  -0.594   1.000
```

**(B-aux) ridge_level projection 제거 후 subject cosine**
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   1.000  -0.395  -0.009   0.393   0.080  -0.434   0.067   0.121
 Counting & Probability  -0.395   1.000   0.006  -0.733   0.428  -0.299   0.423  -0.730
               Geometry  -0.009   0.006   1.000  -0.216  -0.180  -0.386   0.029   0.033
   Intermediate Algebra   0.393  -0.733  -0.216   1.000  -0.285   0.058  -0.530   0.697
          Number Theory   0.080   0.428  -0.180  -0.285   1.000  -0.539   0.195  -0.641
                  Other  -0.434  -0.299  -0.386   0.058  -0.539   1.000  -0.275   0.311
             Prealgebra   0.067   0.423   0.029  -0.530   0.195  -0.275   1.000  -0.586
            Precalculus   0.121  -0.730   0.033   0.697  -0.641   0.311  -0.586   1.000
```

### 작업1 — 3 방식 행렬 일관성 (off-diag Pearson r)
- r(A, Bmain) = +0.849
- r(A, Baux) = +0.843
- r(Bmain, Baux) = +0.998
- **평균 일관성 r = +0.897**

### 작업3 — 구조 비자명성: off-diag(A) mean=+0.436, std=0.218, min=+0.045, max=+0.765
### 작업3 — 안정성: pilot1 vs pilot2 within-level(A) 행렬 r = **+0.943**

### 작업3 — 직관 sanity (A 행렬)
- cos(Algebra, Intermediate Algebra) = +0.704
- cos(Algebra, Prealgebra) = +0.678
- cos(Algebra, Precalculus) = +0.584
- cos(Number Theory, Counting & Probability) = +0.670
- cos(Geometry, Number Theory) = +0.536
- 각 subject 최근접(A):
    Algebra → Intermediate Algebra (+0.704)
    Counting & Probability → Number Theory (+0.670)
    Geometry → Algebra (+0.679)
    Intermediate Algebra → Precalculus (+0.765)
    Number Theory → Counting & Probability (+0.670)
    Other → Precalculus (+0.472)
    Prealgebra → Algebra (+0.678)
    Precalculus → Intermediate Algebra (+0.765)

### 작업3 — hierarchical clustering (A, average-linkage)
- 2-cluster cut: G1={Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus} | G2={Other}
- 3-cluster cut: G1={Counting & Probability, Number Theory, Prealgebra} | G2={Algebra, Geometry, Intermediate Algebra, Precalculus} | G3={Other}

### >>> VIEW [layeravg] 게이트: **PASS**  (struct_std=0.218✓, consist_r=+0.897✓, stable_r=+0.943✓)

## ===== VIEW: mid_L11-15 (layers=[11, 12, 13, 14, 15]) =====

### 작업1 — subject 유사도 행렬 (level 통제, 3 방식)

**(A) within-level (level 안 subject centroid cosine, level 가중평균)**
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   1.000   0.398   0.619   0.664   0.590   0.142   0.669   0.503
 Counting & Probability   0.398   1.000   0.357   0.092   0.623   0.121   0.615  -0.032
               Geometry   0.619   0.357   1.000   0.477   0.411   0.136   0.487   0.523
   Intermediate Algebra   0.664   0.092   0.477   1.000   0.342   0.300   0.097   0.729
          Number Theory   0.590   0.623   0.411   0.342   1.000  -0.039   0.610   0.068
                  Other   0.142   0.121   0.136   0.300  -0.039   1.000  -0.016   0.411
             Prealgebra   0.669   0.615   0.487   0.097   0.610  -0.016   1.000   0.066
            Precalculus   0.503  -0.032   0.523   0.729   0.068   0.411   0.066   1.000
```
per-level 사용 현황: L1(n=335):used 8 subj; L2(n=480):used 8 subj; L3(n=480):used 8 subj; L4(n=437):used 8 subj; L5(n=420):used 7 subj; L6(n=420):used 7 subj; L7(n=387):used 7 subj; L8(n=66):used 5 subj

**(B-main) GPT-level centroid 차감 후 subject cosine**
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   1.000  -0.362   0.040   0.381   0.124  -0.528   0.159   0.033
 Counting & Probability  -0.362   1.000  -0.123  -0.725   0.409  -0.218   0.353  -0.732
               Geometry   0.040  -0.123   1.000  -0.112  -0.205  -0.385  -0.036   0.129
   Intermediate Algebra   0.381  -0.725  -0.112   1.000  -0.265  -0.040  -0.536   0.678
          Number Theory   0.124   0.409  -0.205  -0.265   1.000  -0.564   0.252  -0.673
                  Other  -0.528  -0.218  -0.385  -0.040  -0.564   1.000  -0.223   0.292
             Prealgebra   0.159   0.353  -0.036  -0.536   0.252  -0.223   1.000  -0.615
            Precalculus   0.033  -0.732   0.129   0.678  -0.673   0.292  -0.615   1.000
```

**(B-aux) ridge_level projection 제거 후 subject cosine**
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   1.000  -0.368   0.040   0.393   0.089  -0.518   0.092   0.096
 Counting & Probability  -0.368   1.000  -0.127  -0.728   0.428  -0.229   0.384  -0.743
               Geometry   0.040  -0.127   1.000  -0.105  -0.220  -0.381  -0.068   0.152
   Intermediate Algebra   0.393  -0.728  -0.105   1.000  -0.286  -0.032  -0.556   0.682
          Number Theory   0.089   0.428  -0.220  -0.286   1.000  -0.547   0.258  -0.684
                  Other  -0.518  -0.229  -0.381  -0.032  -0.547   1.000  -0.187   0.263
             Prealgebra   0.092   0.384  -0.068  -0.556   0.258  -0.187   1.000  -0.618
            Precalculus   0.096  -0.743   0.152   0.682  -0.684   0.263  -0.618   1.000
```

### 작업1 — 3 방식 행렬 일관성 (off-diag Pearson r)
- r(A, Bmain) = +0.887
- r(A, Baux) = +0.876
- r(Bmain, Baux) = +0.998
- **평균 일관성 r = +0.920**

### 작업3 — 구조 비자명성: off-diag(A) mean=+0.356, std=0.243, min=-0.039, max=+0.729
### 작업3 — 안정성: pilot1 vs pilot2 within-level(A) 행렬 r = **+0.966**

### 작업3 — 직관 sanity (A 행렬)
- cos(Algebra, Intermediate Algebra) = +0.664
- cos(Algebra, Prealgebra) = +0.669
- cos(Algebra, Precalculus) = +0.503
- cos(Number Theory, Counting & Probability) = +0.623
- cos(Geometry, Number Theory) = +0.411
- 각 subject 최근접(A):
    Algebra → Prealgebra (+0.669)
    Counting & Probability → Number Theory (+0.623)
    Geometry → Algebra (+0.619)
    Intermediate Algebra → Precalculus (+0.729)
    Number Theory → Counting & Probability (+0.623)
    Other → Precalculus (+0.411)
    Prealgebra → Algebra (+0.669)
    Precalculus → Intermediate Algebra (+0.729)

### 작업3 — hierarchical clustering (A, average-linkage)
- 2-cluster cut: G1={Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus} | G2={Other}
- 3-cluster cut: G1={Algebra, Counting & Probability, Number Theory, Prealgebra} | G2={Geometry, Intermediate Algebra, Precalculus} | G3={Other}

### >>> VIEW [mid_L11-15] 게이트: **PASS**  (struct_std=0.243✓, consist_r=+0.920✓, stable_r=+0.966✓)

## ===== 작업2 — Supervised 대조 (검증용, mid-L11–15 PCA→LDA) =====
- PCA(150) → LDA; pilot1 train(n=1608) / pilot2 test(n=1417); **macro-F1 = 0.697** (chance≈0.125).
- subject confusion matrix (row-normalized, test):
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   0.567   0.029   0.014   0.176   0.024   0.019   0.124   0.048
 Counting & Probability   0.038   0.814   0.005   0.019   0.033   0.038   0.048   0.005
               Geometry   0.021   0.021   0.765   0.000   0.016   0.032   0.027   0.118
   Intermediate Algebra   0.228   0.011   0.000   0.589   0.017   0.061   0.006   0.089
          Number Theory   0.090   0.024   0.005   0.024   0.790   0.010   0.057   0.000
                  Other   0.044   0.089   0.000   0.072   0.006   0.744   0.006   0.039
             Prealgebra   0.122   0.011   0.022   0.033   0.067   0.022   0.711   0.011
            Precalculus   0.060   0.000   0.080   0.093   0.007   0.120   0.013   0.627
```
- LDA 공간 subject centroid 거리 (작을수록 유사):
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   0.000   4.263   4.250   1.585   3.410   3.596   2.806   2.640
 Counting & Probability   4.263   0.000   5.685   4.609   4.378   4.597   4.172   4.991
               Geometry   4.250   5.685   0.000   4.536   5.624   5.173   4.787   3.494
   Intermediate Algebra   1.585   4.609   4.536   0.000   3.828   3.293   3.563   1.953
          Number Theory   3.410   4.378   5.624   3.828   0.000   4.823   3.682   4.703
                  Other   3.596   4.597   5.173   3.293   4.823   0.000   4.469   3.423
             Prealgebra   2.806   4.172   4.787   3.563   3.682   4.469   0.000   4.063
            Precalculus   2.640   4.991   3.494   1.953   4.703   3.423   4.063   0.000
```

### 작업2 — LDA confusion(대칭) vs unsup A(mid) 행렬 r = **+0.555** (양수 = 헷갈리는 subject 쌍이 unsup 에서도 고유사 → 대조 일치)

## ===== 종합 게이트 판정 =====
- **layeravg**: PASS (struct_std=0.218, consist_r=+0.897, stable_r=+0.943)
- **mid_L11-15**: PASS (struct_std=0.243, consist_r=+0.920, stable_r=+0.966)

**판정 규칙**: subject 신호는 mid-layer 에 집중 → layeravg 는 희석될 수 있으므로 두 view 중 *어느 하나라도* (특히 mid) PASS 면 그 view 를 배치 근거로 채택. 둘 다 FAIL 이면 subject 는 mixing(다양성) 용도로만 권고.

### ⇒ 종합: **PASS** (배치 근거 view = `mid_L11-15`). 작업4 배치 재료 생성.

## ===== 작업4 — 배치 재료 (확정 X, view=mid_L11-15) =====
- subject 그룹 후보 (3-cluster cut @ mid_L11-15):
    그룹 1: {Algebra, Counting & Probability, Number Theory, Prealgebra}
    그룹 2: {Geometry, Intermediate Algebra, Precalculus}
    그룹 3: {Other}
- 그룹 간 평균 cosine (높을수록 인접 → 인접 stage 후보):
    G1↔G2: +0.306
    G1↔G3: +0.052
    G2↔G3: +0.282

- **결합 골격(확정 X)**: 난이도축=GPT level stage(고정) × subject 그룹(위 cluster). 각 level stage 내부에서 유사 subject 그룹을 같은/인접 배치, 그룹 간 거리로 인접성 결정. stage 경계·schedule·혼합비는 이후 세션에서 점진 확정.