# SUBJECT 단독 검정 (LEVEL 통제) — 2026-06-22  (tag=subjctrl)

> **가설.** 직전 세션에서 unit(subject×level) 구조는 LEVEL이 지배하고 SUBJECT는
> 거의 무신호(within/across≈1.0, silhouette 음수)였다. "level 신호의 절댓값이
> 더 커서 subject 신호가 묻힌다"면, level을 고정/제거하면 subject 신호가
> 살아나야 한다. 이를 (1) level-고정 pairwise 비교, (2) 순열검정, (3) level-잔차
> 후 silhouette 로 검정한다.

**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. finite **N=3025** (raw 3025, non-finite drop 0). CPU only. seed=42, n_perm=2000.
- provenance: pilot1=1608, pilot2=1417
- metric: per-layer L2-normalized → layer-averaged sample-pairwise cosine. views = ['layeravg', 'mid_L11-15']. (mid = 직전 세션 subject 윈도우 참고값)

## view = `layeravg` · space = `raw`

### (1) LEVEL 고정 → SUBJECT 단독 검정
- within-level **same-subject** mean cos = +0.1822 (n_pairs=85359); **diff-subject** = +0.1073 (n_pairs=548588); ratio = **1.699x**
- **Cohen's d (same−diff) = +0.325**; Mann–Whitney p(same>diff) = 0.00e+00
- **permutation**(level 고정, subject 셔플 ×2000): stat=+0.07502, **p=0.0005** (blocks=8)
- per-level (same / diff / n_same / n_diff):
    L1 (n=335): same=+0.3857 diff=+0.3014  (8295/47650)
    L2 (n=480): same=+0.2769 diff=+0.1730  (14160/100800)
    L3 (n=480): same=+0.1663 diff=+0.0662  (14160/100800)
    L4 (n=437): same=+0.1160 diff=+0.0333  (12526/82740)
    L5 (n=420): same=+0.1050 diff=+0.0506  (12390/75600)
    L6 (n=420): same=+0.1243 diff=+0.0748  (12390/75600)
    L7 (n=387): same=+0.1542 diff=+0.1231  (10971/63720)
    L8 (n=66): same=+0.1980 diff=+0.1696  (467/1678)

### (2) 대칭 대조 — SUBJECT 고정 → LEVEL 단독 검정
- within-subject **same-level** mean cos = +0.1822 (n_pairs=85359); **diff-level** = +0.0431 (n_pairs=506927); ratio = **4.230x**
- **Cohen's d = +0.597**; Mann–Whitney p = 0.00e+00
- **permutation**(subject 고정, level 셔플 ×2000): stat=+0.13780, **p=0.0005** (blocks=8)

### (3a) within-level subject centroid cosine (level 가중평균)
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
- 각 subject 최근접 과목:
    Algebra → Intermediate Algebra (+0.704)
    Counting & Probability → Number Theory (+0.670)
    Geometry → Algebra (+0.679)
    Intermediate Algebra → Precalculus (+0.765)
    Number Theory → Counting & Probability (+0.670)
    Other → Precalculus (+0.472)
    Prealgebra → Algebra (+0.678)
    Precalculus → Intermediate Algebra (+0.765)

### (3b) subject silhouette (raw) = **-0.0227** (직전 unit-report subject raw=-0.160)

## view = `layeravg` · space = `resid_on_level`

### (1) LEVEL 고정 → SUBJECT 단독 검정
- within-level **same-subject** mean cos = +0.0864 (n_pairs=85359); **diff-subject** = +0.0015 (n_pairs=548588); ratio = **55.920x**
- **Cohen's d (same−diff) = +0.363**; Mann–Whitney p(same>diff) = 0.00e+00
- **permutation**(level 고정, subject 셔플 ×2000): stat=+0.08608, **p=0.0005** (blocks=8)
- per-level (same / diff / n_same / n_diff):
    L1 (n=335): same=+0.1105 diff=-0.0135  (8295/47650)
    L2 (n=480): same=+0.1200 diff=-0.0123  (14160/100800)
    L3 (n=480): same=+0.0995 diff=-0.0113  (14160/100800)
    L4 (n=437): same=+0.0854 diff=-0.0002  (12526/82740)
    L5 (n=420): same=+0.0823 diff=+0.0275  (12390/75600)
    L6 (n=420): same=+0.0662 diff=+0.0173  (12390/75600)
    L7 (n=387): same=+0.0398 diff=+0.0082  (10971/63720)
    L8 (n=66): same=+0.0102 diff=-0.0125  (467/1678)

### (2) 대칭 대조 — SUBJECT 고정 → LEVEL 단독 검정
- within-subject **same-level** mean cos = +0.0864 (n_pairs=85359); **diff-level** = +0.0572 (n_pairs=506927); ratio = **1.512x**
- **Cohen's d = +0.141**; Mann–Whitney p = 2.68e-153
- **permutation**(subject 고정, level 셔플 ×2000): stat=+0.02970, **p=0.0005** (blocks=8)

### (3b) subject silhouette (resid_on_level) = **-0.0051** (직전 unit-report subject raw=-0.160)

## view = `mid_L11-15` · space = `raw`

### (1) LEVEL 고정 → SUBJECT 단독 검정
- within-level **same-subject** mean cos = +0.1644 (n_pairs=85359); **diff-subject** = +0.0755 (n_pairs=548588); ratio = **2.176x**
- **Cohen's d (same−diff) = +0.433**; Mann–Whitney p(same>diff) = 0.00e+00
- **permutation**(level 고정, subject 셔플 ×2000): stat=+0.08839, **p=0.0005** (blocks=8)
- per-level (same / diff / n_same / n_diff):
    L1 (n=335): same=+0.3311 diff=+0.2345  (8295/47650)
    L2 (n=480): same=+0.2386 diff=+0.1176  (14160/100800)
    L3 (n=480): same=+0.1460 diff=+0.0330  (14160/100800)
    L4 (n=437): same=+0.1051 diff=+0.0113  (12526/82740)
    L5 (n=420): same=+0.0958 diff=+0.0264  (12390/75600)
    L6 (n=420): same=+0.1198 diff=+0.0559  (12390/75600)
    L7 (n=387): same=+0.1598 diff=+0.1197  (10971/63720)
    L8 (n=66): same=+0.2033 diff=+0.1719  (467/1678)

### (2) 대칭 대조 — SUBJECT 고정 → LEVEL 단독 검정
- within-subject **same-level** mean cos = +0.1644 (n_pairs=85359); **diff-level** = +0.0484 (n_pairs=506927); ratio = **3.394x**
- **Cohen's d = +0.563**; Mann–Whitney p = 0.00e+00
- **permutation**(subject 고정, level 셔플 ×2000): stat=+0.11513, **p=0.0005** (blocks=8)

### (3a) within-level subject centroid cosine (level 가중평균)
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
- 각 subject 최근접 과목:
    Algebra → Prealgebra (+0.669)
    Counting & Probability → Number Theory (+0.623)
    Geometry → Algebra (+0.619)
    Intermediate Algebra → Precalculus (+0.729)
    Number Theory → Counting & Probability (+0.623)
    Other → Precalculus (+0.411)
    Prealgebra → Algebra (+0.669)
    Precalculus → Intermediate Algebra (+0.729)

### (3b) subject silhouette (raw) = **-0.0131** (직전 unit-report subject raw=-0.160)

## view = `mid_L11-15` · space = `resid_on_level`

### (1) LEVEL 고정 → SUBJECT 단독 검정
- within-level **same-subject** mean cos = +0.0862 (n_pairs=85359); **diff-subject** = -0.0090 (n_pairs=548588); ratio = **-9.625x**
- **Cohen's d (same−diff) = +0.479**; Mann–Whitney p(same>diff) = 0.00e+00
- **permutation**(level 고정, subject 셔플 ×2000): stat=+0.09668, **p=0.0005** (blocks=8)
- per-level (same / diff / n_same / n_diff):
    L1 (n=335): same=+0.1135 diff=-0.0136  (8295/47650)
    L2 (n=480): same=+0.1262 diff=-0.0146  (14160/100800)
    L3 (n=480): same=+0.1049 diff=-0.0144  (14160/100800)
    L4 (n=437): same=+0.0855 diff=-0.0102  (12526/82740)
    L5 (n=420): same=+0.0665 diff=-0.0043  (12390/75600)
    L6 (n=420): same=+0.0640 diff=-0.0011  (12390/75600)
    L7 (n=387): same=+0.0411 diff=-0.0008  (10971/63720)
    L8 (n=66): same=+0.0161 diff=-0.0171  (467/1678)

### (2) 대칭 대조 — SUBJECT 고정 → LEVEL 단독 검정
- within-subject **same-level** mean cos = +0.0862 (n_pairs=85359); **diff-level** = +0.0628 (n_pairs=506927); ratio = **1.374x**
- **Cohen's d = +0.127**; Mann–Whitney p = 1.08e-78
- **permutation**(subject 고정, level 셔플 ×2000): stat=+0.02430, **p=0.0005** (blocks=8)

### (3b) subject silhouette (resid_on_level) = **+0.0016** (직전 unit-report subject raw=-0.160)

## 요약 — LEVEL 통제 후 SUBJECT vs (대칭) SUBJECT 통제 후 LEVEL

| view | space | subj(level통제) ratio·d·p | level(subj통제) ratio·d·p |
|---|---|---|---|
| `layeravg` | raw | 1.699x · d=+0.325 · p=0.0005 | 4.230x · d=+0.597 · p=0.0005 |
| `layeravg` | resid_on_level | 55.920x · d=+0.363 · p=0.0005 | 1.512x · d=+0.141 · p=0.0005 |
| `mid_L11-15` | raw | 2.176x · d=+0.433 · p=0.0005 | 3.394x · d=+0.563 · p=0.0005 |
| `mid_L11-15` | resid_on_level | -9.625x · d=+0.479 · p=0.0005 | 1.374x · d=+0.127 · p=0.0005 |

## 결론

1. **LEVEL 고정 후 subject 신호**(layeravg/raw): within/across = 1.699x, Cohen's d = +0.325, permutation p = 0.0005 → 유의: level 통제 시 subject 신호 존재(=level 교란에 묻혀 있었음).
2. **대칭 대조(subject 고정 후 level)**: Cohen's d = +0.597, p = 0.0005. 두 효과크기 비교(|d_subject|=0.325 vs |d_level|=0.597) → subject가 더 약함(=여전히 level 우위).
3. **silhouette**(위 §3b): level-residual 후 subject silhouette 가 raw 대비 개선되는지 각 view 항목 참조 — 음수 유지면 subject 군집은 여전히 약함.

> 주의: 8과목/소표본 + 불균형 설계. permutation p 와 효과크기, raw/resid 일관성을 함께 보고 단일 수치로 강결론 금지. ratio≈1·d≈0·p≫0.05 면 'level 통제 후에도 subject 무신호' = 직전 결론을 (교란 통제 후에도) 재확인하는 것.

### 부록: 직전 보고서 §3 정정 메모
- 직전 unit-report §3 "고립=고난도" 문구는 데이터(고립=저난도 Pcalc/Geom/Algebra)와 반대 → "고립 과목은 **저난도** 경향"으로 정정 필요(본 분석 범위 밖, 메모만).