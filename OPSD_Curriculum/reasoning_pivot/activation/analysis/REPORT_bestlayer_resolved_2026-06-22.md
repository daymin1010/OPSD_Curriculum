# best-layer 윈도우별 SUBJECT/LEVEL/UNIT 정밀 공정 비교 — 2026-06-22  (tag=bestlayer)

> **목적.** 직전 보고서(`REPORT_levelunit_controlled_2026-06-22.md`)는 §B/§C 를
> `levwin_L16-32`(=level 우세 구간) 한 윈도우로만 SUBJECT/LEVEL 을 같이 평가해서
> subject 한테 불리했다. 본 보고서는 각 축마다 *그 축에 유리한 윈도우*(데이터에서
> 도출된 Fisher∪probe-F1 Top-5 union) 를 쓰는 *공정 비교* 를 별도 파일로 기록한다.
> 새 통계 로직은 없음 — `subject_layer_resolved`/`level_unit_resolved` 의 헬퍼만 호출.

**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. finite **N=3025** (raw 3025, non-finite drop 0). units=58. CPU only. seed=42.
- provenance: pilot1=1608, pilot2=1417

**윈도우 상수 (직전 `levunit_artifacts.npz` 의 36-vec Fisher/probe-F1 에서 도출)**
- `W_SUBJ` = [9, 10, 11, 12, 14]  (SUBJECT Fisher∪probe-F1 Top-5 union, n=5)
- `W_LEV ` = [20, 25, 26, 27, 29, 30, 31]  (LEVEL Fisher∪probe-F1 Top-5 union, n=7)
- `W_ALL ` = list(range(36))  (전 레이어 기준선)
- `W_SUBJ ∩ W_LEV` = ∅  (두 신호는 서로 다른 레이어 대역에 분리)

**측정 매트릭스**
- §S SUBJECT 정밀검증 : 윈도우 = W_SUBJ, W_ALL  (블록=level, 라벨=subject)
- §L LEVEL   정밀검증 : 윈도우 = W_LEV,  W_ALL  (블록=subject, 라벨=level)
- §U UNIT    정밀검증 : 윈도우 = W_SUBJ, W_LEV, W_ALL  (silhouette + η² + cohesion)

## §S. SUBJECT 정밀검증 (block=level, label=subject)

metric = per-layer L2-norm 후 layer-averaged sample-pairwise cosine. level 블록 안에서 same-subject vs diff-subject 비교(→ level 오염 0). block-permutation 은 level 고정·subject 라벨만 셔플.

### S/Wsubj  layers=[9, 10, 11, 12, 14]
- within-level **same-subject** mean cos = +0.1633 (n_pairs=85359); **diff-subject** mean cos = +0.0707 (n_pairs=548588); ratio = **2.311x**
- **Cohen's d (same−diff) = +0.446**; Mann–Whitney p(same>diff) = 0.00e+00
- **block-permutation**(level 고정, subject 셔플 ×1000): stat(mean_same−mean_diff)=+0.0920, **p=0.0010** (blocks=8)
- per-level (same_mean / diff_mean / n_same / n_diff):
    level=1 (n=335): same=+0.3208 diff=+0.2224  (8295/47650 pairs)
    level=2 (n=480): same=+0.2341 diff=+0.1101  (14160/100800 pairs)
    level=3 (n=480): same=+0.1421 diff=+0.0260  (14160/100800 pairs)
    level=4 (n=437): same=+0.1051 diff=+0.0070  (12526/82740 pairs)
    level=5 (n=420): same=+0.0981 diff=+0.0239  (12390/75600 pairs)
    level=6 (n=420): same=+0.1239 diff=+0.0545  (12390/75600 pairs)
    level=7 (n=387): same=+0.1627 diff=+0.1201  (10971/63720 pairs)
    level=8 (n=66): same=+0.2052 diff=+0.1758  (467/1678 pairs)
- LDA (PCA(150)→LDA, pilot1→pilot2): **macro-F1=0.703** (chance≈0.125)
  confusion (row-normalized, test):
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   0.586   0.019   0.010   0.181   0.024   0.024   0.124   0.033
 Counting & Probability   0.057   0.790   0.010   0.024   0.038   0.038   0.043   0.000
               Geometry   0.021   0.016   0.786   0.021   0.011   0.032   0.021   0.091
   Intermediate Algebra   0.211   0.006   0.000   0.617   0.011   0.067   0.006   0.083
          Number Theory   0.095   0.033   0.005   0.019   0.790   0.010   0.048   0.000
                  Other   0.050   0.067   0.000   0.083   0.011   0.750   0.006   0.033
             Prealgebra   0.133   0.000   0.022   0.033   0.078   0.000   0.722   0.011
            Precalculus   0.067   0.000   0.080   0.127   0.007   0.107   0.013   0.600
```
  top 혼동쌍(=내부 유사): Algebra↔Intermediate Algebra=0.196; Algebra↔Prealgebra=0.129; Intermediate Algebra↔Precalculus=0.105; Geometry↔Precalculus=0.085; Intermediate Algebra↔Other=0.075

### S/Wall  layers=[L0..L35 n=36]
- within-level **same-subject** mean cos = +0.1822 (n_pairs=85359); **diff-subject** mean cos = +0.1073 (n_pairs=548588); ratio = **1.699x**
- **Cohen's d (same−diff) = +0.325**; Mann–Whitney p(same>diff) = 0.00e+00
- **block-permutation**(level 고정, subject 셔플 ×1000): stat(mean_same−mean_diff)=+0.0750, **p=0.0010** (blocks=8)
- per-level (same_mean / diff_mean / n_same / n_diff):
    level=1 (n=335): same=+0.3857 diff=+0.3014  (8295/47650 pairs)
    level=2 (n=480): same=+0.2769 diff=+0.1730  (14160/100800 pairs)
    level=3 (n=480): same=+0.1663 diff=+0.0662  (14160/100800 pairs)
    level=4 (n=437): same=+0.1160 diff=+0.0333  (12526/82740 pairs)
    level=5 (n=420): same=+0.1050 diff=+0.0506  (12390/75600 pairs)
    level=6 (n=420): same=+0.1243 diff=+0.0748  (12390/75600 pairs)
    level=7 (n=387): same=+0.1542 diff=+0.1231  (10971/63720 pairs)
    level=8 (n=66): same=+0.1980 diff=+0.1696  (467/1678 pairs)
- LDA (PCA(150)→LDA, pilot1→pilot2): **macro-F1=0.633** (chance≈0.125)
  confusion (row-normalized, test):
```
                        Algebra Countin Geometr Interme Number    Other Prealge Precalc
                Algebra   0.467   0.024   0.048   0.195   0.081   0.024   0.114   0.048
 Counting & Probability   0.033   0.733   0.010   0.014   0.100   0.052   0.043   0.014
               Geometry   0.027   0.043   0.738   0.053   0.016   0.037   0.016   0.070
   Intermediate Algebra   0.211   0.011   0.022   0.506   0.028   0.050   0.006   0.167
          Number Theory   0.076   0.048   0.014   0.043   0.762   0.005   0.052   0.000
                  Other   0.033   0.078   0.017   0.072   0.028   0.711   0.011   0.050
             Prealgebra   0.111   0.033   0.044   0.011   0.111   0.000   0.633   0.056
            Precalculus   0.053   0.013   0.087   0.167   0.007   0.113   0.013   0.547
```
  top 혼동쌍(=내부 유사): Algebra↔Intermediate Algebra=0.203; Intermediate Algebra↔Precalculus=0.167; Algebra↔Prealgebra=0.113; Number Theory↔Prealgebra=0.082; Other↔Precalculus=0.082

## §L. LEVEL 정밀검증 (block=subject, label=level)

metric 동일. subject 블록 안에서 same-level vs diff-level 비교(→ subject 오염 0). block-permutation 은 subject 고정·level 라벨만 셔플.

### L/Wlev  layers=[20, 25, 26, 27, 29, 30, 31]
- within-subject **same-level** mean cos = +0.2304 (n_pairs=85359); **diff-level** mean cos = +0.0411 (n_pairs=506927); ratio = **5.608x**
- **Cohen's d (same−diff) = +0.684**; Mann–Whitney p(same>diff) = 0.00e+00
- **block-permutation**(subject 고정, level 셔플 ×1000): stat(mean_same−mean_diff)=+0.1872, **p=0.0010** (blocks=8)
- per-subject (same_mean / diff_mean / n_same / n_diff):
    Algebra (n=420): same=+0.2609 diff=+0.0236  (12390/75600 pairs)
    Counting & Probability (n=430): same=+0.2246 diff=+0.0344  (12435/79800 pairs)
    Geometry (n=406): same=+0.2315 diff=+0.0155  (11322/70893 pairs)
    Intermediate Algebra (n=387): same=+0.2152 diff=+0.0282  (10795/63896 pairs)
    Number Theory (n=443): same=+0.2320 diff=+0.0311  (12643/85260 pairs)
    Other (n=397): same=+0.2062 diff=+0.0995  (10974/67632 pairs)
    Prealgebra (n=197): same=+0.2884 diff=+0.1339  (5446/13860 pairs)
    Precalculus (n=345): same=+0.2066 diff=+0.0430  (9354/49986 pairs)
- LDA (PCA(150)→LDA, pilot1→pilot2): **macro-F1=0.577** (chance≈0.143)
  confusion (row-normalized, test):
```
               1       2       3       4       5       6       7
       1   0.835   0.165   0.000   0.000   0.000   0.000   0.000
       2   0.146   0.688   0.125   0.033   0.004   0.004   0.000
       3   0.004   0.171   0.537   0.246   0.042   0.000   0.000
       4   0.000   0.000   0.210   0.429   0.305   0.048   0.010
       5   0.000   0.010   0.043   0.224   0.424   0.252   0.048
       6   0.000   0.000   0.005   0.019   0.211   0.531   0.234
       7   0.000   0.000   0.000   0.006   0.069   0.263   0.662
```
  top 혼동쌍(=내부 유사): 4↔5=0.264; 6↔7=0.248; 5↔6=0.231; 3↔4=0.228; 1↔2=0.156

### L/Wall  layers=[L0..L35 n=36]
- within-subject **same-level** mean cos = +0.1822 (n_pairs=85359); **diff-level** mean cos = +0.0431 (n_pairs=506927); ratio = **4.230x**
- **Cohen's d (same−diff) = +0.597**; Mann–Whitney p(same>diff) = 0.00e+00
- **block-permutation**(subject 고정, level 셔플 ×1000): stat(mean_same−mean_diff)=+0.1378, **p=0.0010** (blocks=8)
- per-subject (same_mean / diff_mean / n_same / n_diff):
    Algebra (n=420): same=+0.1985 diff=+0.0208  (12390/75600 pairs)
    Counting & Probability (n=430): same=+0.1824 diff=+0.0473  (12435/79800 pairs)
    Geometry (n=406): same=+0.1829 diff=+0.0253  (11322/70893 pairs)
    Intermediate Algebra (n=387): same=+0.1755 diff=+0.0310  (10795/63896 pairs)
    Number Theory (n=443): same=+0.1927 diff=+0.0417  (12643/85260 pairs)
    Other (n=397): same=+0.1560 diff=+0.0804  (10974/67632 pairs)
    Prealgebra (n=197): same=+0.2243 diff=+0.1103  (5446/13860 pairs)
    Precalculus (n=345): same=+0.1592 diff=+0.0440  (9354/49986 pairs)
- LDA (PCA(150)→LDA, pilot1→pilot2): **macro-F1=0.574** (chance≈0.143)
  confusion (row-normalized, test):
```
               1       2       3       4       5       6       7
       1   0.835   0.165   0.000   0.000   0.000   0.000   0.000
       2   0.138   0.700   0.125   0.033   0.004   0.000   0.000
       3   0.004   0.167   0.550   0.246   0.033   0.000   0.000
       4   0.000   0.000   0.200   0.462   0.290   0.048   0.000
       5   0.000   0.005   0.033   0.220   0.421   0.258   0.062
       6   0.000   0.005   0.000   0.019   0.240   0.481   0.255
       7   0.000   0.000   0.006   0.006   0.057   0.287   0.643
```
  top 혼동쌍(=내부 유사): 6↔7=0.271; 4↔5=0.255; 5↔6=0.249; 3↔4=0.223; 1↔2=0.151

## §U. UNIT(subject×level) 정밀검증

metric: same-unit vs diff-unit *전체쌍* cosine (블록 없음). silhouette 는 동일 (1−cos) precomputed distance 로 unit/level/subject 3 라벨 모두 계산. level-residualize(ssg.level_centroid_residual) 후 subject/unit silhouette 재계산. η² 는 전 레이어 1번 계산 후 윈도우 평균.

### U/Wsubj  layers=[9, 10, 11, 12, 14]
- same-unit cos = +0.1633 (n=85359) / diff-unit cos = +0.0006 (n=4488441); ratio = **296.221x**; Cohen's d = +0.846
- silhouette: unit=-0.1031, level=-0.0338, subject=-0.0115
- (level-residual) subject silhouette -0.0115 → +0.0015; unit silhouette -0.1031 → -0.0868
- 윈도우 평균 η²: level=0.090, subject=0.060, interaction=0.023

### U/Wlev  layers=[20, 25, 26, 27, 29, 30, 31]
- same-unit cos = +0.2304 (n=85359) / diff-unit cos = +0.0151 (n=4488441); ratio = **15.257x**; Cohen's d = +0.821
- silhouette: unit=-0.1425, level=-0.0425, subject=-0.0350
- (level-residual) subject silhouette -0.0350 → -0.0120; unit silhouette -0.1425 → -0.1047
- 윈도우 평균 η²: level=0.163, subject=0.045, interaction=0.022

### U/Wall  layers=[L0..L35 n=36]
- same-unit cos = +0.1822 (n=85359) / diff-unit cos = +0.0120 (n=4488441); ratio = **15.233x**; Cohen's d = +0.786
- silhouette: unit=-0.1141, level=-0.0354, subject=-0.0227
- (level-residual) subject silhouette -0.0227 → -0.0051; unit silhouette -0.1141 → -0.0850
- 윈도우 평균 η²: level=0.110, subject=0.041, interaction=0.021

## §X. cross-window 비교표 (요약)

**SUBJECT / LEVEL** (supervised + permutation)

| axis | window | n_layers | same / diff cos | ratio | Cohen's d | MWU p | perm p | LDA macro-F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| SUBJECT | Wsubj | 5 | +0.1633 / +0.0707 | 2.311x | +0.446 | 0.00e+00 | 0.0010 | 0.703 |
| SUBJECT | Wall | 36 | +0.1822 / +0.1073 | 1.699x | +0.325 | 0.00e+00 | 0.0010 | 0.633 |
| LEVEL | Wlev | 7 | +0.2304 / +0.0411 | 5.608x | +0.684 | 0.00e+00 | 0.0010 | 0.577 |
| LEVEL | Wall | 36 | +0.1822 / +0.0431 | 4.230x | +0.597 | 0.00e+00 | 0.0010 | 0.574 |

**UNIT** (cohesion + silhouette + η²)

| window | n_layers | same / diff cos | ratio | d | sil(unit) | sil(level) | sil(subj) | sil(subj|residL) | sil(unit|residL) | η²_lev | η²_subj | η²_inter |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wsubj | 5 | +0.1633 / +0.0006 | 296.221x | +0.846 | -0.1031 | -0.0338 | -0.0115 | +0.0015 | -0.0868 | 0.090 | 0.060 | 0.023 |
| Wlev | 7 | +0.2304 / +0.0151 | 15.257x | +0.821 | -0.1425 | -0.0425 | -0.0350 | -0.0120 | -0.1047 | 0.163 | 0.045 | 0.022 |
| Wall | 36 | +0.1822 / +0.0120 | 15.233x | +0.786 | -0.1141 | -0.0354 | -0.0227 | -0.0051 | -0.0850 | 0.110 | 0.041 | 0.021 |

## §Y. 결론 (요약 가이드 — 수치 해석은 표 참조)
- §S(SUBJECT): `W_SUBJ` 에서의 효과크기/perm-p/LDA-F1 가 `W_ALL` 보다 *유리* 한지 비교.
- §L(LEVEL): `W_LEV` 에서의 효과크기/perm-p/LDA-F1 가 `W_ALL` 보다 *유리* 한지 비교.
- §U(UNIT): 세 윈도우에서 same-unit/diff-unit ratio, silhouette(unit), η²_level vs η²_subject 비교.
- level-residual 후 subject/unit silhouette 가 어떻게 변하는지로 'level 제거 시 잔여 구조'를 확인.

> 주의: η² 분해는 불균형 설계의 근사 (interaction 음수면 0 취급). 단일 행렬 결과 하나로 강결론 금지. permutation p · 효과크기 · pilot1/pilot2 일반화(LDA) 를 함께 본다.

**산출물.**
- `REPORT_bestlayer_resolved_2026-06-22.md` (이 파일)
- `bestlayer_artifacts.npz` (모든 raw 수치)
- `bestlayer_pairhist_{subject,level}_{Wsubj,Wlev,Wall}.png`, `bestlayer_unithist_{Wsubj,Wlev,Wall}.png` (옵션)