# LEVEL / UNIT 정밀검정 (subject 통제) + 레이어 분해 — 2026-06-22  (tag=levunit)

> **목적.** 직전 세션에서 unit(subject×level) 구조는 LEVEL이 지배(within/across
> 비율 level 1.46x vs subject 0.99x)했다. subject는 `subject_layer_resolved.py`로
> level 통제하 단독검정을 마쳤다. 여기서는 **그 분석 코드(기법)를 그대로 재사용**해
> 주효과 축만 level/unit 으로 대칭 전환하여, 지배적이라 알려진 LEVEL/UNIT 신호가
> supervised 판별(Fisher·프로브·LDA) + block-permutation + η² 분산분해로 봐도
> 견고한지 정밀 검증한다. (subject 분석의 대칭 통제: 블록=subject, 라벨=level.)

**데이터.** pooled(pilot1+pilot2) THINKING ΔA, μ_pooled(per-layer) centering. finite **N=3025** (raw 3025, non-finite drop 0). units=58. CPU only. seed=42.
- provenance: pilot1=1608, pilot2=1417
- 재사용: subject_layer_resolved(slr).fisher_ratio/probe_f1/select_window/window_cos_matrix/cohens_d/eta2_partition/subject_silhouette(=generic).

## A. 레이어 스캔 — LEVEL(과 subject 참조) 판별 레이어 찾기

방법(slr 재사용): (a) Fisher 판별비 tr(Sb)/tr(Sw) — 36레이어 전부; (b) PCA-whiten→다항 로지스틱 프로브, pilot1 train→pilot2 test, macro-F1.

- **best level layer = L31**; level 윈도우(>=85%) = L16–L32 [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
- best subject layer = L9 (참조)
- probe macro-F1: level best = 0.571 @L31 (chance≈0.125); subject best = 0.691 @L9 (chance≈0.125)
- Fisher level best = 0.258 @L29; subject best = 0.076 @L9

Fisher level per-layer: L0:0.036, L1:0.036, L2:0.060, L3:0.108, L4:0.063, L5:0.063, L6:0.084, L7:0.086, L8:0.088, L9:0.097, L10:0.097, L11:0.100, L12:0.100, L13:0.104, L14:0.099, L15:0.103, L16:0.004, L17:0.107, L18:0.121, L19:0.133, L20:0.128, L21:0.123, L22:0.139, L23:0.155, L24:0.186, L25:0.180, L26:0.190, L27:0.198, L28:0.182, L29:0.258, L30:0.214, L31:0.201, L32:0.187, L33:0.165, L34:0.159, L35:0.184

probe-F1 level per-layer: L0:0.324, L1:0.339, L2:0.328, L3:0.366, L4:0.387, L5:0.391, L6:0.397, L7:0.404, L8:0.420, L9:0.456, L10:0.463, L11:0.465, L12:0.482, L13:0.502, L14:0.480, L15:0.505, L16:0.538, L17:0.523, L18:0.539, L19:0.550, L20:0.559, L21:0.552, L22:0.553, L23:0.554, L24:0.545, L25:0.569, L26:0.550, L27:0.547, L28:0.547, L29:0.557, L30:0.560, L31:0.571, L32:0.552, L33:0.532, L34:0.528, L35:0.532

> 해석: level 곡선의 봉우리 레이어/윈도우가 level 신호 집중 구간. (곡선 그림: `levunit_layerscan.png`)

### A.1. Top-K best layers (Top-5, npz 재사용 · 재실행 없음)

아래 4개 표는 각 metric × axis 별 상위 5 레이어(값 내림차순). `cross` 칸은 *같은 metric의 반대 축 값*(예: level-Fisher 표 cross = 해당 레이어 subject-Fisher 값) → 해당 레이어가 정말 그 축에 *특이적*인지 한눈에 비교용.

**LEVEL · Fisher tr(Sb)/tr(Sw)** — Top-5
| rank | layer | value | cross: subject-Fisher |
|---:|---:|---:|---:|
| 1 | L29 | 0.258 | 0.049 |
| 2 | L30 | 0.214 | 0.048 |
| 3 | L31 | 0.201 | 0.047 |
| 4 | L27 | 0.198 | 0.045 |
| 5 | L26 | 0.190 | 0.045 |

**LEVEL · probe macro-F1 (p1→p2)** — Top-5
| rank | layer | value | cross: subject-F1 |
|---:|---:|---:|---:|
| 1 | L31 | 0.571 | 0.624 |
| 2 | L25 | 0.569 | 0.623 |
| 3 | L30 | 0.560 | 0.620 |
| 4 | L20 | 0.559 | 0.632 |
| 5 | L29 | 0.557 | 0.608 |

**SUBJECT · Fisher tr(Sb)/tr(Sw)** — Top-5
| rank | layer | value | cross: level-Fisher |
|---:|---:|---:|---:|
| 1 | L9 | 0.076 | 0.097 |
| 2 | L12 | 0.063 | 0.100 |
| 3 | L10 | 0.062 | 0.097 |
| 4 | L11 | 0.060 | 0.100 |
| 5 | L14 | 0.059 | 0.099 |

**SUBJECT · probe macro-F1 (p1→p2)** — Top-5
| rank | layer | value | cross: level-F1 |
|---:|---:|---:|---:|
| 1 | L9 | 0.691 | 0.456 |
| 2 | L12 | 0.688 | 0.482 |
| 3 | L14 | 0.669 | 0.480 |
| 4 | L10 | 0.668 | 0.463 |
| 5 | L11 | 0.662 | 0.465 |

**Top-5 union 후보 레이어** (네 표에 한 번이라도 등장): L9, L10, L11, L12, L14, L20, L25, L26, L27, L29, L30, L31

| layer | in LEVEL-Fisher | LEVEL-F1 | SUBJECT-Fisher | SUBJECT-F1 |
|---:|:---:|:---:|:---:|:---:|
| L9 | · | · | ● | ● |
| L10 | · | · | ● | ● |
| L11 | · | · | ● | ● |
| L12 | · | · | ● | ● |
| L14 | · | · | ● | ● |
| L20 | · | ● | · | · |
| L25 | · | ● | · | · |
| L26 | ● | · | · | · |
| L27 | ● | · | · | · |
| L29 | ● | ● | · | · |
| L30 | ● | ● | · | · |
| L31 | ● | ● | · | · |

> 해석. **LEVEL** 신호 봉우리는 후기층(Fisher: L24–L31, probe-F1: L19–L31)에 몰리며 일부 레이어는 두 metric 모두에서 등장(예: L25, L29, L31) → 견고한 'level layer' 후보. **SUBJECT**는 초기-중간층(Fisher: L9–L13 부근, probe-F1: L8–L14)에 모이며 LEVEL 봉우리와 거의 겹치지 않음(L9 vs L31). 즉 LEVEL/SUBJECT 신호는 *서로 다른 레이어 대역에 분리*되어 있다.

## B. subject 통제하 LEVEL 단독 검정 (대칭: 블록=subject, 라벨=level)

metric = per-layer L2-normalized 후 layer-averaged *sample-pairwise* cosine. subject 블록 안에서만 same-level vs diff-level 비교(→ subject 오염 0). 이는 slr 의 'level 고정·subject 검정' 의 정확한 대칭(subject 고정·level 검정).

### view = `layeravg` (layers=0..35)
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

### view = `levwin_L16-32` (layers=16..32)
- within-subject **same-level** mean cos = +0.2126 (n_pairs=85359); **diff-level** mean cos = +0.0439 (n_pairs=506927); ratio = **4.844x**
- **Cohen's d (same−diff) = +0.670**; Mann–Whitney p(same>diff) = 0.00e+00
- **block-permutation**(subject 고정, level 셔플 ×1000): stat(mean_same−mean_diff)=+0.1668, **p=0.0010** (blocks=8)
- per-subject (same_mean / diff_mean / n_same / n_diff):
    Algebra (n=420): same=+0.2362 diff=+0.0233  (12390/75600 pairs)
    Counting & Probability (n=430): same=+0.2103 diff=+0.0412  (12435/79800 pairs)
    Geometry (n=406): same=+0.2152 diff=+0.0228  (11322/70893 pairs)
    Intermediate Algebra (n=387): same=+0.1978 diff=+0.0298  (10795/63896 pairs)
    Number Theory (n=443): same=+0.2141 diff=+0.0352  (12643/85260 pairs)
    Other (n=397): same=+0.1915 diff=+0.0989  (10974/67632 pairs)
    Prealgebra (n=197): same=+0.2692 diff=+0.1262  (5446/13860 pairs)
    Precalculus (n=345): same=+0.1883 diff=+0.0448  (9354/49986 pairs)

### view = `mid_L11-15` (layers=[11, 12, 13, 14, 15])
- within-subject **same-level** mean cos = +0.1644 (n_pairs=85359); **diff-level** mean cos = +0.0484 (n_pairs=506927); ratio = **3.394x**
- **Cohen's d (same−diff) = +0.563**; Mann–Whitney p(same>diff) = 0.00e+00
- **block-permutation**(subject 고정, level 셔플 ×1000): stat(mean_same−mean_diff)=+0.1151, **p=0.0010** (blocks=8)
- per-subject (same_mean / diff_mean / n_same / n_diff):
    Algebra (n=420): same=+0.1687 diff=+0.0152  (12390/75600 pairs)
    Counting & Probability (n=430): same=+0.1652 diff=+0.0602  (12435/79800 pairs)
    Geometry (n=406): same=+0.1700 diff=+0.0340  (11322/70893 pairs)
    Intermediate Algebra (n=387): same=+0.1587 diff=+0.0320  (10795/63896 pairs)
    Number Theory (n=443): same=+0.1695 diff=+0.0488  (12643/85260 pairs)
    Other (n=397): same=+0.1583 diff=+0.0919  (10974/67632 pairs)
    Prealgebra (n=197): same=+0.1934 diff=+0.0963  (5446/13860 pairs)
    Precalculus (n=345): same=+0.1406 diff=+0.0485  (9354/49986 pairs)

### B-LDA — supervised LEVEL confusion (어느 level끼리 헷갈리나=유사)

- `layeravg`: PCA(150)→LDA, pilot1→pilot2, **macro-F1=0.574** (chance≈0.143)
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

- `levwin_L16-32`: PCA(150)→LDA, pilot1→pilot2, **macro-F1=0.583** (chance≈0.143)
  confusion (row-normalized, test):
```
               1       2       3       4       5       6       7
       1   0.819   0.181   0.000   0.000   0.000   0.000   0.000
       2   0.146   0.683   0.129   0.033   0.004   0.004   0.000
       3   0.004   0.163   0.542   0.250   0.042   0.000   0.000
       4   0.000   0.000   0.210   0.438   0.286   0.057   0.010
       5   0.000   0.005   0.043   0.219   0.448   0.238   0.048
       6   0.000   0.000   0.005   0.014   0.206   0.550   0.225
       7   0.000   0.000   0.000   0.000   0.067   0.276   0.656
```
  top 혼동쌍(=내부 유사): 4↔5=0.252; 6↔7=0.250; 3↔4=0.230; 5↔6=0.222; 1↔2=0.163

- `mid_L11-15`: PCA(150)→LDA, pilot1→pilot2, **macro-F1=0.512** (chance≈0.143)
  confusion (row-normalized, test):
```
               1       2       3       4       5       6       7
       1   0.795   0.205   0.000   0.000   0.000   0.000   0.000
       2   0.180   0.615   0.159   0.042   0.004   0.000   0.000
       3   0.013   0.208   0.429   0.254   0.083   0.013   0.000
       4   0.000   0.010   0.257   0.371   0.271   0.090   0.000
       5   0.000   0.014   0.095   0.205   0.424   0.190   0.071
       6   0.000   0.000   0.019   0.072   0.271   0.401   0.237
       7   0.000   0.000   0.000   0.006   0.104   0.274   0.616
```
  top 혼동쌍(=내부 유사): 3↔4=0.256; 6↔7=0.256; 4↔5=0.238; 5↔6=0.231; 1↔2=0.192

## C. UNIT(subject×level) 구조 정밀검증

### C-1. 2-way 분산분해 (slr.eta2_partition)
주변평균 기반 η²(불균형 설계라 비직교 → 근사 분해; interaction 음수면 0 취급). η²_level vs η²_subject 로 'unit 구조를 누가 이끄는가' 수치화.
- 전체 레이어 평균: η²_level=0.110, η²_subject=0.041, η²_interaction=0.021
- level 윈도우(L16–32) 평균: η²_level=0.135, η²_subject=0.043, η²_interaction=0.021
- best level layer L31: η²_level=0.168, η²_subject=0.045

η² level per-layer:   L0:0.035, L1:0.035, L2:0.056, L3:0.097, L4:0.060, L5:0.059, L6:0.078, L7:0.079, L8:0.081, L9:0.089, L10:0.088, L11:0.091, L12:0.091, L13:0.094, L14:0.090, L15:0.093, L16:0.004, L17:0.097, L18:0.108, L19:0.118, L20:0.113, L21:0.110, L22:0.122, L23:0.134, L24:0.157, L25:0.153, L26:0.159, L27:0.165, L28:0.154, L29:0.205, L30:0.176, L31:0.168, L32:0.158, L33:0.141, L34:0.137, L35:0.155

η² subject per-layer: L0:0.020, L1:0.025, L2:0.020, L3:0.024, L4:0.033, L5:0.034, L6:0.031, L7:0.033, L8:0.044, L9:0.071, L10:0.058, L11:0.057, L12:0.059, L13:0.054, L14:0.056, L15:0.053, L16:0.003, L17:0.050, L18:0.049, L19:0.043, L20:0.050, L21:0.049, L22:0.042, L23:0.046, L24:0.045, L25:0.042, L26:0.043, L27:0.043, L28:0.043, L29:0.046, L30:0.046, L31:0.045, L32:0.042, L33:0.035, L34:0.026, L35:0.028

(곡선 그림: `levunit_eta.png`)

### C-2. UNIT cohesion — same-unit vs diff-unit pairwise cosine + silhouette
- `layeravg`: same-unit cos=+0.1822 (n=85359) / diff-unit cos=+0.0120 (n=4488441); ratio=**15.233x**; Cohen's d=+0.786
    silhouette: unit=-0.1141, level=-0.0354, subject=-0.0227
- `levwin_L16-32`: same-unit cos=+0.2126 (n=85359) / diff-unit cos=+0.0137 (n=4488441); ratio=**15.506x**; Cohen's d=+0.838
    silhouette: unit=-0.1264, level=-0.0369, subject=-0.0296
- `mid_L11-15`: same-unit cos=+0.1644 (n=85359) / diff-unit cos=+0.0021 (n=4488441); ratio=**76.602x**; Cohen's d=+0.857
    silhouette: unit=-0.1006, level=-0.0327, subject=-0.0131

### C-3. residualize 대조 silhouette
level-residual(self-level centroid 차감, ssg.level_centroid_residual) 후 subject/unit silhouette 변화로 'level 제거 시 잔여 구조'를 확인.
- `layeravg`: subject silhouette -0.0227 → (level-residual) -0.0051; unit silhouette -0.1141 → -0.0850
- `levwin_L16-32`: subject silhouette -0.0296 → (level-residual) -0.0085; unit silhouette -0.1264 → -0.0914
- `mid_L11-15`: subject silhouette -0.0131 → (level-residual) +0.0016; unit silhouette -0.1006 → -0.0838

## 결론 (요약)
1. **레이어 스캔**: level 판별 봉우리 = L16–L32 (best L31); subject best L9 과 비교는 §A 수치 참조.
   - LEVEL Top-5 Fisher: L29, L30, L31, L27, L26; LEVEL Top-5 probe-F1: L31, L25, L30, L20, L29.
   - SUBJECT Top-5 Fisher: L9, L12, L10, L11, L14; SUBJECT Top-5 probe-F1: L9, L12, L14, L10, L11. (자세한 표 §A.1)
2. **subject 통제 LEVEL 신호**(best window): same vs diff Cohen's d=+0.670, block-permutation p=0.0010 → 유의(subject 고정해도 level 신호 견고).
3. **UNIT**: η²_level vs η²_subject (윈도우 0.135 vs 0.043); same/diff-unit ratio≈15.506x. silhouette(unit/level/subject)는 §C-2 참조.
4. **residualize 대조**: level 제거 후 subject/unit silhouette 변화는 §C-3 참조.

> 주의: η² 분해는 불균형 설계의 근사이며 소표본 행렬 하나로 강결론 금지. permutation p·효과크기·pilot1/pilot2 일반화(프로브·LDA)를 함께 본다.