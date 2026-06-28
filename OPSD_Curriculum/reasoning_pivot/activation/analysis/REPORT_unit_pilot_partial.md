# ΔA Unit Analysis — pilot_partial

- N loaded: **1506**
- subject×level units: 58
- finish_reason: {'stop': 1204, 'length': 302}
- truncated: 302/1506
- correct rate: 0.809 (n=1505)

### counts by (subject, level)
```
level                    1   2   3   4   5   6   7   8
subject                                               
Algebra                 27  29  29  29  28  28  27   0
Counting & Probability  30  27  28  29  27  28  29  10
Geometry                28  28  28  30  27  29  30   8
Intermediate Algebra    16  29  27  28  26  28  27  11
Number Theory           26  29  30  27  29  27  29  21
Other                   22  26  29  29  29  29  28  11
Prealgebra              29  28  28  15   0   0   0   0
Precalculus             17  27  27  26  29  29  25   0
```

### truncation% by level (confound watch)
```
level
1    0.01
2    0.02
3    0.07
4    0.15
5    0.24
6    0.34
7    0.50
8    0.57
```

- faithful clean subset (finish=stop): 1204/1506

## [FAITHFUL] magnitude (per-layer |dA| L2)
- layer-avg |dA|: mean=406.72 std=2.81
- ρ(|dA|, level)        = +0.081
- ρ(|dA|, r1_cot_tokens)= +0.064
- ρ(|dA|, gen_len)      = +0.188
- partial ρ(|dA|, level | gen_len) = -0.080
- |dA| correct=406.47 (n=1218) vs incorrect=407.75 (n=287)

  |dA| by level:
  level
  1    405.880005
  2    406.190002
  3    406.200012
  4    406.529999
  5    406.880005
  6    407.390015
  7    407.670013
  8    408.109985

- best layer for ρ(|dA_layer|, level): L20 ρ=-0.586

## [FAITHFUL] global PCA of per-layer |dA| profile (36-d)
- PC var-explained: PC1=0.524 PC2=0.369 PC3=0.045 PC4=0.019 PC5=0.017
- ρ(PC1, level)  = +0.096
- ρ(PC1, gen_len)= +0.104
  (PC1이 gen_len과만 강상관 & level과 약하면 = 길이 교란 신호)

## [FAITHFUL(clean)] layerwise linear probe (full 12288-d → PCA50)
- probe N = 1204

### is_correct probe (chance bal-acc=0.50; majority=0.90)
- best layer L26: bal-acc=0.770
- mean over layers: 0.736
- curve(every4): L0:0.67 L4:0.73 L8:0.73 L12:0.74 L16:0.74 L20:0.75 L24:0.75 L28:0.76 L32:0.75

### level probe (Spearman of CV-pred vs level; chance≈0)
- best layer L21: ρ=+0.909
- mean |ρ| over layers: 0.884
- curve(every4): L0:+0.78 L4:+0.86 L8:+0.86 L12:+0.89 L16:+0.90 L20:+0.90 L24:+0.90 L28:+0.90 L32:+0.91

### subject probe (8 classes; majority=0.14)
- best layer L10: bal-acc=0.594
- mean over layers: 0.559
- curve(every4): L0:0.59 L4:0.59 L8:0.57 L12:0.59 L16:0.57 L20:0.55 L24:0.54 L28:0.54 L32:0.53

## [FAITHFUL] SELECTED UNIT view (unit=subject×level)
- units total=58, units with n≥5=58

### within- vs between-unit dispersion (cosine) per layer
  layer | within_mean_cos | between_mean_cos | sep(=bet-with, lower=more separable... ) 
  L0   | +0.955          | +0.986          | -0.031
  L6   | +1.000          | +1.000          | -0.000
  L12  | +0.865          | +0.953          | -0.089
  L18  | +0.805          | +0.959          | -0.154
  L24  | +0.800          | +0.960          | -0.160
  L30  | +0.810          | +0.915          | -0.105
  L35  | +0.997          | +0.999          | -0.002
  (within≫between 이면 unit이 응집/분리됨. 둘 다 ~1 이면 공통 성분 지배=신호 약함)

### correct−incorrect contrast consistency @ L18
- usable units=51; mean pairwise cos of (correct−incorrect) direction = +0.307 (std=0.374)
  (>0.2 이면 난이도 통제 후에도 일관된 'competence' 축 존재 시사)

## [THINKING] magnitude (per-layer |dA| L2)
- layer-avg |dA|: mean=69.28 std=5.56
- ρ(|dA|, level)        = -0.371
- ρ(|dA|, r1_cot_tokens)= -0.395
- ρ(|dA|, gen_len)      = -0.492
- partial ρ(|dA|, level | gen_len) = -0.028
- |dA| correct=70.57 (n=1218) vs incorrect=63.78 (n=287)

  |dA| by level:
  level
  1    71.570000
  2    71.489998
  3    70.760002
  4    69.769997
  5    68.849998
  6    67.970001
  7    65.599998
  8    64.110001

- best layer for ρ(|dA_layer|, level): L28 ρ=-0.771

## [THINKING] global PCA of per-layer |dA| profile (36-d)
- PC var-explained: PC1=0.752 PC2=0.201 PC3=0.019 PC4=0.012 PC5=0.007
- ρ(PC1, level)  = -0.034
- ρ(PC1, gen_len)= +0.154
  (PC1이 gen_len과만 강상관 & level과 약하면 = 길이 교란 신호)

## [THINKING(all)] layerwise linear probe (full 12288-d → PCA50)
- probe N = 1506

### is_correct probe (chance bal-acc=0.50; majority=0.81)
- best layer L17: bal-acc=0.834
- mean over layers: 0.801
- curve(every4): L0:0.73 L4:0.75 L8:0.80 L12:0.80 L16:0.82 L20:0.81 L24:0.82 L28:0.81 L32:0.81

### level probe (Spearman of CV-pred vs level; chance≈0)
- best layer L27: ρ=+0.939
- mean |ρ| over layers: 0.881
- curve(every4): L0:+0.64 L4:+0.78 L8:+0.85 L12:+0.90 L16:+0.92 L20:+0.93 L24:+0.94 L28:+0.94 L32:+0.93

### subject probe (8 classes; majority=0.14)
- best layer L9: bal-acc=0.631
- mean over layers: 0.553
- curve(every4): L0:0.35 L4:0.44 L8:0.58 L12:0.62 L16:0.59 L20:0.57 L24:0.58 L28:0.58 L32:0.57

## [THINKING] SELECTED UNIT view (unit=subject×level)
- units total=58, units with n≥5=58

### within- vs between-unit dispersion (cosine) per layer
  layer | within_mean_cos | between_mean_cos | sep(=bet-with, lower=more separable... ) 
  L0   | +0.878          | +0.958          | -0.079
  L6   | +0.821          | +0.910          | -0.089
  L12  | +0.870          | +0.921          | -0.051
  L18  | +0.865          | +0.913          | -0.048
  L24  | +0.898          | +0.932          | -0.035
  L30  | +0.891          | +0.912          | -0.021
  L35  | +0.910          | +0.947          | -0.037
  (within≫between 이면 unit이 응집/분리됨. 둘 다 ~1 이면 공통 성분 지배=신호 약함)

### correct−incorrect contrast consistency @ L18
- usable units=51; mean pairwise cos of (correct−incorrect) direction = +0.311 (std=0.234)
  (>0.2 이면 난이도 통제 후에도 일관된 'competence' 축 존재 시사)

## SIGNAL VERDICT (heuristic)
- [FAITHFUL(clean)] level probe best |ρ|=0.909 (STRONG)
- [FAITHFUL(clean)] is_correct probe best bal-acc=0.770 (STRONG)
- [THINKING(all)] level probe best |ρ|=0.939 (STRONG)
- [THINKING(all)] is_correct probe best bal-acc=0.834 (STRONG)

→ 모든 probe가 WEAK/~chance 이고 PC1이 길이 교란이면: 방향 전환 권고 (token 위치 재선정: A_t1_think / A_tK_think / A_prompt_last 대비, 혹은 thinking-vs-nonthinking 추출).
---

## ⚠️ CAVEAT & NEXT STEP (added post-run)
- level probe |ρ|≈0.91–0.94 는 **의심스럽게 높음**. magnitude-only ρ(|dA|,level)=0.08 (faithful)
  와 큰 괴리 → full-dim probe 가 **gen_len / 길이 정보**를 읽고 있을 강한 가능성.
  (level ↔ r1_cot ρ≈0.61 로 길이와 교란.)
- 따라서 현재 STRONG verdict 는 **길이 통제 전 잠정치**. 다음 검증 필요:
  1. gen_len 을 PCA 입력에서 partial-out (또는 길이 매칭 서브샘플) 후 level/is_correct probe 재측정.
  2. is_correct probe(0.77/0.83)는 정답률이 길이와 약상관이라 상대적으로 더 신뢰 가능 —
     이쪽이 'competence 축' 핵심 근거. contrast consistency(+0.311)와 함께 해석.
- 결론: **신호 존재는 확인(특히 is_correct + contrast)**, 단 level 축은 길이 통제 후 재확인 단계로 넘김.
