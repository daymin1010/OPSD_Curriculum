# Unsupervised Difficulty Axis + Length Gate (tag=unsupdiff)

> ⚠️ **정정 노트 (2026-06-13, 사용자 결정)**: 아래 "GATE = FAIL" 및 "GPT level FALLBACK" 권고는 **현 단계 의사결정에 반영하지 않고 보류**한다(철회 아님). 어려운 문제일수록 추론·서술이 길어지는 것은 자명하므로 활성화 난이도 신호를 "length 때문에 생긴 가짜"로 단정할 수 없다. 현재는 활성화 난이도 축을 정당한 난이도 신호로 사용한다. length confound 재검은 추후 필요 시 옵션. (맥락: `HANDOFF_CURRICULUM_SUBJECTxLEVEL_2026-06-13.md` §0) 아래 본문 수치는 기계 출력 원본으로 보존.

- pooled N = **3025** (pilot1=1608, pilot2=1417); subjects=8, levels=[1, 2, 3, 4, 5, 6, 7, 8]
- is_correct non-null = 3024 (overall rate=0.818)
- Method: difficulty 방향은 **unsupervised PCA(공분산만)**; GPT level 은 PC 선택·부호·평가에만 사용(학습 X) → circularity 회피. CPU only, THINKING ΔA.

## TASK 1 — Unsupervised difficulty axis (PCA, no GPT-label fitting)
- [raw] PCA top-5 fit (n=3025, F=442368); EVR=[0.41690000891685486, 0.09210000187158585, 0.05249999836087227, 0.035100001841783524, 0.03420000150799751] (4s)

| PC | EVR | ρ(level) | ρ(is_correct) | ρ(gen_len) |
|----|-----|----------|---------------|------------|
| PC1 | 0.4169 | +0.707 | -0.404 | +0.739 |
| PC2 | 0.0921 | +0.618 | +0.038 | +0.443 |
| PC3 | 0.0525 | +0.428 | -0.005 | +0.313 |
| PC4 | 0.0351 | +0.186 | -0.308 | +0.167 |
| PC5 | 0.0342 | +0.067 | -0.198 | +0.031 |
- **adopted PC = PC1** (|ρ(level)| max = +0.707).
- unit-centroid 1D ordering proxy: ρ(unit mean diff_score, unit level) = **+0.938** (units n≥10: 57). 양수 = unsupervised score 가 난이도 순서를 재현.

## (보조) Supervised ridge_level — 대조용 (partition 정의엔 안 씀)
- ridge_level (α=10000, pilot1 5-fold cv ρ=+0.941); pilot2 test: ρ(level)=+0.936, ρ(is_correct)=-0.286, ρ(gen_len)=+0.750

## TASK 2 — Length residualize gate ★

### (a) score-level residualize (adopted unsup difficulty)
- ρ(diff_score, level)  before = +0.707 → after gen_len residual = -0.166
- partial ρ(diff_score, level | gen_len) = **+0.386**
- partial ρ(diff_score, is_correct | gen_len) = -0.232

### (b) feature-level residualize (각 차원에서 gen_len 회귀 후 PCA 재fit)
- residualized features in place (1s)
- [residual] PCA top-5 fit (n=3025, F=442368); EVR=[0.3418000042438507, 0.08510000258684158, 0.05209999904036522, 0.04170000180602074, 0.039500001817941666] (5s)

| PC | EVR | ρ(level) | ρ(is_correct) | ρ(gen_len) |
|----|-----|----------|---------------|------------|
| PC1 | 0.3418 | +0.236 | +0.283 | +0.284 |
| PC2 | 0.0851 | +0.242 | +0.020 | +0.004 |
| PC3 | 0.0521 | +0.265 | +0.115 | +0.003 |
| PC4 | 0.0417 | +0.286 | -0.108 | +0.139 |
| PC5 | 0.0395 | +0.204 | -0.275 | +0.035 |
- **adopted PC = PC4** (|ρ(level)| max = +0.286).

### GATE 판정
- 기준: feature-level ρ(level) 잔차후 |+0.286| ≥ 0.5×|+0.707| 이고 ≥ 0.3; AND partial ρ(score,level|gen_len) |+0.386| ≥ 0.3
- **GATE = FAIL ⚠️** → difficulty 가 상당 부분 length proxy. unsupervised 난이도 주의(아래 fallback 참조).
- 주의: 난이도-길이는 본질적 비례일 수 있으므로 이 게이트는 confound 단정이 아니라 'gen_len 제거 후 잔존 난이도 신호' 확인용.

## TASK 3 — Subject 검증 (측정용, partition 아님)
- unsupervised PC-5 공간 silhouette: subject=-0.129 vs level=-0.114 (subject ≪ level 이면 unsupervised 공간에서 subject 약함 = 예상된 결과).
- supervised subject LDA (mid-layer L11-15, PCA→100d, pilot1 train / pilot2 test): macro-F1=**0.681**, weighted-F1=0.690, chance≈0.125 (8 subj).
- per-subject F1: Algebra=0.52, Counting & Probability=0.82, Geometry=0.80, Intermediate Algebra=0.55, Number Theory=0.80, Other=0.73, Prealgebra=0.60, Precalculus=0.61
- 해석: chance 보다 높으면 subject 정보는 존재하나(F1 작으면) representation 이 약함. 이는 검증·대조용이며 curriculum partition 정의가 아님(circularity 회피).

## TASK 4 — 대조 (unsup vs GPT level vs ridge) + fallback 판정

### pairwise Spearman ρ
| pair | ρ |
|------|---|
| unsup difficulty ↔ GPT level | +0.707 |
| ridge_level ↔ GPT level | +0.964 |
| unsup difficulty ↔ ridge_level | +0.743 |
| unsup difficulty ↔ gen_len | +0.739 |

### unsup difficulty 와 GPT level 의 rank 차 큰 sample top-10
| problem_id | subject | level | gen_len | unsup_rank | level_rank |
|---|---|---|---|---|---|
| 2a60900bf8da | Counting & Probability | 1 | 8192 | 2619 | 168 |
| 856a7d3c042a | Other | 1 | 8192 | 2592 | 168 |
| a1d966bb4c5d | Geometry | 1 | 2198 | 2587 | 168 |
| 029023296ee9 | Other | 1 | 2896 | 2540 | 168 |
| 39aa0f11d75a | Number Theory | 1 | 4388 | 2527 | 168 |
| aaa94f1298d2 | Other | 1 | 1063 | 2502 | 168 |
| 77aac4db819e | Counting & Probability | 1 | 2425 | 2429 | 168 |
| 349290bfa711 | Other | 1 | 6587 | 2410 | 168 |
| fad1613ffcb8 | Other | 1 | 2814 | 2380 | 168 |
| ebf1aae622a1 | Other | 1 | 1886 | 2337 | 168 |

### 안정성 (split-half by pilot)
- ρ(diff_score, level): pilot1=+0.707, pilot2=+0.707 → 안정 ✅ (|차|≤0.20 & 둘다≥0.3)

### Fallback 판정
- (a) length 게이트 = FAIL, (b) 안정성 = OK.
- **권고: GPT level FALLBACK** — unsupervised 난이도가 length proxy 거나 불안정. curriculum 난이도는 GPT level 사용 권장.
- (subject 는 원래 GPT mixing 이 기본 → fallback 대상 아님; TASK3 은 대조 측정용.)
