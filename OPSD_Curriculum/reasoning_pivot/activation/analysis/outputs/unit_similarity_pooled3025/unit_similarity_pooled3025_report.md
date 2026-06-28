# POOLED Qwen3-8B Unit Prototype Similarity (raw=과거 / resid=현재)

**Data:** OPSD reasoning_pivot pilot1+pilot2 (THINKING ΔA = dA_thinking), model = **Qwen/Qwen3-8B**, pooled finite N = **3025** (raw loaded 3025, non-finite dropped 0).

## Method (REF: 4.6_Task2 nait_unit_similarity; mechanism unchanged)
- UNIT = subject × level.
- v_u^l = sign-calibrated PC1 of {ΔA_s^l : s∈u} (unit-norm, Eq.4).
- Per-layer sim:  S_l[u,u'] = v_u^l · v_{u'}^l (= cos).
- Aggregated:     S_agg[u,u'] = mean_l v_u^l · v_{u'}^l.
- **raw (과거)**: prototypes on raw ΔA.  **resid (현재)**: after removing per-layer global PC1.
- PC1 via torch.pca_lowrank(q=6).
- U=49, L=36, N=3025, D=12288, MIN_N=10, excludeOther.

## Clustering quantification (subject vs level)
| version | silhouette(subject) | silhouette(level) | cophenetic |
|---|---|---|---|
| raw   | -0.160313680768013 | -0.09478555619716644 | 0.929 |
| resid | -0.06437492370605469 | -0.03595835343003273 | 0.852 |

Higher silhouette under a label ⇒ units cluster primarily by that label.

## raw ΔA (과거)

- **Subject block** (avg cos): within = **0.385**, across = **0.388**, ratio = **0.99x**
- **Level   block** (avg cos): within = **0.581**, across = **0.397**, ratio = **1.46x**
- **Silhouette** (units on 1−S_agg): by subject = **-0.160313680768013**, by level = **-0.09478555619716644**  (higher ⇒ that label explains clustering)
- **Cophenetic corr** (avg-linkage dendrogram fidelity) = **0.929**

### Top 15 most similar unit pairs (aggregated cos)

| cos | u1 | u2 | same subj? | same lvl? |
|---|---|---|---|---|
| 0.889 | IA_L6 | IA_L7 | ✓ | · |
| 0.885 | C&P_L6 | NT_L6 | · | ✓ |
| 0.883 | C&P_L5 | C&P_L6 | ✓ | · |
| 0.878 | C&P_L5 | NT_L5 | · | ✓ |
| 0.878 | C&P_L5 | NT_L6 | · | · |
| 0.876 | Algebra_L7 | Pcalc_L6 | · | · |
| 0.872 | Algebra_L6 | IA_L6 | · | ✓ |
| 0.868 | C&P_L4 | NT_L5 | · | · |
| 0.868 | Pcalc_L4 | Pcalc_L5 | ✓ | · |
| 0.867 | NT_L6 | Pcalc_L6 | · | ✓ |
| 0.866 | Algebra_L7 | C&P_L6 | · | · |
| 0.863 | C&P_L7 | IA_L6 | · | · |
| 0.863 | IA_L6 | NT_L7 | · | · |
| 0.862 | C&P_L6 | Pcalc_L6 | · | ✓ |
| 0.862 | Algebra_L7 | NT_L5 | · | · |

### Bottom 10 most DIS-similar pairs

| cos | u1 | u2 |
|---|---|---|
| -0.095 | IA_L4 | NT_L1 |
| -0.058 | Pcalc_L1 | Pcalc_L3 |
| -0.046 | Pcalc_L2 | Pcalc_L3 |
| -0.033 | Geom_L3 | Pcalc_L6 |
| -0.031 | Geom_L3 | Geom_L5 |
| -0.024 | Algebra_L1 | Pcalc_L6 |
| -0.021 | Algebra_L1 | Geom_L5 |
| -0.019 | Pcalc_L2 | Pcalc_L6 |
| -0.018 | Pcalc_L2 | Pcalc_L5 |
| -0.016 | NT_L3 | Pcalc_L6 |

## resid ΔA (현재)

- **Subject block** (avg cos): within = **0.188**, across = **0.185**, ratio = **1.02x**
- **Level   block** (avg cos): within = **0.320**, across = **0.174**, ratio = **1.84x**
- **Silhouette** (units on 1−S_agg): by subject = **-0.06437492370605469**, by level = **-0.03595835343003273**  (higher ⇒ that label explains clustering)
- **Cophenetic corr** (avg-linkage dendrogram fidelity) = **0.852**

### Top 15 most similar unit pairs (aggregated cos)

| cos | u1 | u2 | same subj? | same lvl? |
|---|---|---|---|---|
| 0.724 | IA_L3 | IA_L4 | ✓ | · |
| 0.715 | Algebra_L6 | IA_L7 | · | · |
| 0.715 | IA_L7 | Pcalc_L6 | · | · |
| 0.691 | Geom_L7 | IA_L7 | · | ✓ |
| 0.684 | Geom_L7 | Pcalc_L6 | · | · |
| 0.679 | Algebra_L6 | Pcalc_L6 | · | ✓ |
| 0.668 | Algebra_L6 | Geom_L7 | · | · |
| 0.665 | Algebra_L7 | IA_L7 | · | ✓ |
| 0.656 | Algebra_L7 | Geom_L7 | · | ✓ |
| 0.637 | Algebra_L7 | Pcalc_L6 | · | · |
| 0.637 | Geom_L5 | Pcalc_L6 | · | · |
| 0.627 | Geom_L4 | Pcalc_L4 | · | ✓ |
| 0.625 | NT_L6 | Pcalc_L6 | · | ✓ |
| 0.623 | IA_L7 | NT_L5 | · | · |
| 0.619 | Algebra_L6 | Algebra_L7 | ✓ | · |

### Bottom 10 most DIS-similar pairs

| cos | u1 | u2 |
|---|---|---|
| -0.212 | IA_L2 | Pcalc_L6 |
| -0.206 | IA_L2 | NT_L8 |
| -0.172 | Geom_L5 | IA_L2 |
| -0.165 | IA_L2 | IA_L7 |
| -0.162 | Algebra_L1 | Geom_L4 |
| -0.155 | Geom_L7 | IA_L2 |
| -0.147 | Algebra_L1 | Geom_L5 |
| -0.144 | IA_L2 | NT_L5 |
| -0.140 | Algebra_L6 | IA_L2 |
| -0.134 | Algebra_L1 | Pcalc_L6 |

## Figures
- `unit_sim_layer_{raw,resid}_L*.png`
- `unit_sim_agg_{raw,resid}.png`
- `unit_sim_block_subject_{raw,resid}.png`
- `unit_sim_block_level_{raw,resid}.png`
- `unit_sim_dendrogram_{raw,resid}.png`

---

## [추가 분석] raw Top 15: same-subject vs same-level 분류 (2026-06-22)

**채택 버전: raw ΔA** (resid는 global PC1 제거 후 수치이므로 해석 시 주의 필요)

### 같은 subject, 다른 level (✓ ·) — raw Top 15 내

| cos | u1 | u2 | level 차이 |
|---|---|---|---|
| 0.889 | **IA**_L6 | **IA**_L7 | 1 |
| 0.883 | **C&P**_L5 | **C&P**_L6 | 1 |
| 0.868 | **Pcalc**_L4 | **Pcalc**_L5 | 1 |

→ 패턴: **인접 level(차이=1)끼리만** 상위 등장. level 차이 ≥2인 같은 subject 쌍은 Top 15에 없음.

### 같은 level, 다른 subject (· ✓) — raw Top 15 내

| cos | u1 | u2 | level |
|---|---|---|---|
| 0.885 | C&P_**L6** | NT_**L6** | L6 |
| 0.878 | C&P_**L5** | NT_**L5** | L5 |
| 0.872 | Algebra_**L6** | IA_**L6** | L6 |
| 0.867 | NT_**L6** | Pcalc_**L6** | L6 |
| 0.862 | C&P_**L6** | Pcalc_**L6** | L6 |

→ 패턴: **L5~L6 고난이도**에 집중. C&P↔NT, Algebra↔IA, NT/C&P↔Pcalc 조합.

### 요약 카운트 (raw Top 15)

| 카테고리 | 쌍 수 |
|---|---|
| 같은 subject, 다른 level (✓ ·) | **3** |
| 같은 level, 다른 subject (· ✓) | **5** |
| 둘 다 다름 (· ·) | **7** |
| 둘 다 같음 (✓ ✓) | 0 |

### 커리큘럼 설계 시사점 (raw 기준)

1. **Level이 subject보다 activation 구조를 더 강하게 결정** — "같은 level × 다른 subject"가 "같은 subject × 다른 level"보다 상위 쌍에 더 많이 등장 (5 vs 3).
2. **같은 subject 내에서는 인접 level 간(차이=1)만 높은 유사도** → 같은 과목이라도 난이도가 2 이상 벌어지면 activation 패턴이 달라짐.
3. **L5~L6에서 cross-subject 수렴** → 고난이도 구간에서 과목 경계가 약해짐(상위 유사쌍이 L5–L6에 집중). 커리큘럼 후반 stage에서 과목 혼합이 가능.
   - ⚠️ 단, 이는 "고립=고난도"를 의미하지 **않는다**. (정정 항목 참조)
4. **권장 커리큘럼 축**: difficulty(난이도) 축 우선 → 그 안에서 subject 그루핑. 순수 subject stage 방식보다 level-first 방식이 activation 구조에 align됨.

### ⚠️ 정정 (2026-06-22): "고립=고난도"는 데이터와 반대 — 실제로는 "고립=저난도"

이전 서술에서 "수렴=고난도"를 "고립=고난도"로 확대해석할 여지가 있었으나, **most-DIS-similar(가장 고립된) 쌍**의 데이터는 정반대를 가리킨다:

- **raw Bottom 10** 의 고립 단위는 거의 전부 **저난도(L1–L3)**: `IA_L4·NT_L1`, `Pcalc_L1·L2·L3`, `Geom_L3·L5`, `Algebra_L1`.
- **resid Bottom 10** 도 마찬가지로 **저난도** 지배: `IA_L2`(10쌍 중 6쌍), `Algebra_L1`(3쌍), `Geom_L4/L5`, `Pcalc`(저~중level).

즉 활성 공간에서 **가장 동떨어진(고립된) unit 은 저난도 과목(Pcalc/Geom/Algebra/IA 의 L1–L2)** 이며, 고난도가 아니다.
종합하면: **고난도(L5–L6) = 과목 간 수렴(경계 약화)**, **저난도(L1–L2) = 과목 간 분기·고립**. 이는 §3의 수렴 서술과 모순되지 않으며, "고립=고난도" 표현만 잘못이었으므로 위와 같이 바로잡는다.

