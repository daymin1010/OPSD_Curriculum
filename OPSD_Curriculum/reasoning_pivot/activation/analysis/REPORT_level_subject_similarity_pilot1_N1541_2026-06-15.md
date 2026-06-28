# LEVEL·SUBJECT 활성-이동(ΔA) 유사도 — pilot1 (N=1541)

작성: 2026-06-15 / 트랙: reasoning_pivot, THINKING ΔA
출처(원본 행렬): `REPORT_similarity_pilot.md` (pilot1 단독, N=1541)
관련: pooled 3025 버전은 별도 보고서 `REPORT_level_subject_similarity_pooled_N3025_2026-06-15.md` 참조.

> ⚠️ **표본 범위 주의**: 이 보고서의 8×8 행렬 수치는 **pilot1 단독 N=1541** 기준이다. canonical pooled(3025)와는 표본이 다르며, pooled 요약 통계(gap/ordinality)는 본문에 병기하되 **행렬 셀 값 자체는 pilot1**임을 명시한다.

---

## 0. 방법 요약
- 표현: 각 문제 i → ΔA_i ∈ R^(36 layer × 12288). 유사도는 **layer-averaged cosine**(레이어별 12288-d를 L2 정규화 후 코사인, 36개 레이어 평균)으로 계산.
- **centering(공통성분 제거)**: ΔA는 지배적 공통방향이 있어 raw 코사인은 거의 1로 무의미. 따라서 전역평균 μ=mean_i(ΔA_i)를 per-layer로 빼고(=centered) centroid를 계산 — 이것이 primary 결과.
- 그룹 centroid C_g[l] = 그룹 g 멤버들의 평균. 유사도 행렬 S[g,h] = mean_l cos(C_g[l], C_h[l]).
- 분리도 gap = within_mean − between_mean (클수록 그룹 내 응집·그룹 간 분리). 유의성은 라벨 permutation(N_PERM=200).
- LEVEL ordinality: 레벨이 순서형이므로 S[a,b](비대각)와 −|a−b| 간 Spearman ρ. 양수면 인접 레벨이 더 닮음.

---

## 1. LEVEL 유사도 — 커리큘럼의 1차 축 (강함)

**분리도**: within_mean = +0.279 | between_mean = −0.154 | **gap = +0.433**, perm-p = 0.005.
**ordinality**: ρ(S, −|Δlevel|) = **+0.849** → 인접 레벨일수록 강하게 닮음(연속·단조적 난이도 축).

### 1.1 centroid-cosine 행렬 (THINKING, centered, pilot1 N=1541)
```
        L1      L2      L3      L4      L5      L6      L7      L8
 L1   1.000   0.904   0.572  -0.267  -0.883  -0.929  -0.803  -0.677
 L2   0.904   1.000   0.783  -0.043  -0.801  -0.950  -0.906  -0.829
 L3   0.572   0.783   1.000   0.429  -0.425  -0.698  -0.905  -0.883
 L4  -0.267  -0.043   0.429   1.000   0.413   0.104  -0.263  -0.370
 L5  -0.883  -0.801  -0.425   0.413   1.000   0.851   0.585   0.433
 L6  -0.929  -0.950  -0.698   0.104   0.851   1.000   0.829   0.719
 L7  -0.803  -0.906  -0.905  -0.263   0.585   0.829   1.000   0.921
 L8  -0.677  -0.829  -0.883  -0.370   0.433   0.719   0.921   1.000
```

### 1.2 해석
1. **인접 레벨일수록 유사(순차성).** 바로 옆 레벨 코사인이 가장 큼:
   L1–L2 **+0.904**, L2–L3 +0.783, L5–L6 +0.851, L6–L7 +0.829, L7–L8 **+0.921**.
   거리가 멀어질수록 단조 감소 → ordinality ρ=+0.849.
2. **2-블록 구조.** 쉬움 묶음 {L1, L2, L3}은 자기들끼리 강한 양(+), 어려움 묶음 {L5, L6, L7, L8}도 자기들끼리 강한 양(+).
   **두 블록 사이는 강한 음(−0.8 ~ −0.95)** — 쉬운 문제와 어려운 문제의 활성-이동 방향이 거의 정반대(부호 뒤집힘). 이것이 difficulty 축.
3. **L4 = 경계/전이점(pivot).** L4는 L3(+0.429)·L5(+0.413) 양쪽에만 약하게 붙고 나머지 레벨엔 ~0 또는 음. 어느 블록에도 속하지 않는 전이 구간.
4. **함의**: 활성화 공간에 쉬움→어려움으로 이어지는 연속적·단조적 난이도 축이 실재. 커리큘럼의 staging 1차 축으로 적합.

### 1.3 caveat
- **L8 단독결론 금지**: L8은 n이 적고(pilot1 한정) subject 불균형이 큼. level 보고는 L1–L8 / L1–L7 병기 권장(ordinality는 L1–L7에서 더 깨끗).
- length confound는 현재 **보류**(어려운 문제일수록 추론이 길어지는 것은 자명 → 가짜 신호로 보지 않음).

---

## 2. SUBJECT 유사도 — marginal 신호는 있으나 level 통제 시 약함

**분리도**: within_mean = +0.226 | between_mean = −0.128 | **gap = +0.354**, perm-p = 0.005.
group sizes: Algebra 201, Counting & Probability 212, Geometry 211, Intermediate Algebra 199, Number Theory 224, Other 206, Prealgebra 103, Precalculus 185.

### 2.1 centroid-cosine 행렬 (THINKING, centered, pilot1 N=1541)
약어: Alg=Algebra, C&P=Counting & Probability, Geo=Geometry, IntAlg=Intermediate Algebra, NumTh=Number Theory, Pre=Prealgebra, Prec=Precalculus.
```
          Alg     C&P     Geo   IntAlg  NumTh   Other    Pre    Prec
 Alg     1.000  -0.418  -0.184   0.275  -0.094  -0.246   0.280   0.010
 C&P    -0.418   1.000   0.137  -0.602   0.488  -0.389   0.044  -0.640
 Geo    -0.184   0.137   1.000  -0.231  -0.107  -0.426   0.045   0.001
 IntAlg  0.275  -0.602  -0.231   1.000  -0.311   0.049  -0.436   0.640
 NumTh  -0.094   0.488  -0.107  -0.311   1.000  -0.459   0.004  -0.639
 Other  -0.246  -0.389  -0.426   0.049  -0.459   1.000  -0.267   0.259
 Pre     0.280   0.044   0.045  -0.436   0.004  -0.267   1.000  -0.372
 Prec    0.010  -0.640   0.001   0.640  -0.639   0.259  -0.372   1.000
```

### 2.2 해석
- **닮은 과목 쌍(양의 코사인):**
  - Intermediate Algebra ↔ Precalculus **+0.640** (최강)
  - Counting & Probability ↔ Number Theory **+0.488**
  - Algebra ↔ Prealgebra +0.280, Algebra ↔ Intermediate Algebra +0.275
  - Other ↔ Precalculus +0.259
- **상반된 과목 쌍(음의 코사인):**
  - Precalculus ↔ C&P −0.640, Precalculus ↔ Number Theory −0.639
  - C&P ↔ Intermediate Algebra −0.602
  - Number Theory ↔ Other −0.459, Geometry ↔ Other −0.426
  - Intermediate Algebra ↔ Prealgebra −0.436, Algebra ↔ C&P −0.418
- **묶음 구조(2계열 + 고립):**
  1. **대수/해석 계열** {Intermediate Algebra, Precalculus, (Algebra)} — 서로 양, 대표 +0.640.
  2. **이산/수론 계열** {Counting & Probability, Number Theory} — 서로 양 +0.488.
  3. 두 계열은 **서로 강한 음**(예: Prec–C&P −0.640) → 방향이 반대.
  4. **Geometry**는 약하게 C&P쪽(+0.137), 나머지엔 약한 음.
  5. **'Other'는 거의 모든 과목과 음**(NumTh −0.459, Geo −0.426 등) → 고립된 고유 패턴.

### 2.3 핵심 caveat — subject 분리는 marginal일 뿐
- 이 subject 행렬의 분리(gap +0.354)는 **난이도를 고정하지 않은 marginal** 신호다.
- **난이도(level)를 고정하면**(within-level / between-SUBJECT) subject gap = **−0.0418** (거의 0) → "같은 난이도 안에서는 과목으로 거의 안 갈린다".
- 예외: **'Other'만** 난이도와 무관한 독립 클러스터(고유 활성화 패턴)를 형성.
- 따라서 subject는 staging 1차 축이 아니라 **각 stage 내부의 mixing/balancing 축**으로 두는 것이 데이터와 정합.

---

## 3. 종합

| 축 | 분리 gap | perm-p | 핵심 구조 | 커리큘럼 역할 |
|---|---|---|---|---|
| LEVEL | **+0.433** | 0.005 | 인접 유사(ordinality +0.849), 쉬움{L1–3}↔어려움{L5–8} 부호반전, L4 경계 | **1차 staging 축(난이도)** |
| SUBJECT | +0.354 (marginal) | 0.005 | 대수계열↔이산계열 반대, 'Other' 고립 / **level 통제 시 −0.04** | stage 내부 mixing 축 |

**한 줄 결론**: 활성화 1차 결정자는 **LEVEL(난이도)** — 인접 레벨이 닮고 쉬움/어려움이 반대 방향. SUBJECT 기여는 약하고(난이도 통제 시 사실상 0) 'Other'에 국한.

---

## 4. 출처/재현
- 행렬 원본: `REPORT_similarity_pilot.md` (LEVEL/SUBJECT centered, THINKING, N=1541), npz: `sim_matrices_pilot.npz`.
- heatmap: `heatmap_level_THINKING_pilot.png`, `heatmap_subject_THINKING_pilot.png`.
- pooled 요약(3025): subject gap +0.353, level gap +0.434, ordinality +0.841(L1–L8)/+0.896(L1–L7) — `REPORT_pooled_3025.md` / `MASTER_REPORT_phase3_2026-06-12.md`. (행렬 셀 값의 3025 재산출은 별도 보고서.)
- pilot2(N=1417) 재현: `REPORT_similarity_pilot2.md` (ordinality +0.879).
