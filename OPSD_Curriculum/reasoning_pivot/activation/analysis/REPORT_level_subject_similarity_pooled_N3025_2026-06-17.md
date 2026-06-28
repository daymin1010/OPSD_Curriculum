# REPORT — LEVEL / SUBJECT centroid-cosine similarity (POOLED N=3025)

작성: 2026-06-17 / 트랙: reasoning_pivot, pooled(pilot1+pilot2) **THINKING ΔA**, centered (per-layer μ_pooled)
산출물: `SIM_pooled3025_levsubj.txt`, `sim_matrices_pooled3025_levsubj.npz`
스크립트: `level_subject_similarity_pooled.py` (CPU only)
선행: `REPORT_level_subject_similarity_pilot1_N1541_2026-06-15.md` (pilot1 단독), `MASTER_REPORT_phase3_2026-06-12.md` (정본)

---

## 0. 한 줄 요약
- pooled N=3025 전체에서도 **LEVEL 축이 SUBJECT 축보다 분리력이 강하다**: level gap **+0.4345** vs subject gap **+0.3528**, 둘 다 perm-p = 0.005.
- LEVEL centroid는 **연속·단조적 난이도 축**을 형성: ordinality ρ(cos, −|Δlevel|) = **+0.841 (L1–L8) / +0.896 (L1–L7)**.
- → 핸드오프 §C의 "활성화 1차 결정자 = LEVEL, subject는 stage 내 mixing 축" 결론을 **full pooled(3025)에서 재확인**.

---

## 1. 메서드 (요약)
- 입력: pooled THINKING ΔA, per-layer μ_pooled centering, layeravg 뷰.
- 그룹 centroid 간 코사인 → within-group mean vs between-group mean의 gap.
- 유의성: label permutation N_PERM=200, p = #(perm gap ≥ obs)/N.
- ORDINALITY: level centroid cosine과 −|Δlevel|의 Spearman ρ.
- 데이터 사실: N=3025 (pilot1 1608 + pilot2 1417), subjects=8-canonical, levels=1–8.

---

## 2. SUBJECT grouping (groups=8)
group sizes: Algebra 420, Counting&Probability 430, Geometry 406, Intermediate Algebra 387, Number Theory 443, Other 397, Prealgebra 197, Precalculus 345

- within_mean cos = **+0.2316** | between_mean cos = **−0.1212** | **gap = +0.3528**
- permutation p(gap ≥ obs) = **0.0050** (N_PERM=200)

centroid cosine matrix (centered, THINKING):
```
                        Alg    C&P    Geo  IntAlg    NT  Other  Prealg  Precal
Algebra                1.000 -0.419 -0.062  0.291  0.049 -0.483  0.330  0.133
Counting & Prob       -0.419  1.000  0.044 -0.657  0.448 -0.231  0.074 -0.728
Geometry              -0.062  0.044  1.000 -0.196 -0.176 -0.331 -0.085  0.016
Intermediate Algebra   0.291 -0.657 -0.196  1.000 -0.292  0.104 -0.445  0.666
Number Theory          0.049  0.448 -0.176 -0.292  1.000 -0.514  0.126 -0.653
Other                 -0.483 -0.231 -0.331  0.104 -0.514  1.000 -0.387  0.277
Prealgebra             0.330  0.074 -0.085 -0.445  0.126 -0.387  1.000 -0.293
Precalculus            0.133 -0.728  0.016  0.666 -0.653  0.277 -0.293  1.000
```
관찰:
- **Intermediate Algebra ↔ Precalculus = +0.666** (가장 강한 양의 과목쌍): 고급 대수/해석 계열이 활성화 공간에서 인접.
- **Counting&Probability ↔ Precalculus = −0.728**, **C&P ↔ Intermediate Algebra = −0.657**, **Number Theory ↔ Precalculus = −0.653**: 조합/정수론 계열 vs 연속수학 계열이 뚜렷이 대립.
- subject marginal gap(+0.3528)은 핸드오프 §C.2 인용치(+0.353)와 일치 → 과목 자체로는 분리 신호 있음. (단 §C.2의 within-level 통제 시 약화되는 비대칭은 본 산출물 범위 밖, 정본 §C 유지.)

---

## 3. LEVEL grouping (groups=8)
group sizes: L1 335, L2 480, L3 480, L4 437, L5 420, L6 420, L7 387, L8 66

- within_mean cos = **+0.3242** | between_mean cos = **−0.1103** | **gap = +0.4345**
- permutation p(gap ≥ obs) = **0.0050** (N_PERM=200)
- **ORDINALITY ρ(cos, −|Δlevel|): L1–L8 = +0.8411 | L1–L7 = +0.8959**

centroid cosine matrix (centered, THINKING):
```
        1      2      3      4      5      6      7      8
1   1.000  0.918  0.594 -0.353 -0.880 -0.917 -0.802 -0.663
2   0.918  1.000  0.783 -0.140 -0.822 -0.970 -0.902 -0.797
3   0.594  0.783  1.000  0.417 -0.449 -0.783 -0.916 -0.898
4  -0.353 -0.140  0.417  1.000  0.474  0.113 -0.190 -0.321
5  -0.880 -0.822 -0.449  0.474  1.000  0.782  0.581  0.415
6  -0.917 -0.970 -0.783  0.113  0.782  1.000  0.910  0.794
7  -0.802 -0.902 -0.916 -0.190  0.581  0.910  1.000  0.933
8  -0.663 -0.797 -0.898 -0.321  0.415  0.794  0.933  1.000
```
관찰:
- 인접 레벨일수록 코사인 높음 (L1–L2 +0.918, L6–L7 +0.910, L7–L8 +0.933), 먼 레벨일수록 음 (L1–L6 −0.917, L2–L6 −0.970) → **단조적 난이도 축**.
- **L4가 전이대(transition band)**: L1–L3(쉬움 클러스터)과 L5–L8(어려움 클러스터) 사이에서 양쪽과 약한 상관(L4–L3 +0.417, L4–L5 +0.474, 그 외 0 근처/약한 음). 쉬움↔어려움 경계가 대략 L3/L4~L5 부근임을 시사 → **stage 경계 설계 시 L4 부근이 자연 분기점**.
- gap(+0.4345) 및 ρ(L1–L8 +0.841)은 핸드오프 §C.1 인용치(+0.434, ρ +0.84~0.90)와 일치 → full pooled에서 재확인.

---

## 4. SUBJECT vs LEVEL 비교 (pooled 3025)
| 축 | within | between | gap | perm-p | ordinality |
|---|---|---|---|---|---|
| SUBJECT | +0.2316 | −0.1212 | **+0.3528** | 0.005 | (n/a) |
| LEVEL | +0.3242 | −0.1103 | **+0.4345** | 0.005 | ρ +0.841 (L1–L8) / +0.896 (L1–L7) |

- **LEVEL gap > SUBJECT gap** (+0.4345 > +0.3528). within-group 응집도도 level이 더 높음(+0.324 > +0.232).
- LEVEL은 단조적 순서구조(ordinality)까지 갖춤 → curriculum staging의 **1차 축으로 정당**.
- SUBJECT는 marginal 신호는 있으나(특히 IntAlg–Precalc 계열 vs C&P–NT 계열의 대립 구조), 순서 구조는 없음 → **stage 내 mixing/balancing 축**으로 사용 권장.

---

## 5. caveats
- **L8 단독 결론 금지**: L8 n=66, 전부 pilot1, 5/8 subject만. level 보고는 L1–L8 / L1–L7 병기(본 보고서 준수).
- length confound: 핸드오프 §0대로 **보류**(현 의사결정 미반영).
- 본 산출물은 marginal(subject, level) 축. joint unit(57셀)은 비사용(별도 트랙).
- centering/뷰: per-layer μ_pooled, layeravg. midL11-15 뷰는 robustness용으로 동일 정렬 기대(정본 §D).

---

## 6. 산출물 인덱스
- `SIM_pooled3025_levsubj.txt` — 본 수치 원본 텍스트.
- `sim_matrices_pooled3025_levsubj.npz` — subject/level centroid cosine 행렬, gap, perm 결과 배열.
- 로그: `runs/levsubj_pooled3025.log`.

---

## Length-confound robustness — 3-method 일치 (POOLED N=3025)

**스크립트:** `levsubj_length_confound_pooled.py` → `LENGTHCONF_pooled3025.txt`, `lengthconf_pooled_outputs.json`
**설정:** pooled N=3025 (pilot1=1608, pilot2=1417); μ_pooled(per-layer) centering; seed=42; CPU only. ρ(level, gen_len)=+0.709.

**세 방법(중복 아님):**
- ① **Mantel** — 그룹 활성화 거리행렬(1−cos) vs gen_len 거리행렬(평균차/Wasserstein) off-diag Pearson r + perm p. 낮고 비유의 → length 정렬 아님.
- ② **residual survival** — 각 feature 를 gen_len(±log/±quad, +level)에 GLOBAL 회귀한 잔차에서 centroid-cosine 재계산 → 원본과 상관. 높게 생존 → content.
- ③ **gen_len-balanced** — gen_len 5분위로 그룹 매칭한 부분표본에서 gap 재현. 유지되면 length artifact 아님.

### SUBJECT (groups=8) — PASS(content-driven)
| view | Mantel r (mean / Wass.) | perm p | residual survival min | balanced |
|---|---|---|---|---|
| layeravg | +0.027 / +0.058 | 0.819 / 0.624 | +0.958 | gap +0.353→+0.201 (유지 57%), p=0.005, N_bal=1200 |
| mid_L11-15 | +0.009 / +0.036 | 0.943 / 0.795 | +0.974 | — |

### LEVEL (groups=8) — Mantel/residual FAIL, balanced 유지
| view | Mantel r (mean / Wass.) | perm p | residual survival min | balanced |
|---|---|---|---|---|
| layeravg | +0.858 / +0.858 | 0.0001 | +0.624 | gap +0.435→+0.384 (유지 88%), p=0.005, N_bal=142 |
| mid_L11-15 | +0.850 / +0.850 | 0.0001 | +0.668 | — |

### 종합 / 해석
- **SUBJECT**: Mantel 비유의(r≈+0.03), residual 생존(min +0.96~+0.97). length 와 거의 독립 = content-driven. (단, balanced 유지율 57%로 smoke(109%)보다 낮아짐 — 그래도 p=0.005 유의.)
- **LEVEL**: Mantel 고상관(r≈+0.86, p≈0.0001) & residual 부분붕괴(min +0.62~+0.67)로 length 와 강하게 동행하나, **gen_len-balanced subsample 에서 gap 88% 유지(p=0.005)** → length 만으로는 설명 안 됨.
- 8×8/8그룹 행렬 1개로 강결론 금지(Mantel 은 perm p 만). residual survival + balanced gap 을 주증거로.
- **§0 정합:** 어려운 문제일수록 추론이 길어지는 것은 자명하므로 level↔gen_len 동행은 가짜 신호 근거가 아니며, balanced 검정에서 gap 이 유지되므로 활성화 난이도 축을 정당한 신호로 계속 사용한다(length confound = 보류).
