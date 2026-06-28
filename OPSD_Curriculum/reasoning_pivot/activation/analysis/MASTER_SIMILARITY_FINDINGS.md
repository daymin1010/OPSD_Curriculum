# MASTER — ΔA Group-Similarity 분석 결과 정리 (thinking-mode pilot)

작성일: 2026-06-04
대상: Qwen3-8B thinking-mode pilot activation-shift (ΔA)
입력: `src/OPSD_Curriculum/reasoning_pivot/activation/outputs/shifts/*.pt`  (N=1541 problems)
분석 스크립트: `analysis/similarity_analysis.py`
자동생성 원시 리포트(행렬 전체·보존): `analysis/REPORT_similarity_pilot.md`
행렬 아카이브: `analysis/sim_matrices_pilot.npz`
히트맵: `analysis/heatmap_{subject,level,unit}_{THINKING,FAITHFUL}_pilot.png`

> 이 문서는 자동생성 REPORT 를 **해석·방법론·수식·코드주석** 관점에서 사람이 읽도록 재정리한 마스터 요약본입니다.
> 자동생성 REPORT(수치/행렬 원본)는 그대로 보존합니다.

---

## 0. 한 줄 결론 (Verdict)

> **ΔA(activation-shift)에는 라벨과 무관한 강한 공통 reasoning-shift 축이 지배적이지만, 이 공통 성분을 제거(global-mean centering)하면 subject·level 구조가 통계적으로 유의하게 드러난다.**
> 특히 **LEVEL(난이도)은 강한 ordinality(단조성, ρ≈0.85)** 를 보이며 — 가까운 난이도일수록 활성-이동 방향이 닮았다.
> **think-블록 내부 shift(`dA_thinking`)가 전체-시퀀스 shift(`dA_faithful`)보다 더 강하고 길이 교란에 더 견고**했다 → 이전 세션의 *thinking-mode 가설* 을 지지.

---

## 1. 데이터 & 표현 (Representation)

각 문제 `i` 에 대해 Qwen3-8B 를 thinking-mode 로 생성시키고, 5개 토큰 위치에서
**36개 레이어의 MLP `down_proj` 입력**(hidden = 12288-d)을 forward-pre-hook 으로 캡처.
(추출 코드: `extract_thinking_pilot.py` L191–321)

저장된 위치 텐서 (각 `(36, 12288)`):
`A_pos0`(시퀀스 0번), `A_prompt_last`(프롬프트 마지막), `A_t1_think`(`<think>` 다음),
`A_tK_think`(`</think>` 직전, 잘리면 마지막 토큰), `A_last`(전체 마지막).

본 분석이 사용하는 두 가지 ΔA 정의:

| 이름 | 정의 (코드) | 의미 |
|---|---|---|
| **`dA_thinking`** (분석 라벨 `THINKING`) | `A_tK_think − A_t1_think` (L320) | **think 블록 내부**의 시작→끝 활성 변화. `think_valid` 아니면 0벡터 |
| **`dA_faithful`** (분석 라벨 `FAITHFUL`) | `A_last − A_pos0` (L319) | **프롬프트 첫 토큰 → 생성 끝** 전체 활성 변화 (NAIT-faithful) |

→ "think 구간 사이의 shift" 와 "전체 프롬프트부터 끝까지의 shift" **둘 다** 측정·분석함.

각 ΔA_i ∈ ℝ^(36 × 12288). (fp16 로딩 시 (N,36,12288)×2 ≈ 2.7 GB, CPU.)

**그룹핑 3종:**
- **UNIT** = (subject, level) 셀, 멤버 수 n ≥ `MIN_N=10` 인 셀만 → 57 셀
- **SUBJECT** = 8 canonical subjects
- **LEVEL** = 1..8 (ordinal; 1=쉬움 ↔ 8=어려움)

**N=1541** (subjects=8, levels=[1..8], units(n≥10)=57).
그룹 크기는 대체로 균형(subject당 ~100–224, level당 ~200, L8만 62로 희소).

---

## 2. 방법론 + 계산식 (Methodology)

표기: `Â_l = A_l / ‖A_l‖₂` (레이어 l 의 L2-정규화 벡터), `⟨·,·⟩` 내적, `D = 12288`, `L = 36`.

### 2.1 Layer-averaged cosine (유사도의 기본 단위)
두 활성-이동 텐서 A, B ∈ ℝ^(L×D) 사이 유사도는 **레이어별 코사인을 레이어에 대해 평균**:

```
s(A, B) = (1/L) · Σ_{l=1..L}  ⟨ Â_l , B̂_l ⟩
```

- 평탄화(442k-d) 코사인 대신 레이어별로 정규화 후 평균 → 소수의 고-norm 레이어가 지배하는 것을 방지.
- 코드: `layeravg_cos()` (L101–105)

### 2.2 공통 성분 제거 (Global-mean Centering) — PRIMARY
원시 ΔA 는 라벨 무관 공통축이 지배적이라 centroid 코사인이 거의 1 → 정보가 없음.
그래서 **전역 평균을 레이어별로 빼고** centroid 를 계산 (이게 주 결과; raw 는 투명성용 병기):

```
μ[l] = (1/N) · Σ_{i=1..N}  ΔA_i[l]            # 레이어별 전역 평균
ΔÃ_i[l] = ΔA_i[l] − μ[l]                       # centered
```

- 코드: `main()` L327–328 `mu = DA.mean(axis=0); DA_c = DA − mu`

### 2.3 그룹 centroid
```
C_g[l] = (1/|g|) · Σ_{i∈g}  ΔÃ_i[l]           # centered 데이터의 그룹 평균
```
- 코드: `centroids()` (L108–110)

### 2.4 유사도 행렬
```
S[g,h] = s(C_g, C_h)        # centroid 간 layer-averaged cosine
```
- 코드: `sim_matrix()` (L113–120)

### 2.5 분리도 (Separability) — 그룹핑이 의미 있는가?
```
within  = ⟨ s(ΔÃ_i, C_g) ⟩_{i∈g, g}          # 멤버↔자기 centroid 평균 코사인
between = ⟨ s(C_g, C_h) ⟩_{g≠h}              # 서로 다른 centroid 간 평균 코사인
gap     = within − between
```
- `gap` 이 클수록 그룹이 **내부적으로 일관**되고 **상호 구별**됨.
- 코드: `within_between()` (L129–147). within 계산은 사전 정규화한 멤버 `DAn` 으로 `einsum("mld,ld->ml")` 벡터화.

### 2.6 유의성 (Permutation test)
귀무가설 = "그룹 라벨은 ΔA 방향과 무관". 라벨을 `N_PERM=200` 회 셔플 후 `gap` 재계산:

```
p = ( #{ gap_perm ≥ gap_obs } + 1 ) / ( N_PERM + 1 )
```

- 관측 gap 이상이 0회면 p = 1/201 ≈ **0.0050** (해상도 하한).
- 코드: `perm_pvalue()` (L150–171)

### 2.7 Ordinality (LEVEL 전용)
level 은 순서형이므로 "인접 레벨일수록 더 유사한가"를 검정:
```
ρ = Spearman( S[a,b] ,  −|level_a − level_b| )    (off-diagonal 쌍에 대해)
```
- ρ > 0 → 가까운 난이도일수록 centroid 코사인이 큼 (연속 난이도 축의 증거).
- 코드: `run_grouping()` L264–272, `spearman()` L86–94

### 2.8 길이 교란 강건성 (gen_len-balanced robustness)
centroid 코사인이 단지 생성 길이(gen_len) 차이를 반영할 수 있으므로,
**gen_len 5분위 층화 후 각 분위 내에서 그룹별 최소 멤버수만큼 균등 추출**한 부분표본에서
centered SUBJECT/LEVEL gap 을 재계산.
- 코드: `genlen_balanced_indices()` (L283–301)

---

## 3. 결과 (Results)

### 3.1 RAW vs CENTERED — centering 이 왜 필수인가
원시 ΔA 는 within·between 둘 다 +0.84~+0.99 로 거의 1 → gap **음수**.
즉 라벨 무관 공통 reasoning-shift 축이 압도. centering 후 비로소 구조가 드러남.

| 변형 (raw) | within | between | gap |
|---|---:|---:|---:|
| THINKING SUBJECT | +0.858 | +0.981 | −0.123 |
| THINKING LEVEL | +0.856 | +0.949 | −0.093 |
| FAITHFUL SUBJECT | +0.858 | +0.996 | −0.138 |
| FAITHFUL LEVEL | +0.836 | +0.972 | −0.136 |

(이전 `unit_analysis` 에서 within/between 모두 ~0.9 였던 관찰과 일치.)

### 3.2 CENTERED (PRIMARY) — subject·level·unit 모두 유의
| (centered) | within | between | **gap** | perm p |
|---|---:|---:|---:|---:|
| **THINKING** SUBJECT | +0.226 | −0.128 | **+0.354** | 0.0050 |
| **THINKING** LEVEL | +0.323 | −0.110 | **+0.433** | 0.0050 |
| **THINKING** UNIT (57셀) | +0.441 | −0.012 | **+0.453** | 0.0050 |
| FAITHFUL SUBJECT | +0.165 | −0.125 | +0.290 | 0.0050 |
| FAITHFUL LEVEL | +0.287 | −0.124 | +0.411 | 0.0050 |
| FAITHFUL UNIT (57셀) | +0.382 | +0.006 | +0.377 | 0.0050 |

→ 모든 grouping 에서 permutation p = 0.005 (200회 셔플 중 관측 gap 이상 0회).
그룹 라벨이 ΔA 방향과 실제로 결합되어 있음.

### 3.3 ORDINALITY (LEVEL) — 강한 단조 난이도 축
| variant | ρ( centroid_cos, −|Δlevel| ) |
|---|---:|
| THINKING (centered) | **+0.849** |
| FAITHFUL (centered) | **+0.860** |
| THINKING (raw) | +0.976 |
| FAITHFUL (raw) | +0.817 |

→ **난이도가 가까운 레벨일수록 activation-shift 방향이 더 유사**. activation 공간에 연속적 난이도 축이 존재.

### 3.4 길이 교란 강건성 — THINKING 이 더 견고
| balanced (centered) | N | gap | perm p | ρ(ordinality) |
|---|---:|---:|---:|---:|
| THINKING SUBJECT | 632 | **+0.249** | 0.0050 | — |
| THINKING LEVEL | 106 | **+0.399** | 0.0050 | **+0.905** |
| FAITHFUL SUBJECT | 632 | +0.117 | 0.0199 | — |
| FAITHFUL LEVEL | 106 | +0.258 | **0.4030** | +0.209 |

→ THINKING 은 길이 균형 후에도 subject·level 분리도와 단조성 **모두 생존**.
→ FAITHFUL 은 subject 는 약화하나 생존, **LEVEL 은 동일 소표본(N=106)에서 붕괴**(p=0.40, ρ=0.21).
즉 같은 검정력 조건에서 **think-span shift 신호가 길이 교란에 더 강건**.

### 3.5 SUBJECT family 구조 — 계층 클러스터링 (dendrogram)

centered SUBJECT centroid-cosine 행렬 `S` 를 거리 `D[a,b]=1−S[a,b]` 로 변환해
average-linkage 계층 클러스터링(scipy)을 수행. 코드: `dendrogram_subjects.py`
(입력: 기존 `sim_matrices_pilot.npz` — 재계산 없음). 그림:
`dendrogram_subject_{THINKING,FAITHFUL}_pilot.png`, 멤버십: `cluster_subject_families_pilot.txt`.

**flat 2-cluster cut 결과:**

| ΔA | family A (연속/대수 계열) | family B (이산/조합·기하 계열) |
|---|---|---|
| **FAITHFUL** | Algebra, Intermediate Algebra, Precalculus, Prealgebra, Other | **Counting & Probability, Geometry, Number Theory** |
| THINKING | Algebra, Counting & Probability, Geometry, Number Theory, Prealgebra | Intermediate Algebra, Precalculus, Other |

→ **FAITHFUL(전체-시퀀스 shift)이 교과서적 "연속(대수/해석) vs 이산(조합·정수·기하)" 분리를 가장 깨끗하게 재현.**
   Counting & Probability / Number Theory / Geometry 가 한 가족으로 묶이는 것은 이 과목들이
   공통의 이산·열거·조합 추론 패턴을 공유한다는 직관과 일치.
→ THINKING(think-span shift)은 분리축이 다소 다름(Intermediate Algebra·Precalculus·Other 가
   "고급 해석" 가족으로 분기) — think 내부 이동은 *표면 과목 라벨*보다 *추론 난이도/추상도* 축을
   더 강하게 반영함을 시사(§3.3 ordinality 우위와 정합).

> 주의: subject family 는 2-cluster cut 의 편의적 요약. 8개 과목·N=1541 기준 스크리닝이며,
> Other(혼합 라벨) 포함 가족은 해석에 신중할 것(hand-off Quirk #2).

### 3.6 Residualize ablation (PCA top-K 제거) — 진행 중/병기

centering(전역평균=top-0)이 공통 reasoning-shift 축 제거에 충분한지 확인하기 위해,
**레이어별 PCA 상위 K개 주성분을 추가 제거**(K∈{0,1,2})한 뒤 subject/level gap·유의성·ordinality
재계산. 코드: `residualize_analysis.py` (`similarity_analysis.py` helper 재사용),
산출물: `REPORT_residualize_pilot.md`, `residualize_summary_pilot.csv`.
(K=0 은 §3.2 mean-centered 수치를 재현하는 sanity cross-check.)

**full pilot(N=1608) 결과 (perm p 전부 = 0.005):**

| ΔA | group | K=0 gap | K=1 gap | K=2 gap | ordinality ρ (K0/K1/K2) |
|---|---|---:|---:|---:|---|
| THINKING | subject | 0.353 | **0.359** | 0.363 | — |
| THINKING | level | **0.432** | 0.350 | 0.308 | 0.862 / 0.826 / 0.808 |
| FAITHFUL | subject | 0.295 | 0.296 | 0.294 | — |
| FAITHFUL | level | **0.406** | 0.319 | 0.298 | 0.860 / 0.744 / 0.776 |

(K=0 은 §3.2 mean-centered 수치를 정확히 재현 — sanity cross-check 통과.)

**해석:**
- **SUBJECT gap 은 PC 추가제거(K=1,2)에 거의 불변**(THINKING 0.353→0.363, FAITHFUL 0.295→0.294).
  → 과목 구조는 top-variance 공통축과 **사실상 직교**. 전역평균 centering(K=0)만으로 충분하며,
    공통 reasoning-shift 축은 과목 정보를 거의 운반하지 않음.
- **LEVEL gap 은 PC 제거 시 단조 감소**(THINKING 0.432→0.308, FAITHFUL 0.406→0.298).
  → 난이도 신호는 top 주성분과 **부분적으로 겹침**. 즉 공통축(특히 PC1)이 난이도 정보를 일부 운반.
    단 ordinality ρ 는 K=2 에서도 0.8 안팎으로 견고 → 난이도의 **순서(ordinal) 구조 자체는
    PC1 제거 후에도 생존**(THINKING ρ 0.81 유지가 특히 견고).
- **종합:** 공통축은 "순수 잡음"이 아니라 **난이도가 부분적으로 실려 있는 축**이다.
  따라서 NAIT 난이도 방향 추출 시 top PC 를 통째로 폐기하면 신호 손실 위험 → **centering(K=0)
  기반이 적절**하고, residualize 는 "공통축≠난이도 전부"를 보이는 진단 도구로 둔다.
  (이전 세션 "PC1 단일축 신호 부족"은 공통축이 과목엔 무관·난이도엔 부분중첩이라는 이 이중성과 정합.)

원시 수치/행렬: `residualize_summary_pilot.csv`, 상세: `REPORT_residualize_pilot.md`.

---

## 4. 주석 달린 핵심 코드 (발췌)


```python
# (2.1) layer-averaged cosine : 레이어별 L2정규화 후 내적, 레이어 평균
def layeravg_cos(A, B):                 # A,B: (36, D)
    An = l2norm_rows(A.astype(np.float32))   # 각 레이어 벡터를 단위벡터로
    Bn = l2norm_rows(B.astype(np.float32))
    return float((An * Bn).sum(axis=1).mean())   # (36,) 코사인 → 평균

# (2.2) global-mean centering : ΔA의 공통 reasoning-shift 축 제거 (PRIMARY)
mu   = DA.astype(np.float32).mean(axis=0, keepdims=True)  # μ[l] : 레이어별 전역평균
DA_c = DA.astype(np.float32) - mu                          # ΔÃ_i = ΔA_i - μ

# (2.3) 그룹 centroid : centered 데이터의 그룹 평균
def centroids(DA, idx_by_group):
    return {g: DA[idx].mean(axis=0) for g, idx in idx_by_group.items()}  # C_g (36,D)

# (2.5) 분리도 : within(멤버↔centroid) - between(centroid간)
def within_between(DA, idx_by_group, cents, order, DAn=None):
    if DAn is None: DAn = normalize_members(DA)   # 멤버 1회 정규화(perm 루프 비용 절감)
    withins = []
    for g in order:
        Cn = l2norm_rows(cents[g])                 # 자기 centroid 정규화
        cos_ml = np.einsum("mld,ld->ml", DAn[idx_by_group[g]], Cn)  # (멤버, 36)
        withins.append(float(cos_ml.mean()))
    within_mean = float(np.mean(withins))
    bet = [layeravg_cos(cents[a], cents[b]) for a<b]   # 모든 centroid 쌍
    between_mean = float(np.mean(bet))
    return within_mean, between_mean, within_mean - between_mean, withins

# (2.6) permutation p : 라벨 셔플 후 gap 분포에서 관측 gap의 위치
def perm_pvalue(DA, labels, order, observed_gap, DAn=None):
    ge = 0
    for _ in range(N_PERM):                        # 200회
        perm = rng.permutation(labels)             # 라벨만 섞음(데이터 고정)
        idxg = {g: where(perm==g) ...}             # 셔플된 그룹 인덱스
        _, _, gap, _ = within_between(DA, idxg, centroids(DA, idxg), ...)
        if gap >= observed_gap: ge += 1
    return (ge + 1) / (N_PERM + 1)                 # +1 smoothing

# (2.7) ordinality(LEVEL) : centroid 코사인 vs -|레벨차| 의 Spearman ρ
rho = spearman(pairs_cos, pairs_negdist)           # >0 => 가까운 레벨일수록 유사
```

(전체 구현은 `similarity_analysis.py` 참조 — 메서드 docstring 이 본 수식과 1:1 대응.)

---

## 5. 그림 인덱스 (heatmaps)

`vmin=-1, vmax=1`, RdBu_r. 대각=자기 자신(=1.0). 값 = centered centroid 간 layer-avg cosine.

- `heatmap_subject_THINKING_pilot.png` / `heatmap_subject_FAITHFUL_pilot.png` — 8×8 subject 유사도
- `heatmap_level_THINKING_pilot.png` / `heatmap_level_FAITHFUL_pilot.png` — 8×8 level 유사도 (대각 밴드 구조 = ordinality 시각화)
- `heatmap_unit_THINKING_pilot.png` — 57×57 (subject×level) 유사도

---

## 6. 해석 & 연구적 함의

1. **공통축 + 구조의 이중구조.** ΔA 는 (a) 라벨 무관 공통 reasoning-shift 성분(원시 코사인 ~0.9)과 (b) subject/level-특이 성분이 중첩. NAIT 류 방향 추출은 반드시 (a)를 제거(centering/residualize)한 뒤 (b)를 봐야 한다. (이전 세션 "PC1 단일축 신호 부족"의 원인 = (a)가 PC1을 점유.)

2. **연속적 난이도 축 존재.** LEVEL ordinality ρ≈0.85 (길이 균형 후 THINKING ρ≈0.90 유지) → 활성공간에 단조적 난이도 차원이 있다. **NAIT-curriculum 의 난이도 정렬을 activation 기반으로 정의할 직접 근거.**

3. **thinking-mode 채택 권고.** think-span shift 가 전체-시퀀스 shift 보다 분리도·단조성·길이강건성 모두 우위 → 후속 방향 추출/커리큘럼은 `dA_thinking` 기반 권장.

4. **subject 구조도 유의.** centered subject gap +0.35(THINKING), 길이균형 후도 생존 → 과목별 추론 이동 방향이 구별됨. 과목 혼합 커리큘럼 설계 시 활용 가능.

---

## 7. 한계 & 다음 단계

- **balanced LEVEL 검정력 부족(N=106, 레벨당 ~14).** FAITHFUL LEVEL p=0.40 은 "신호 없음"이 아니라 "이 소표본에선 검출 불가". THINKING 은 같은 N 에서 생존하므로 *상대비교*는 유효하나, 절대 결론은 표본 확대 필요.
- **L8 희소(전체 62, 일부 subject 0).** L8 단독 cell 결론 금지(hand-off Quirk #4와 일관).
- **MIN_N=10, N_PERM=200** 은 스크리닝 수준. 확정 단계에선 N_PERM≥1000, pilot 확대(예: 3,000 풀 활용) 권장.
- **다음 분석 후보:**
  - `is_correct`(저장됨) · `pass_rate` · `r1_cot_token_count` 와 ΔA centroid/probe 결합 (난이도 축 ↔ 정답률 검증).
  - layer-window별 분리도(어느 레이어대가 subject/level을 가장 잘 분리하는가) — 이전 1.5B 의 layer-ablation 재현.
  - centered ΔA 에 대한 supervised direction(level 회귀/LDA) 으로 단일 난이도 축 추출.
  - `A_prompt_last` 활용한 제3 shift("프롬프트끝→생성끝") 비교 (재추출 불필요, .pt 에 이미 저장).

---

## 8. 재현 (Reproduce)

```bash
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
cd /scratch/lami2026/personal/jimin_2782
OMP_NUM_THREADS=4 $PY -u \
  src/OPSD_Curriculum/reasoning_pivot/activation/analysis/similarity_analysis.py \
  --shifts-dir src/OPSD_Curriculum/reasoning_pivot/activation/outputs/shifts \
  --out-dir    src/OPSD_Curriculum/reasoning_pivot/activation/analysis \
  --tag pilot
# CPU only. 산출물: REPORT_similarity_pilot.md, sim_matrices_pilot.npz, heatmap_*.png
```

행렬 재로딩:
```python
import numpy as np
z = np.load("analysis/sim_matrices_pilot.npz", allow_pickle=True)
print(z.files)                                  # *_S, *_order 키
S = z["THINKING_centered_level_S"]; order = z["THINKING_centered_level_order"]
```
