# MASTER SUMMARY — NAIT Activation Analysis on Qwen3-8B
*pilot 2,666 sample · 36 layers · D=12288 · ΔA = h_tK − h_t1 (MLP down_proj input)*

작성일: 2026-05-25 (Phase 0 / Phase A / Phase Critical 완료 시점)

---

## 0. Overview

| 항목 | 값 |
|---|---|
| Pilot universe | 2,666 sample (subject × level unit-stratified, MATH L1–L8) |
| Backbone | Qwen3-8B (non-reasoning, enable_thinking=False) |
| 활성화 추출 지점 | last-token, **MLP `down_proj` input** (per layer) |
| Hidden dim | 12,288 (intermediate size) |
| 레이어 수 | 36 (transformer blocks) |
| Pass rate measurement | 8-sample rollout per problem (Track A) |
| 산출물 베이스 | `src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/` |

3-phase 분석을 차례로 수행했으며, 핵심 질문은 다음과 같다:

> **Q: ΔA(activation shift) 안에 subject/level/difficulty 신호가 존재하는가?**
> **Q′: 만약 존재한다면 그것이 "모델의 추론 과정"에서 온 신호인가, 아니면 prompt token만으로 이미 결정된 footprint인가?**

---

## 1. Phase 0 — Calibrated PC1 (unsupervised direction)

레이어별 ΔA 집합에 대해 (1) PCA → PC1 추출, (2) μ_diff = mean(h_tK − h_t1) 계산, (3) sign(μ_diff · v_l) 으로 부호 보정 (논문 Eq.4). 보정된 단일 방향 `v_cal[l]`로 한 차원 projection score 산출.

📄 `outputs/REPORT.md`, `outputs/scores_calibrated.npy`, `outputs/directions.npz`

### Phase 0 핵심 결과

| 측정 | best layer | 값 | 해석 |
|---|---|---|---|
| ρ(PC1 score, **level**) | L21 | **−0.808** | PC1 한 축에도 level/난이도 신호 강함 |
| ρ(PC1 score, **pass**) | L11 | **+0.561** | model-internal difficulty와 정합 |
| **subject silhouette (1D PC1)** | — | **−0.13 ~ −0.31** | **PC1 한 축에서 subject 구조 거의 없음** |
| subject silhouette (top-8 PC) | — | −0.02 ~ −0.09 | top-8까지 늘려도 미약 |
| subject F1 (linear probe, PCA-256) | L14 | 0.737 | **PC1엔 안 보이지만 고차원에선 readable** |
| level R² (ridge, PCA-256) | L18 | **+0.878** | 거의 포화 |
| pass R² (ridge, PCA-256) | L16 | +0.293 | 약하지만 양 |

### Phase 0 결론

- **"PC1 기준만 보면 subject/level 정보가 안 보인다"는 사용자의 직관은 정확함.**
  PC1은 사실상 magnitude/length의 공통 shift 축이며, level 신호 일부가 우연히 그 축에 정렬돼 있다 (강한 음의 상관: 어려운 문제일수록 ΔA 크기가 다른 방향으로 끌림).
- subject 신호는 PC1 한 축이 아니라 **subspace** 에 분포 → supervised direction 필요.

---

## 2. Phase A — Supervised direction (LDA / ridge)

PCA-256 축소 후 supervised: subject는 multi-class LDA, level은 (a) ridge 회귀 (b) median-split binary LDA-1. 5-fold CV. 적합된 방향을 D=12288로 back-project하여 `lda_directions.npz` 저장.

📄 `outputs/REPORT_supervised.md`, `outputs/supervised_per_layer.csv`, `outputs/lda_directions.npz`

### Phase A 핵심 결과

| Top-5 layer | F1/R² | metric |
|---|---|---|
| **Subject window** [11, 12, 13, 14, 15] | 0.757 – **0.779** | LDA macro-F1 |
| **Level window**   [15, 16, 17, 18, 19] | +0.865 – **+0.879** | ridge R² |
| Level (LDA-1 ρ best) | L23 | +0.892 |

### PC1 vs supervised direction 정합도 (best layer 18)

| direction pair | |cosine| |
|---|---|
| PC1 vs v_level_ridge | **0.107** |
| PC1 vs v_level_LDA   | 0.075 |
| PC1 vs W_subject (7-axis 평균) | 0.004 |

→ **거의 직교**. unsupervised PC1과 supervised difficulty/subject axis는 별개 방향이다.

### Phase A 결론

- ΔA의 **subject 신호는 mid layer (11–15)**, **level/difficulty 신호는 그 직후 (15–19)** 에 separated window로 분포.
- 두 window는 일부 겹치지만 분리된 axis → curriculum stage feature 설계 시 두 window 각각 projection이 자연스러움 (NAIT 원 논문 가정과 일치).
- ΔA 안에 subject·level 정보 모두 **linear probe로는 풍부하게 잡힘**.

---

## 3. Phase Critical — ΔA vs A_prompt (confound check)

**가장 중요한 검증**. 만약 ΔA 안의 subject/level 신호가 단지 *prompt token 자체의 footprint* 라면 (예: "Geometry" 라는 단어 자체, prompt length 분포 등), ΔA는 모델의 **reasoning** 과정에 대한 정보를 거의 추가하지 않는다는 뜻이 된다.

이를 검증하기 위해 동일 2666 sample에 대해:
- **A_prompt** = last prompt token activation (generation 시작 전, `h_t1` 시점)
- 36 레이어 × 12288 차원 → 4.72 GB 캐시 생성 (`outputs/critical/prompt_act_cache.npy`)
- Phase A 와 동일 pipeline (PCA-256 → LDA / ridge, 5-fold CV)

📄 `outputs/critical/REPORT_critical.md`, `outputs/critical/prompt_per_layer.csv`, `outputs/critical/critical_compare.csv`

### Critical: prompt-only activation 단독으로 얼마나 잡히는가

| signal | best layer | A_prompt 값 |
|---|---|---|
| subject macro-F1 | L13 | **0.788** |
| level ridge R²   | L21 | **+0.908** |
| level LDA-1 \|ρ\| | L18 | **0.912** |
| pass_rate ridge R² | L27 | **+0.326** |

### ΔA vs A_prompt — best layer head-to-head

| signal | **ΔA best** | **A_prompt best** | gap (ΔA − A_prompt) |
|---|---:|---:|---:|
| subject F1     | 0.779 | **0.788** | **−0.009** |
| level R²       | +0.879 | **+0.908** | **−0.029** |
| pass R²        | +0.293 | **+0.326** | **−0.033** |

### Phase Critical 결론 — ⚠️ 매우 중요

1. **Subject 신호는 거의 100% prompt token의 keyword footprint다** (gap −0.009).
   "ΔA가 subject를 안다"라는 Phase A 표현은 **"prompt가 이미 subject를 거의 다 알려준다"** 로 reframe해야 한다.

2. **Level 신호도 prompt가 ΔA보다 *더 잘* 예측한다** (R² gap −0.029).
   Level 정보 또한 prompt의 표면 feature (문제 길이, 어휘, LaTeX 빈도, 키워드)에서 대부분 결정된다.

3. **Pass-rate조차 prompt만으로 R² ≈ 0.33 예측 가능**.
   즉 model-internal pass rate의 1/3 분산이 prompt 표면 feature로 설명된다 (문제 길이, 어휘 난이도가 reasoning 성공 가능성과 강하게 상관).

4. **시사점**: ΔA를 curriculum feature로 쓸 때, 그것이 *prompt feature 위에 추가로 무엇을 더 알려주는지* 는 별도로 측정해야 함. naively ΔA를 쓰면 사실상 prompt embedding을 쓰는 것과 거의 같다.

---

## 4. 종합 결론 & 권고

### 4.1 사용자의 직관 검증

> "지금 activation shift의 subject, level 정보가 잘 나타나지 않은 걸로 보이는데, PC1 기준으로만 파악한 거지?"

**정확함.** Phase 0 PC1 한 축에는 subject 신호가 거의 없다 (silhouette ≈ −0.04). 다만:
- Level 신호는 PC1에 우연히 강하게 정렬돼 있다 (ρ ≈ −0.81).
- Subject·Level 신호 모두 고차원(PCA-256) supervised probe에서는 풍부하게 잡힌다 (F1 ≈ 0.78, R² ≈ 0.88).
- **그러나** 이들 신호는 prompt token 활성화만으로도 거의 동일하게 추출 가능하다 (Phase Critical).

### 4.2 curriculum 설계 함의

NAIT의 본래 motivation은 *"모델이 풀어보면서 발생하는 representation shift (ΔA) 가 prompt embedding보다 더 유용한 difficulty signal"* 인데, Qwen3-8B + MATH에서는:

- **Subject 분리는 ΔA 없이 prompt embedding만으로 충분**.
- **Difficulty(level) 분리도 prompt에서 거의 다 잡힌다**.
- **pass_rate의 진짜 model-internal residual**(prompt feature로 설명 못 하는 부분)은 ΔA에서도 매우 약함 (R² 0.29 ↔ 0.33).

→ Curriculum 정의에 ΔA를 무겁게 쓸지, **prompt embedding + pass_rate 직접 사용**으로 갈지 재검토 필요. (사용자 지시 시 두 옵션 비교 분석 가능.)

### 4.3 다음 step 후보 (사용자 결정)

| 옵션 | 내용 | 비용 |
|---|---|---|
| **B1** | ΔA residual probe: ΔA에서 A_prompt 선형 prediction을 뺀 *residual* 에 대해 F1/R² 재측정. 진짜 "추가 정보량" 정량화. | 30분, CPU |
| **B2** | curriculum stage 정의 (사용자 지시 대기) — Phase A LDA score + pass_rate 결합 | 1–2h, CPU |
| **B3** | 40K full extension (Track A pass_rate 확장) | ~45h, GPU |

---

## 5. Artifact index

```
src/OPSD_Curriculum/analysis_qwen3_8b/nait/
├── MASTER_SUMMARY_qwen3_8b.md     ← (this file)
├── ANALYSIS_LOG.md
├── direction_calibrated.py        # Phase 0 driver
├── supervised_direction.py        # Phase A driver
├── extract_prompt_activation.py   # Phase Critical: A_prompt 추출
├── analyze_critical.py            # Phase Critical: 비교 분석
├── build_curriculum.py            # (대기) curriculum 빌더
├── sbatch/
│   ├── run_smoke_prompt_act.sh
│   └── run_pilot_prompt_act.sh
└── outputs/
    ├── REPORT.md                   ← Phase 0
    ├── REPORT_supervised.md        ← Phase A
    ├── directions.npz              # v_cal[L,D], V_top[L,8,D], mu[L,D]
    ├── lda_directions.npz          # W_subj[L,D,7], v_lvl_lda, v_lvl_ridge
    ├── scores_calibrated.npy       # (N, L) PC1 projection
    ├── scores_topK.npy             # (N, L, 8)
    ├── diagnostics_per_layer.csv
    ├── linear_probe_per_layer.csv
    ├── supervised_per_layer.csv
    ├── plots/layer_window_compare.png
    └── critical/
        ├── REPORT_critical.md      ← Phase Critical
        ├── prompt_act_cache.npy    # (N, L, D) bf16 → 4.72 GB
        ├── prompt_act_meta.parquet
        ├── prompt_per_layer.csv
        ├── critical_compare.csv
        └── plots/
```

### 외부 의존
- Track A pass rates: `outputs/pass_rate_pilot_2666.parquet`
- Track B activation shifts: `activation/outputs/shifts/{id}.pt`
- Track B prompt activations (Critical): `outputs/prompt_act/{id}.pt`
- pilot universe metadata: `_nait_common.load_metadata(resolve_shift_dirs(None) + ['src/4.6_Task2/activation/full_shifts_l7l8'])`

---

*End of MASTER SUMMARY — curriculum 단계 정의는 사용자 지시 대기 중.*
