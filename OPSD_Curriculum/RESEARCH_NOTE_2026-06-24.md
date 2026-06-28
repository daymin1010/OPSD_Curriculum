# OPSD 2축 커리큘럼 연구 정리 (2026-06-24)

> On-policy self-distillation(OPSD)에 **난이도 + subject 활성 기하** 2축 커리큘럼을 얹어, 난이도-only baseline 대비 SOTA를 만드는 연구. 이 문서는 ① 커리큘럼 구성 method 2버전(왜·어떻게 바꿨는지) ② 실험 종류와 결과를 체계적으로 정리한다.

---

## A. 연구 개요 & 동기

- **OPSD (on-policy self-distillation)**: 한 모델이 teacher와 student를 겸한다. teacher는 ground-truth 정답을 조건으로 받고, student는 문제만 보고 on-policy rollout을 생성한다. teacher가 student의 rollout에 **token-level dense supervision**을 준다.
- **comprehension threshold**: 문제가 teacher가 정답으로부터 rationalize할 수 있는 한계를 넘으면 supervision 신호가 degrade한다 → **커리큘럼 학습**의 동기.
- **연구 질문**: 난이도(difficulty)는 라벨로 이미 정렬 가능하다. 그렇다면 **subject(topic)가 두 번째 축**으로 유효한가? 그리고 그 구조를 **라벨이 아니라 모델 내부 활성 기하(activation geometry)**에서 읽을 수 있는가?
- **핵심 차별점**: subject–subject 거리를 카테고리 라벨이 아니라 **activation residual geometry**에서 추출 → 난이도와 직교하는 두 번째 축으로 사용.
- **타깃**: 경시 수학 AIME24/25, HMMT25 (+ MATH-500), Qwen3-8B, non-thinking eval.

---

## B. Method — 커리큘럼 구성 ★핵심

### B-1. 공통 재료
- **unit** = (subject × level) cell, 예: `Algebra|L3`. (subject 7종, level 1–8)
- **활성 출처**: Qwen3-8B **thinking 모드** activation shift, NAIT-thinking span, **faithful(DAF) 폐기**, pooled pilot1+pilot2 **N=3025**.
- **난이도 직교화**: per-level 평균을 빼서 residualize → subject 기하만 남김 ($\tilde a_i = \Delta a_i - \mu_{\ell(i)}$).
- **cluster**: `C_alg = {Algebra, Intermediate Algebra, Precalculus}`, `C_geo = {Geometry}`, `C_disc = {Counting&Probability, Number Theory, Prealgebra}`.
- universe: Set-A (OpenThoughts-Math-30K), **N = 28,771** (include_other=False).

### B-2. v1 (tiered) — 첫 시도
- method id: `tiered_difficulty_backbone_residual_within_tier`
- 구성: units를 난이도로 정렬 → **n_tiers개 equal-mass tier**로 분할 → **각 tier 내부를 subject residual nearest-path로 재정렬** → flatten 후 5 stage equal-mass 분할.
- gate: `ρ(diff, ours) ∈ [0.4, 0.7]` + 단조성 → **n_tiers = 2** 선택.
- **결과: 난이도-only(diff)에 짐** (아래 D 참조).

### B-3. 왜 v2로 갔나 — 차별화 추구와 표현 도약(jump)의 트레이드오프
v1은 **subject가 난이도와 구별되는 독립 축임을 증명**하기 위해, diff와 의도적으로 다른 stage 구성을 만든 설계였다 ($\rho(\text{diff}, \text{ours})$를 0.4–0.7로 통제 — "난이도 순서와 같지 않게"). 이 차별화 자체는 성공했지만, subject를 강하게 가르는 쪽으로 밀다 보니 다음이 관찰됐다:

| 관찰 | 내용 |
|---|---|
| 전체표현 점프 | v1 ours의 **인접 stage 표현 도약이 diff보다 큼** (0.394 vs 0.237, W_ALL) → "smooth transition" 의도와 어긋남 |
| 난이도 분포 부수효과 | tier가 거칠어(n_tiers=2) **stage2가 level 1–8 광폭**(var 1.31), "level 2–3 마스터 stage"가 흐려짐 |
| full step900 점수 | ours < diff (AIME24 avg 50/90 vs 60/90) |
| MATH-500 분해 | level 2–3 −8~−11%p, Intermediate Algebra −9.3%p (쉬운-중간 구간 손해); 단 마지막 stage subject Geometry도 −2.4%p라 **forgetting은 아님** |

→ 즉 v1은 "차별성"을 얻었지만 **표현 도약이 너무 커졌다.** v2의 동기는 **차별성(subject 축)은 유지하되 인접 stage 간 표현 도약을 줄이는 것.**

### B-4. v2 (subjslack) — 재설계 — "어떻게"
method id: `level_backbone_residual_subject_slack`. 난이도를 **단조·tight backbone**으로 고정하고, subject는 같은 난이도대 안에서만 재배치.

**핵심 식**:
$$\rho(x) = \ell(x) + \alpha \cdot g(s(x))$$

- $g(s) \in [-0.5, +0.5]$: subject 기하 좌표. residual centroid 유사도 행렬의 **leading axis**(classical MDS, 분산 35.4%)를 subject별로 집약·정규화.
  - 값: `Precalc −0.50, IntAlg −0.45, Algebra −0.20, Geometry −0.12, Prealgebra +0.21, NumberTheory +0.33, C&P +0.50`
  - 해석: **g<0(C_alg) → 이른 stage, g>0(C_disc) → 늦은 stage, Geometry 중립.**
- $\alpha = 2.0$: subject가 ~1 level 분량만 이동 → backbone 단조 유지.
- **unit-atomic 분할**: unit을 $\rho$로 정렬 → 쪼개지 않고 mass 균형으로 5 stage 분할 (stage 크기 자연 변동).

**검증 (오프라인, full 28,771 기준)**:

| 지표 | diff | v1 ours(tiered) | **v2 ours(subjslack)** |
|---|---|---|---|
| 단조 min_diff | +0.97 | −0.15 | **+0.86** |
| mean per-stage level var | 0.13 | 1.31 | **0.49** |
| cond5 분리 (perm p) | — | — | **p=0.005** |
| 전체표현 점프 (W_ALL) | 0.237 | 0.394 | **0.226** (<diff) |
| universe md5 (=diff) | 3f54d1a51c71 | — | 3f54d1a51c71 ✓ |

→ v2는 ① 난이도 단조 회복(성능) ② cond5 통계 분리(subject 2축 입증) ③ 점프 < diff(논문 주장 실현)를 **동시에** 달성.

**v2 stage 구성 (α=2.0, unit-atomic)**:

| stage | 문제 수 | 평균 난이도 | level 범위 |
|---|---|---|---|
| 0 | 5,323 | 1.90 | L1–3 |
| 1 | 5,365 | 3.04 | L2–4 |
| 2 | 6,171 | 3.90 | L3–5 |
| 3 | 5,775 | 4.89 | L4–6 |
| 4 | 6,137 | 5.97 | L5–8 |

메커니즘 예: **L3 난이도**가 subject 기하로 갈림 → Precalc/IntAlg L3는 stage 0, C&P/NumTheory L3는 stage 2. 난이도는 단조, 같은 난이도대를 **대수→기하→이산/정수론** 순으로 배치.

### B-5. 비교군 (controls)
- **cond2_diff**: 난이도-only ($\alpha=0$).
- **cond3_ours**: 위 2축.
- **cond5_diffmatched**: ours의 per-(level,stage) 분포는 동일, level 내 subject를 **랜덤**. → ours가 cond5를 이기면 **subject 기하가 난이도 너머의 효과**임이 입증됨 (리뷰어 방어의 load-bearing).

---

## C. 실험 설계

- **스케일 ladder** (학습량 효과 분리): mini50(N≈1,600, T≈50) / mini100(N≈3,200, T≈100) / q4(N≈7,193, T≈225) / full(N=28,771, T≈900).
- **arm**: cond2_diff / cond3_ours / cond5_diffmatched.
- **eval**: AIME24/25(30문제×3), HMMT25(30×3), MATH500(500×1), non-thinking, temp 1.0, TP=2. 지표 pass@n, avg@n.
- **공정성**: 각 rung에서 diff·ours가 **완전히 동일한 problem 집합**(stratified_pick 1회 후 양 arm 적용, md5 일치). stage 배정만 다름.
- 학습 설정 (공통): Qwen3-8B + LoRA r=64/α=128, lr 5e-6, B_glob=32, JSD token clip 0.06, teacher_thinking=True, student_thinking=False, 1 pass.

---

## D. 결과

### D-1. tiered final eval — 맞힌 개수 기준
표기: `pass@n = 30문제 중 맞힌 문제`, `avg@n = (문제×시도) 중 맞힌 시도`.

**pass@n (30문제 중 / MATH500은 500중)**

| config | aime24 | aime25 | hmmt25 | math500 |
|---|---|---|---|---|
| base | 12/30 | 10/30 | 5/30 | 423/500 |
| mini100 diff | 12/30 | 13/30 | 7/30 | 419/500 |
| mini100 ours | 12/30 | 11/30 | **9/30** | 422/500 |
| q4 diff | **15/30** | **14/30** | 7/30 | **394/500** |
| q4 ours | 13/30 | 11/30 | 7/30 | 387/500 |
| full900 diff | **25/30** | **22/30** | 11/30 | **415/500** |
| full900 ours | 24/30 | 20/30 | 11/30 | 398/500 |

**avg@n (90시도 중 / MATH500은 500중)**

| config | aime24 | aime25 | hmmt25 | math500 |
|---|---|---|---|---|
| base | 23/90 | 20/90 | 10/90 | 423/500 |
| mini100 diff | 29/90 | **25/90** | 11/90 | 419/500 |
| mini100 ours | 28/90 | 22/90 | **14/90** | 422/500 |
| q4 diff | 27/90 | **25/90** | 10/90 | **394/500** |
| q4 ours | 26/90 | 22/90 | **13/90** | 387/500 |
| full900 diff | **60/90** | **38/90** | 21/90 | 415/500 |
| full900 ours | 50/90 | 38/90 | **22/90** | 398/500 |

### D-2. 패턴 (전 스케일 일관)
- **AIME(중간 frontier): diff ≥ ours** — q4·full에서 뚜렷 (q4 aime24 15 vs 13문제, full900 avg 60 vs 50/90).
- **HMMT(극한 frontier): ours ≥ diff** — avg 기준 거의 전 스케일 ours 우위 (mini100 14 vs 11, q4 13 vs 10, full 22 vs 21/90).
- **MATH500: diff 약간** (q4 394 vs 387, full900 415 vs 398).
- 학습량 순 상승: base < mini50≈base < mini100 < q4 < full900 (mini50은 noise).
- → 기존 tiered ours의 한계(**AIME 약점**)가 모든 스케일에서 재현. 새 subjslack의 AIME 해소 여부가 다음 관문.

### D-3. 분해 (full step900, MATH-500)
- **level**: ours가 L2 −11%p, L3 −8.6%p (손해), L5 +2.2%p (이득).
- **subject**: IntAlg −9.3%p, NumberTheory +1.6%p, Geometry(마지막 stage) −2.4%p.
- → tiered ours는 **쉬운-중간 손해 / 극한 이득**. 이게 AIME(손해)·HMMT(이득) 패턴과 일치. forgetting 아님(난이도 구조 문제).

### D-4. subjslack 오프라인 검증
- B-4 표 참조: 단조 회복, varL 1.31→0.49, cond5 p=0.005, 점프 0.226<0.237. 세 목표 동시 충족.

### D-5. tiered step-curve — q4·mini100 (2026-06-25)
중간 checkpoint를 평가해 step별 동역학 확인. 값 = **avg@n** (aime/hmmt = 맞힌시도/90, math500 = /500).

**q4 step-curve** (40·90·130·180·225):

| arm·step | aime24 | aime25 | hmmt25 | math500 |
|---|---|---|---|---|
| diff 40 | 24 | 17 | 11 | 415 |
| diff 90 | 22 | 21 | 10 | 413 |
| diff 130 | 24 | 20 | 13 | 405 |
| diff 180 | **31** | 17 | 11 | 402 |
| diff 225 | 27 | **25** | 10 | 394 |
| ours 40 | 20 | 17 | 9 | 418 |
| ours 90 | 24 | 18 | 10 | 404 |
| ours 130 | 21 | 20 | **14** | 399 |
| ours 180 | 25 | 22 | 9 | 398 |
| ours 225 | 26 | 22 | 13 | 387 |

**mini100 step-curve** (40·80·100):

| arm·step | aime24 | aime25 | hmmt25 | math500 |
|---|---|---|---|---|
| diff 40 | 24 | 21 | 11 | 419 |
| diff 80 | 29 | 22 | 14 | 419 |
| diff 100 | 29 | 25 | 11 | 419 |
| ours 40 | 25 | 21 | 10 | 409 |
| ours 80 | 26 | 18 | 11 | 414 |
| ours 100 | 28 | 22 | 14 | **422** |

**핵심:**
1. **MATH500 trade-off (스케일 의존)**: q4는 step↑에 MATH 단조 감소(diff 415→394, ours 418→387) = **후반 over-train**. mini100은 유지(diff 419)~증가(ours 409→422). → q4 후반(180·225)이 손해 구간, mini100(100)은 손해 전.
2. **AIME는 후반 학습이 도움**: q4 step225·mini100 step100에서 최고. 단 **AIME↑와 MATH↓가 같은 후반 구간에서 동시 발생**(trade-off).
3. **ours vs diff (패턴 유지)**: AIME diff≥ours(q4), HMMT ours≥diff(근소), MATH는 q4=diff·mini100=ours. **tiered ours가 AIME 약점을 여전히 못 넘음.**
4. **최적 step 시사**: q4는 ~130(MATH 손해 전 + AIME 적당)이 분기점, mini100은 100까지 손해 없이 개선.
5. **H2 전제 지지**: stage마다 학습 이득이 다름(q4 stage4=aime↑/math↓, mini100은 고른 개선) → "rationalizability가 stage별 이득 설명"을 측정할 토대 확보.
   - *주의*: AIME/HMMT는 90 trial 기준 ±5% noise. 추세 위주로 해석.

### D-6. (진행 중)
- **full subjslack diff/ours** 학습 (C-03 H200, save_steps=50 → step 100~900 곡선, ~20h, 현재 ~13.5h 경과). 학습 후 step별 eval로 "전체 stage 완주가 최선인가 + AIME 약점 해소 여부 + cond5 격파" 검증 예정.

---

## E. 인사이트 / 논의

1. **OPSD "100-step 피크"는 랜덤 학습 특성**: 원본은 30 epoch(28,140 step) 돌리되 ~100 step에서 성능 피크(README). 이는 **랜덤 셔플**이라 11% 데이터로도 전 난이도·subject를 대표하기 때문. **우리 순차 커리큘럼은 다르다** — step 100은 stage 0(쉬운 것)만 본 시점이라, **전체 stage 완주(T 근처)가 본질적으로 필요**. early stopping은 커리큘럼을 무력화한다.
2. **ours 강약점이 난이도 구조로 일관 설명**: 비단조·광폭 난이도 → 쉬운-중간 손해(AIME), 극한 이득(HMMT). v2(subjslack)는 단조를 회복해 AIME 약점 해소를 노린다.
3. **향후 핵심 검증**: subjslack ours가 (a) AIME에서 diff를 따라잡/넘는가, (b) **cond5를 이기는가** ← subject 기하 효과 입증의 핵심.

---

## F. 향후 방향 (조건부 — main이 diff를 이긴 뒤)

> **현 우선순위는 "OPSD 기본 위에서 subjslack ours가 diff를 이기게 하는 것"**이다. 아래는 교수님의 "왜 하필 OPSD인가" 지적에 답하는 motivation 보강 방향으로, ours > diff가 확인된 다음 단계.

### F-0. 현재 즉시 작업
- **full subjslack 체크포인트 곡선**(`save_steps=50`, step 100~900) → **최적 스텝 몇 개 + ours vs diff** 확인.
- 이게 "OPSD 기본 + 표현 기반 커리큘럼"의 성패를 가르는 1차 관문.

### F-1. Rationalizability를 주인공으로 (서사 재편)
- "왜 OPSD인가"의 답: OPSD는 **teacher 실력 < 문제 난이도면 teacher가 정답 $y^*$를 rationalize 못 해 신호가 degrade**하는 고유 한계가 있다. 커리큘럼은 이 한계를 다루는 처방으로 위치 → SFT/GRPO로 대체 불가.
- 인과 재정리: **표현은 rationalizability의 원인이 아니라 측정 좌표.** 표현 → (난이도·도메인) 위치 → teacher 대비 rationalizability → 학습 이득.
- 검증 가설:
  - **H1 (측정)** 표현(g·level)이 rationalizability를 예측.
    - **1단계 (완료, 2026-06-24, generation 0)** — *측정법*: base Qwen3-8B에 **student 포맷 `[문제만]`** 입력 + reference solution(OPSD dataset의 풀이)을 **forced-decode**(vLLM `prompt_logprobs`, 생성 없이 1 forward)하여 **solution 토큰들의 평균 NLL**을 측정. 이 NLL은 *"base가 그 정답 풀이를 얼마나 자연스럽게 따라 읽는가(친숙도)"* — 낮을수록 친숙. pilot 3,000문제(g를 계산한 동일 모집단), unit(subject×level)별 집계 후 g·level과 Spearman.
      - *결과*: **NLL vs g = +0.253 (p=6e-41)**, NLL vs level = +0.005 (**p=0.78, 무관**), **난이도(level) 통제 후에도 NLL vs g = +0.257 (p=3e-42)**. per-subject 평균 NLL이 g 순서와 일치: C_alg(Precalc 0.72·IntAlg 0.76·Algebra 0.83, 친숙) → Geometry 0.94(중립) → C_disc(Prealgebra 0.85·NumberTheory 0.89·C&P 0.98, 낯섦).
      - *의미*: subject 기하 g가 **난이도와 직교**인 "subject별 base 친숙도"를 강하게 인코딩한다. **난이도 라벨(level)은 이 친숙도를 전혀 못 잡는데(p=0.78) g는 잡는다** → 표현 기반 subject 축이 난이도가 놓치는 정보를 보완한다는 직접 증거 = **2축 커리큘럼 정당화**.
      - *한계*: 이는 forced-decode "읽기 친숙도"(student 관점)이지 teacher가 해답 보고 정답을 *생성*하는 **rationalizability(생성 능력)는 아니다.** 또 NLL 높음이 "이해 어려움"인지 "표기·스타일이 base에 생소함"인지는 구분 못 함. → 2단계로 확장 필요.
      - *산출물*: `reasoning_pivot/activation/analysis/rationalizability_nll.py`, 출력 `rationalizability_nll_out/{per_problem_nll.parquet, by_subject.csv, by_unit.csv}`.
    - **2단계 (B안, 미실시)**: teacher 포맷 `[문제+해답+transition]`으로 정답을 *생성*시켜 unit별 정답률(소규모 N≈20) → "teacher 해답-조건부 rationalizability". ours>diff 확인 후 진행.
  - **H2 (설명)** stage별 rationalizability가 stage별 학습 이득(체크포인트 곡선 증가분)을 예측.
  - **H3 (처방)** rationalizability가 유지·상승하도록 stage를 배치(=우리 커리큘럼)하면 이득 증가. ours vs diff vs cond5가 이 답.
- 공통 도구 = **체크포인트 곡선** (F-0에서 이미 확보).

### F-2. Stage-wise Teacher Upgrade (처방)
- stage k 학습 종료 → 그 checkpoint를 stage k+1의 **teacher(고정) + student 초기값**으로 갱신. teacher가 난이도와 함께 강해져 **고난도 구간까지 rationalizability 유지**.
- OPSD 전용: "teacher를 단계적으로 키운다"는 *teacher가 정답을 보는* OPSD에서만 성립 → "왜 OPSD" 직접 답.
- 위험: iterative self-improvement화 → 한 단계 오류가 다음 teacher로 누적(collapse). **갱신 빈도(매 stage / 2 stage / 없음)·teacher 정의(직전 / EMA)가 연구 질문.**
- ★**시너지(우리 method가 이걸 떠받침)**: subjslack의 **표현 점프 최소화**(0.226 < diff)는 갱신된 teacher가 **표현상 인접한 다음 stage만 감당**하면 되게 한다 → teacher 갱신을 안전하게(분포 이동·오류 누적 작게) 만든다. 즉 "표현 기반 순서 + OPSD teacher 갱신"이 **필연적으로 결합**.

### F-3. 실행 순서
1. **(현재)** full subjslack 체크포인트 곡선 → 최적 스텝 + ours vs diff (+ cond5)
2. **ours > diff 확인 시** → ① rationalizability 파일럿(H1: teacher 정답 NLL vs g·level) ② teacher-upgrade prototype + 갱신빈도 ablation
3. 운영: step은 데이터 축소 아닌 **max-step**으로만 조절(형평성), non-thinking eval, 전 checkpoint 평가.

---

## 부록: 파일/경로
- **v2 빌드·문서**: `training/stages_subjslack_20260624/`
  - `build_stages_subjslack.py` (재현 빌드), `make_scales_subjslack.py` (subsample), `measure_fulljump.py` (점프 측정)
  - `stages_cond3_ours_subjslack.json` (+ q4/mini50/100/150), `stages_cond2_diff*.json`, `stages_cond5_diffmatched_seed{0,1,2}.json`
  - `REPORT_stagebuild_subjslack_2026-06-24.md` (상세), `METHOD_subject_geometry_curriculum.md` (논문 method draft), `manifest.json`, `g_subject_axis.json`
- **v1**: `training/stages_tiered_20260622/`
- **학습 config/sbatch**: `training/curriculum/configs/*_subjslack.yaml`, `training/curriculum/sbatch/*_subjslack_h200.sh`
- **eval 결과**: `outputs/eval_opsd_curriculum/`
- **활성/기하 분석**: `reasoning_pivot/activation/analysis/stagebuild.py`, `analysis_qwen3_8b/`
