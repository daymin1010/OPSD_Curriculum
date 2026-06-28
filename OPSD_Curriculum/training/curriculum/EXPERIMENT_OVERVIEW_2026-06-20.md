# 실험 개요서: Activation-Guided Curriculum Learning for OPSD Self-Distillation

**작성일:** 2026년 6월 20일  
**상태:** Arm ②-A(diff-only) 학습 완료 / Arm ③-A(main) 학습 진행 중 / Eval 진행 중

---

## 1. 연구 배경 및 목적

On-Policy Self-Distillation (OPSD)은 동일 모델이 teacher(풀 정보 조건)와 student(문제만 조건)를 동시에 수행하며, student의 온라인 trajectory에서 token-level Jensen-Shannon Divergence (JSD) 분포 매칭을 학습 신호로 사용하는 방법론입니다. TRL GOLD Trainer 기반(arXiv 2601.18734)으로, 별도의 외부 teacher 모델 없이 수학 추론 능력을 자기증류(self-distillation)를 통해 향상시킵니다.

본 연구는 다음 질문을 다룹니다:
> **"OPSD 자기증류 훈련 시, 학습 예제를 난이도·과목 기반의 커리큘럼 순서로 배치하면 최종 수학 추론 성능이 향상되는가?"**

커리큘럼 순서 설계의 근거는 **실측 활성화 분석(activation analysis)**에서 도출하였습니다. Qwen3-8B 모델로부터 N=3,025개 수학 문제 풀이의 레이어별 잔차 스트림 변화량(THINKING Δ-activation)을 계산하여, 문제의 난이도·과목에 따라 모델 내부 표현이 어떻게 조직화되는지 분석하였습니다.

---

## 2. OPSD 원본 코드 구조

OPSD 원본 코드(`opsd_src/`)는 다음 파일들로 구성됩니다.

| 파일 | 역할 |
|------|------|
| `opsd_train.py` | 학습 진입점. `siyanzhao/Openthoughts_math_30k_opsd` 전체를 **순서 그대로** 로드 → `OPSDTrainer`에 전달합니다. |
| `opsd_trainer.py` | 핵심 self-distillation 트레이너. vLLM colocated 모드로 student rollout → teacher vs student JSD 산출 → GOLD loss 역전파합니다. |
| `data_collator.py` | student 입력(문제만) / teacher 입력(문제+정답) 분리 구성합니다. |
| `sft_train.py` / `grpo_train.py` | SFT / GRPO 비교 실험용 진입점입니다. |
| `accelerate.yaml` | multi-GPU Accelerate 설정 파일입니다. |
| `scripts/run_opsd_{1b,4b,8b}.sh` | 각 모델 크기별 공식 실행 런처입니다. |
| `eval/evaluate_math.py` | vLLM 기반 수학 평가 코드입니다. (본 연구에서 수정 없이 그대로 사용) |
| `eval/run_eval.sh` | 원본 평가 실행 스크립트입니다. |

### OPSD 원본 핵심 학습 흐름

```
1. 전체 데이터셋 (30,000행) → 별도 정렬 없이 로드
2. 각 step마다:
   a. student: <문제>만을 입력으로 vLLM 생성
   b. teacher: <문제+정답 solution>을 입력으로 병렬 scoring
   c. JSD(teacher_dist ‖ student_dist) 계산 (per-token, clip=0.05)
   d. GOLD loss = β·JSD + (1-β)·[-log P_student(token)]
   e. LoRA 가중치 역전파
```

### OPSD 원본 평가 설정 (`eval/run_eval.sh`)

| 항목 | 원본 설정 |
|------|---------|
| 평가 데이터셋 | **AIME 2024 단일** (`HuggingFaceH4/aime_2024`) |
| val_n | 12 |
| temperature | 1.0 |
| tensor_parallel_size | **4** (원본 서버 기준 GPU 4장) |
| checkpoint 평가 | step 25, 50, 75, 100 (원본은 단기 실험) |

---

## 3. 본 연구에서 수정/추가된 부분

OPSD 원본 코드(`opsd_src/`)는 **일체 수정하지 않고**, 별도의 커리큘럼 레이어(`curriculum/`)를 새로 개발하여 위에 얹었습니다.

### ✅ 유지된 부분 (그대로 상속)
- **OPSD 손실 함수**: JSD self-distillation loss, GOLD loss 구조 — **변경 없음**
- **Teacher/Student inference**: vLLM colocated, fixed teacher (LoRA merge/unmerge) — **변경 없음**
- **모델 로딩/저장, LoRA 설정, Wandb 로깅 구조** — **변경 없음**
- **평가 코드** (`evaluate_math.py`) — **수정 없이 그대로 사용**

### 🆕 추가/변경된 부분

| 파일 | 역할 | 변경 내용 |
|------|------|-----------|
| `train_opsd_curriculum.py` | 커리큘럼 학습 진입점 | `opsd_train.py`를 대체합니다. (6가지 차이) |
| `curriculum_schedule.py` | 데이터 스케줄 생성기 | phase-0 join + stage 배분 + 결정론적 순서 생성 (신규) |
| `curriculum_collator.py` | 데이터 콜레이터 | `stage_index` 컬럼을 통과시킵니다. (원본 collator 상속) |
| `curriculum_trainer.py` | 트레이너 | `SequentialSampler` 강제 적용합니다. (원본 OPSDTrainer 상속) |
| `curriculum_monitor.py` | 모니터 콜백 | `stage_respected` 게이트 검증합니다. (신규) |
| `opsd_data.py` | 데이터 로더 | Arrow 직접 읽기 방식으로 구현합니다. (datasets 버전 호환 문제로 신규 작성) |

**`train_opsd_curriculum.py`가 `opsd_train.py`와 다른 6가지 핵심 차이:**

```
① build_schedule() 호출 → 전체 T×B_glob 예제를 커리큘럼 순서로 사전 정렬합니다.
② dataset.select(schedule) → 정렬된 순서로 HuggingFace 데이터셋을 재구성합니다.
③ training_args.remove_unused_columns = False → stage_index 컬럼이 collator까지 전달됩니다.
④ CurriculumOPSDTrainer: SequentialSampler 강제 → 스케줄이 곧 학습 순서가 됩니다.
⑤ training_args.max_steps = T (=480) → T×B_glob 예제를 정확히 1회 통과합니다.
⑥ CurriculumMonitorCallback 추가 → 매 step에서 stage 무결성을 검증합니다.
```

### OPSD 원본 대비 평가 설정 변경 (의도된 2가지)

| 항목 | OPSD 원본 | 본 연구 | 이유 |
|------|---------|--------|------|
| tensor_parallel_size | 4 | **2** | L40S 2장 환경 (점수와 무관합니다.) |
| 평가 데이터셋 | AIME 2024 단일 | **AIME 2024 + AIME 2025 + HMMT Feb 2025** | 시간적 일반화 및 다양성 검증을 위해 확대하였습니다. |

*나머지 모든 eval 파라미터(val_n, temperature, thinking mode, top_p, max_model_len 등)는 원본과 동일합니다.*

---

## 4. 활성화 분석 주요 결과 (커리큘럼 설계 근거)

Qwen3-8B의 THINKING mode 추론 시 레이어별 Δ-activation(잔차 스트림 shift)을 N=3,025 샘플에서 분석하였습니다.

| 분석 항목 | 결과 |
|------|------|
| Level(난이도) 효과 | within-between gap **+0.434**, ρ(level) = **+0.84∼0.90**, out-of-sample ρ = **+0.937** |
| Subject(과목) 효과 | marginal +0.353 → level 통제 후 within-level **−0.04** (미미함) |
| 난이도 단조성 | Δlevel 단조성 ρ = **+0.893** |

**결론:** 난이도가 1차 분리 축이며, 과목은 난이도 통제 후 보조적 신호만 존재합니다. 따라서 커리큘럼의 주 축은 **난이도 순서**이며, 과목은 클러스터 혼합 보조 축으로 설계하였습니다.

### 과목 클러스터 결정 (K=4, average & complete linkage 양쪽 일치, cophenetic avg=0.720)

7개 과목('Other' 제외) → 4 클러스터:

| 클러스터 | 구성 과목 |
|----------|-----------|
| C1 | Intermediate Algebra, Precalculus |
| C2 | Counting & Probability, Number Theory |
| C3 | Algebra, Prealgebra |
| C4 | Geometry |

- 방문 순서(σ): **C1 → C4 → C2 → C3** (open-path cost 3.139)

---

## 5. 훈련 데이터

| 항목 | 세부 내용 |
|------|-----------|
| 데이터셋 | `siyanzhao/Openthoughts_math_30k_opsd` (HuggingFace) |
| 전체 규모 | 약 30,000행 |
| Set-A (훈련 대상) | 약 **28,771행** ('Other' 과목 제외 + label 매칭 ≥95% + 빈 셀 없음) |
| 레이블 | subject(8종), level(1∼8), difficulty(D1∼D4), subject_cluster(C1∼C4) |
| 난이도 축 | D1{level 1,2}, D2{level 3,4}, D3{level 5,6}, D4{level 7,8} |
| 조인 키 | sha1(problem_text)[:16] — 결정론적, match rate ≥ 0.95 gate PASS |

**Set-A 4×4 셀 구조 (difficulty × subject_cluster, 최소 셀 ≥ 79행, 빈 셀 0):**

| difficulty | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| D1 (level 1,2) | ✔ | ✔ | ✔ | ✔ |
| D2 (level 3,4) | ✔ | ✔ | ✔ | ✔ |
| D3 (level 5,6) | ✔ | ✔ | ✔ | ✔ |
| D4 (level 7,8) | ✔ (≥79) | ✔ (≥79) | ✔ (≥79) | ✔ (≥79) |

*(정확한 셀별 수치는 phase-0 실행 후 `outputs/REPORT_join_setA.md` 참조)*

---

## 6. 실험 설계 (Arm 비교)

두 arm을 동일 하이퍼파라미터·동일 총 학습 step으로 학습하여 직접 비교합니다.

| Arm | ID | 커리큘럼 구조 | Stage 수 | 순서 |
|-----|----|-------------|---------|------|
| **Diff-only** | ②-A | 난이도 축만 사용 | **4** | D1 → D2 → D3 → D4 |
| **Main** | ③-A | 난이도 × 과목 클러스터 | **16** | 16-stage snake (아래 참조) |

### 16-Stage Snake 구조 (Main arm)

```
D1: C1 → C4 → C2 → C3
D2: C3 → C2 → C4 → C1   (D 경계에서 reversal)
D3: C1 → C4 → C2 → C3
D4: C3 → C2 → C4 → C1   (D 경계에서 reversal)
```

총 16 cell × (480 step ÷ 16) = **stage당 30 optimizer steps**

---

## 7. 모델 및 학습 하이퍼파라미터

### 7-1. 베이스 모델 및 어댑터

| 항목 | 값 |
|------|-----|
| 베이스 모델 | `Qwen/Qwen3-8B` |
| dtype | bfloat16 |
| 어텐션 | Flash Attention 2 |
| 파인튜닝 방식 | **LoRA** (PEFT) |
| LoRA rank (r) | **64** |
| LoRA alpha | **128** |
| LoRA 대상 모듈 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7개) |

### 7-2. OPSD 학습 파라미터

| 항목 | 값 |
|------|-----|
| Learning rate | 5.0 × 10⁻⁶ |
| Max grad norm | 0.1 |
| Total optimizer steps (T) | **480** |
| Per-device batch size | 2 |
| Gradient accumulation steps | 8 |
| World size (Data Parallel) | 2 GPU |
| **Effective global batch size (B_glob)** | **2 × 8 × 2 = 32** |
| Gradient checkpointing | True |
| Max completion length | 1,024 tokens |
| Max sequence length | 20,000 tokens |
| β (JSD 가중치) | 0 (forward KL 방향) |
| λ (lambda) | 1 |
| JSD token clip | **0.06** (원본 기본값 0.05에서 조정) |
| Teacher 방식 | **Fixed teacher** (`--fixed_teacher True`, step 0 LoRA frozen) |
| Sampling temperature (학습 시) | 1.1 |
| Top-p / Top-k (학습 시) | 0.95 / 20 |
| vLLM mode | Colocated, GPU memory utilization 0.6 |
| student_thinking | False |
| teacher_thinking | True |
| attach_gold | True (monitor-only 정답률 측정, 손실에는 영향 없음) |

### 7-3. 커리큘럼 스케줄링

| 항목 | 값 |
|------|-----|
| Stage당 optimizer step budget | round(T / num_stages) |
| Stage 내 샘플링 방식 | Seeded shuffle + cycling (seed=42, 결정론적) |
| 스케줄 무결성 검증 | `stage_distinct > 1` abort guard (CurriculumMonitorCallback) |
| 데이터 순서 주입 방식 | SequentialSampler + CurriculumIndexDataset |

### 7-4. 학습 인프라

| 항목 | 값 |
|------|-----|
| 하드웨어 | NVIDIA H200 × 2 (1노드, iREMB-C-03) |
| GPU 메모리 | 141 GB × 2 |
| 병렬화 | HuggingFace Accelerate (`num_processes=2`) |
| 체크포인트 | 매 30 step 저장, `save_total_limit=3` (최신 3개 유지) |
| Walltime 한도 | 18시간 |

---

## 8. 평가(Evaluation) 설정

OPSD 원본 eval 하네스(`evaluate_math.py`)를 **수정 없이** 그대로 적용합니다.

| 항목 | 값 | OPSD 원본 대비 |
|------|-----|-------------|
| Thinking mode | **ON** (`enable_thinking=True`) | 동일 |
| 샘플 수 | **val_n = 12** → Average@12 집계 | 동일 |
| Temperature | **1.0** | 동일 |
| Top-p | 0.95 | 동일 |
| Top-k | −1 | 동일 |
| Min-p | 0.0 | 동일 |
| max_new_tokens | 38,912 | 동일 |
| max_model_len | 40,960 | 동일 |
| GPU memory utilization | 0.9 | 동일 |
| Tensor parallel size | **2** | 원본 4 → **2** (L40S 2장, 점수 무관) |
| LoRA 로딩 방식 | `--checkpoint_dir` → `LoRARequest` | 동일 |
| 평가 데이터셋 | **AIME 2024, AIME 2025, HMMT Feb 2025** | 원본 AIME 2024 단일 → **3종으로 확대** |
| 채점 방식 | `math_verify` (boxed answer 추출 후 symbolic 등가 검증) | 동일 |
| 평가 인프라 | NVIDIA L40S × 2 (iREMB-C-07, partition l40sq) | — |
| 결과 형식 | JSON 저장 (wandb 미사용) | — |

### OPSD 원본 논문의 평가 벤치마크

원본 `run_eval.sh`에서는 Qwen3-1.7B를 학습한 후 step 25/50/75/100에서 **AIME 2024만** 평가합니다. 본 연구에서는 아래 3종의 벤치마크를 사용합니다.

| 평가 데이터셋 | 내용 | HuggingFace ID |
|--------|------|----------------|
| **AIME 2024** | American Invitational Mathematics Examination 2024 | `HuggingFaceH4/aime_2024` |
| **AIME 2025** | American Invitational Mathematics Examination 2025 | `yentinglin/aime_2025` |
| **HMMT Feb 2025** | Harvard-MIT Mathematics Tournament, February 2025 | `MathArena/hmmt_feb_2025` |

> **HMMT Feb 2025**는 AIME보다 난이도가 높고 다양한 수학 분야를 포함하고 있어, AIME 계열과 함께 경쟁 수학(competition math) 추론 능력을 다각도로 측정하는 용도로 사용합니다.

---

## 9. 평가 지표

| 지표 | 설명 |
|------|------|
| **Average@12** (`average_at_n_pct`) | 12개 샘플의 정답 여부 평균입니다. |
| **Pass@12** (`pass_at_n_pct`) | 12개 중 적어도 1개가 정답인 비율입니다. |
| **Majority Vote@12** (`majority_vote_at_n_pct`) | 다수결 방식의 정답률입니다. |
| **Format rate** (`format_rate`) | `\boxed{}` 형식 준수율입니다. |

---

## 10. 비교 조건 및 진행 현황 (2026-06-20 기준)

| 조건 | 설명 | 상태 |
|------|------|------|
| **Diff-only (②-A)** | 난이도 4단계 순차 커리큘럼 | ✅ 학습 완료 (step 480) |
| **Main (③-A)** | 난이도 × 과목 16-stage snake 커리큘럼 | 🔄 진행 중 (~step 176/480, 완료 예상 6/20 야간) |
| Base (선택적) | 커리큘럼 없는 순수 Qwen3-8B + OPSD (기준선) | ⬜ 미제출 |

**최종 비교표 (결과 집계 후 작성 예정):**

| Dataset | Diff-only Avg@12 | Main Avg@12 | (Base Avg@12) |
|---------|-----------------|-------------|---------------|
| AIME 2024 | — | — | — |
| AIME 2025 | — | — | — |
| HMMT Feb 2025 | — | — | — |

---

*참고 문헌: arXiv 2601.18734 (On-Policy Self-Distillation for Reasoning, Zhao et al. 2025)*
