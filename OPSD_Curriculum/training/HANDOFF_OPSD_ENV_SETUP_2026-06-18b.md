# Hand-off — OPSD 커리큘럼: GPU 구성 확정 + B_glob=32 게이트 통과 (착수 후속)
작성: 2026-06-18 (b)
선행 정본(반드시 같이 읽기):
- `HANDOFF_CURRICULUM_LEARNING_2026-06-18.md` (분석 결론·클러스터·stages·OPSD 코드 안내 정본)
- `HANDOFF_OPSD_ENV_SETUP_2026-06-18.md` (env 설치 단계 정본)
이 파일은 그 후속으로 **GPU 구성 확정 + B_glob=32 정합 + 다음 구현 지시**만 기록.

## 0. 한 줄 요약
- 사용자 디렉션 확정: **GPU 구성·B_glob·reward_proxy·검증 절차** 모두 결정됨.
- `verify_schedule.py` 기본 B_glob 16→**32** 변경 + **재실행 ALL PASS** (opsd env로 검증 완료).
- 다음 세션: opsd_src 통독 후 **collator passthrough / monitor / trainer / config×4 / sbatch** 구현 → smoke 제출.

## A. ★사용자 확정 디렉션 (이번 세션에서 합의·고정)
1. **GPU = 한 번에 4장 상한.** 한 SLURM job = 단일 노드·단일 GPU 종류 (H200+L40S 혼합 불가).
   - 두 종류 **각각 4장(ws=4)** 으로 별개 job. L40S(C-07) 4장 / H200(C-03) 4장.
   - ★ 사용자 선호: **L40S 우선**(H200은 경쟁 심함). H200은 폴백/교차검증용.
2. **B_glob = 32 고정** (= OPSD 원본 effective batch와 동일).
   - OPSD 원본: `run_opsd_1b.sh` num_processes=4·pd4·ga2 → 4·4·2=**32**; `run_opsd_8b.sh` num_processes=8·pd2·ga2 → 8·2·2=**32**.
   - 우리는 ws=4 통일이므로: **1.7B = pd4·ga2·ws4 = 32**, **8B = pd2·ga4·ws4 = 32** (8B는 ga로 보충).
3. **★8B를 L40S(46GB)에서 돌아가는지 반드시 확인** (feasibility의 핵심 목표). H200(141GB)은 여유라 보조.
4. **reward_proxy = 옵션 C**: smoke(1.7B)는 **loss + completion length만**으로 green. `rollout_acc`(\\boxed vs gold Answer) 배선은 **8B full 직전**에 추가 (smoke 막지 않음). 전부 monitor-only, OPSD generation 경로 미수정.
5. **smoke는 반드시 ws=4** (1-GPU false-pass 금지). schedule-respected 진짜 테스트.
6. 양쪽 노드(L40S/H200) 각각에 smoke 제출 = env/커널 호환까지 교차검증 (사용자 "각 종류 GPU에서 도는지").
7. GPU 혼잡 → **코드 견고히 짜서 sbatch 제출만 해두고 큐에 맡김.** `--exclusive` 금지, 본인 JOBID/NAME만, 산출물 `personal/jimin_2782/` 내부만.

## B. ★이번 세션 완료 (검증됨)
- `curriculum/verify_schedule.py`: argparse 기본 `--B_glob` 16→**32** 변경.
- **재실행 결과 ALL PASS** (opsd env: `envs/opsd/bin/python`):
  - Phase-0 join: n_total=29434, n_setA=28771, unmatched=0, other=663, match_rate=1.0, 16 main cells (min=79, empty=0), 4 diff cells — 전부 frozen ref와 일치.
  - schedule T=480, B_glob=32: diffonly len=15360(4×120×32), main len=15360(16×30×32). 모든 stage pool 비지 않음.
  - ★ 알려진 over-recycle (defer): main **stage12 [D4|C3] pool=79 → cycles=13**, stage0/15 [D1|C1·D4|C1] cycles=4, diffonly stage3 [D4] cycles=3. (선행 §G: L8/D4 단독결론 금지, 그대로 인지만.)

## C. ★다음 세션 구현 순서 (opsd_src 통독 필수 — 추측 금지)
**먼저 읽을 것** (class/함수명·collator 시그니처 확정용):
- `opsd_src/opsd_train.py` (데이터셋 로딩·collator 인스턴스화·trainer 생성·TrainingArguments)
- `opsd_src/opsd_trainer.py` (Trainer 서브클래스명, `training_step` 시그니처)
- `opsd_src/data_collator.py` (collator 클래스명; `__call__`이 feature["problem"]/["solution"] 읽음 확인)
- ※ `remove_unused_columns`는 opsd_src 전체 **0건** (grep 확인). 즉 upstream은 TRL 기본값 의존. 우리는 **명시적으로 False 강제**(config + train_opsd_curriculum 방어코드)할 것.

**구현 (전부 additive, opsd_src 직접수정 금지 → curriculum/ 내 fork):**
1. `curriculum/curriculum_collator.py`: OPSD collator 서브클래스. `batch = super().__call__(features)` 후 원본 feature에서 `stage_index`(및 8B용 gold `Answer`) 뽑아 `batch["stage_index"]` 텐서로 부착.
2. `curriculum/train_opsd_curriculum.py` (placeholder 재작성):
   - schedule 빌드(`curriculum_schedule.build_schedule`, arm·T·B_glob=32·seed) → CurriculumIndexDataset (SequentialSampler가 schedule order 그대로).
   - 데이터셋에 **stage_index 컬럼** 부여 (schedule position→stage. 공식: `slots_per_stage = budget*B_glob`; `stage_order=[s["stage_index"] for s in meta["stages"]]`; `stage_per_pos[i]=stage_order[i//slots_per_stage]`).
   - **`training_args.remove_unused_columns = False` 강제.**
   - curriculum_collator 주입, CurriculumOPSDTrainer 사용.
3. `curriculum/curriculum_trainer.py` (placeholder 재작성): `CurriculumOPSDTrainer(OPSDTrainer)`:
   - `training_step(model, inputs, num_items_in_batch=None)`: `sidx = inputs.pop("stage_index")` → modal·distinct-stage-count 기록 → **반드시 `return super().training_step(model, inputs, num_items_in_batch)`** (loss/generation body 재구현 금지).
4. `curriculum/curriculum_monitor.py`: Callback. 매 optimizer step에서 wandb 로깅:
   - `stage_batch_modal`(batch 내 최빈 stage), `stage_expected`(global_step→ `g//budget`→stage_order), `stage_respected`(modal==expected → 1/0), grad-accum 윈도우 distinct-stage-count(micro-leak 가시화).
   - **hard-fail 규칙**: 어떤 step이라도 modal≠expected → 로그에 `[SMOKE-FAIL]` 마킹. (DDP SequentialSampler shard가 contiguous-chunk라 OK일 것; 만약 섞이면 rank별 interleaved no-shuffle sampler로 교정 — 이 harness 최약점, 최우선 디버그.)
5. `curriculum/configs/`: yaml ×4 (OPSD 하이퍼 미러: lr5e-6, max_grad_norm0.1, gen1024, max_length20000, beta0, vLLM colocate util0.6 tp1, LoRA r64/a128, temp1.1/top_p0.95/top_k20, lmbda1, fixed_teacher, jsd_clip 0.05(1.7B)/0.06(8B), **remove_unused_columns:false**):
   - `smoke_1p7b_l40s.yaml` (pd4·ga2·ws4=32, max_steps 작게 e.g. T=48 → diffonly budget12/main budget3, num_train_epochs 대신 max_steps).
   - `smoke_1p7b_h200.yaml` (동일, 노드만 다름).
   - `full_8b_l40s.yaml` (pd2·ga4·ws4=32) — ★OOM 시 pd1·ga8·ws4 폴백, gpu_mem_util 조정.
   - `full_8b_h200.yaml` (pd2·ga4·ws4=32, 여유).
6. `sbatch/` (참고 템플릿: 기존 `analysis_qwen3_8b/sbatch/run_pass_rate_pilot_l40s.sh`, `4.6_Task2/.../run_*_l40s.sh`):
   - `accelerate launch --num_processes 4 ...` (ws=4). `--nodelist iREMB-C-07`(L40S) / `iREMB-C-03`(H200), `--gres=gpu:4`, no `--exclusive`.
   - opsd env: `source miniforge3/etc/profile.d/conda.sh && conda activate envs/opsd` 또는 `envs/opsd/bin/...`. PYTHONPATH에 opsd_src + curriculum 추가.
   - wandb 환경변수(project OPSD_curriculum 등), HF_HOME=cache/huggingface.
   - smoke 4 job: {diffonly, main} × {L40S, H200}. 8B feasibility: L40S 우선 + H200.
7. smoke 제출 → `stage_respected` rate=1.0 게이트 통과 확인 → reward_proxy rollout_acc 배선 → 8B feasibility(특히 L40S OOM ceiling·step time T_full) → full ②-A(diffonly)/③-A(main) → eval(AIME24/25, HMMT25, Avg@12, thinking ON).

## D. 경로/실행 메모
- opsd env python: `/scratch/lami2026/personal/jimin_2782/envs/opsd/bin/python` (설치 완료·동작 확인됨 — verify_schedule 실행 성공).
- schedule/phase0 산출물: `training/outputs/` (join_setA_meta.json, join_setA_rows.parquet, schedule_meta_*.json), stages: `training/stages/`.
- verify 재현: `cd training/curriculum && envs/opsd/bin/python verify_schedule.py` (기본 T=480 B_glob=32).
- T(max_steps) 선택은 num_stages로 나누어 떨어지게: main은 16의 배수, diffonly는 4의 배수 (budget=round(T/num_stages) 정합; train_opsd_curriculum의 len==T*B_glob assert 만족).

## E. 미완 체크리스트
- [x] GPU 구성·B_glob·reward_proxy·검증절차 확정 (A)
- [x] verify_schedule B_glob=32 기본값 + 재실행 PASS (B)
- [ ] opsd_src 통독 (opsd_train/opsd_trainer/data_collator — class/시그니처)
- [ ] curriculum_collator.py (stage_index[+Answer] passthrough)
- [ ] train_opsd_curriculum.py 재작성 (schedule→dataset, stage_index 컬럼, remove_unused_columns=False)
- [ ] curriculum_trainer.py 재작성 (training_step pop→super)
- [ ] curriculum_monitor.py (stage_respected/distinct-count, hard-fail)
- [ ] configs yaml ×4
- [ ] sbatch ×6 (smoke diffonly/main × L40S/H200, 8B feas L40S/H200)
- [ ] 1.7B smoke 제출·stage_respected 게이트
- [ ] reward_proxy rollout_acc 배선 (8B full 직전)
- [ ] 8B feasibility (★L40S 가능여부) → T_full → full ②-A/③-A
- [ ] eval

## F. 주의 (승계)
- upstream `opsd_src/` 직접수정 금지 → curriculum/ fork. collator/trainer/train placeholder는 **미검증, 재작성 대상**(신뢰 금지).
- C3{Algebra,Prealgebra} difficulty confound, D4|C3 over-recycle 인지. L8/D4 단독결론 금지.
- 공용계정 lami2026: 본인 JOBID/NAME만 식별·조작. 03/07 노드만. sudo/chmod777/공유dotfile·cache·시스템pip 금지.
