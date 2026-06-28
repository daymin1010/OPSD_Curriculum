# Hand-off — OPSD 커리큘럼 학습 env 구축 단계 (착수 직후)
작성: 2026-06-18
선행 정본(반드시 같이 읽기): `HANDOFF_CURRICULUM_LEARNING_2026-06-18.md`
  (분석 결론·클러스터 분할·stages_arm3_excludeOther.json·OPSD 코드 안내·미해결 항목의 정본)
이 파일은 그 후속으로 **env 구축 진행 상태**만 기록.

## 0. 한 줄 요약
- OPSD 커리큘럼 학습의 **환경 게이트 통과 + 전용 env 설치 시작**까지 완료.
- 다음: env 설치 완료 확인 → OPSD 코드 통독 → (사용자 디렉션) 코드 패치/스케줄러 → smoke → 8B feasibility → full.
- ⚠ 실험 디렉션(커리큘럼 주입 방식·arm·budget·eval)은 여전히 **사용자 스펙 대기**.

## 0-UPDATE (2026-06-18, 후속 세션) — 데이터 레이어 완료·검증
- **env 설치 완료 확인**: `envs/opsd/bin/python` 사용. trl.experimental.gold import OK.
- **OPSD 코드 통독 + 조인키 정합성 확정**: key=`sha1(problem)[:16] == labels.problem_id`,
  **match_rate=1.0000** (29,434 전건). opsd_index = train 순서 positional row index.
- ★**블로커 발견·해결**: 캐시된 `dataset_info.json`(및 Arrow schema metadata)이 신버전
  `datasets`의 `Feature 'List'` 사용 → opsd env의 `datasets==3.6.0`이 `load_dataset()`/
  `Dataset(table)` 양쪽에서 하드 실패. → **`curriculum/opsd_data.py` 신설**:
  Arrow shard 2개를 pyarrow로 직접 read → `table.replace_schema_metadata(None)`로
  깨진 메타만 제거(데이터 무손상, feature 재추론) → 진짜 `datasets.Dataset` 반환.
  shard 누락(23,719 단독) 학습 방지 가드 내장(29,434 / 2-shard 강제).
  - 치환 완료: `curriculum_schedule.load_opsd_problems()` + `train_opsd_curriculum.main()`
    둘 다 `load_dataset(OPSD_DATASET)` → `opsd_data.load_opsd_train()`로 교체.
- **phase0 실행 완료**: gate_pass=True, n_total=29,434, **n_setA=28,771**, n_other=663,
  min_cell=79 (D4|C3), empty=0. → `outputs/join_setA_meta.json`, `join_setA_rows.parquet`,
  `stages/stages_diffonly_setA.json`, `outputs/REPORT_join_setA.md` 생성.
- **`curriculum/verify_schedule.py` 신설+통과**: Phase-0 16-cell/4-diff 값을 frozen
  REFERENCE로 박아 데이터 드리프트 회귀 게이트 + 양 arm(main 16-stage / diffonly 4-stage)
  스케줄 드라이런(len==T*B_glob, B_glob 배수, empty pool 없음, budget=round(T/stages)).
  → **ALL CHECKS PASSED**. 재실행: `envs/opsd/bin/python verify_schedule.py --T 480 --B_glob 16`.
  ⚠ confound 플래그: D4|C3 pool=79 → T=480/B16에서 slots=480 → **cycles=7 과재활용**
    (선행 핸드오프 §G의 small-cell 경고와 일치; budget/T 확정 시 재검토).
- 신규 산출물(전부 `training/curriculum/`): `opsd_data.py`, `verify_schedule.py`.
- ⚠ 여전히 **사용자 디렉션 대기**(커리큘럼 주입 방식·arm·budget T·eval). 디렉션 수령 즉시
  1.7B pipeline smoke(T≈48, accelerate launch) → wandb 배관 → 8B feasibility 진행 가능.
- ⚠ `curriculum_schedule.py`/`curriculum_trainer.py`/`train_opsd_curriculum.py`는 데이터
  로딩 경로만 검증됨. trainer/loss/collator 경로는 GPU smoke 전까지 **미검증** — 신뢰 금물.


## A. 운영 환경 (엄수)
- ★ **`lami2026`은 공용(shared) user 계정**. `squeue`의 USER 컬럼도 `lami2026`으로 표시됨
  → 같은 노드/큐에 타인 잡이 섞여 보임. **반드시 본인 JOBID/NAME으로 식별, 본인 잡만 조작.**
  **개인 식별자/작업 네임스페이스 = `jimin_2782`** (작업 디렉토리·workspace 이름).
  → 모든 산출물은 `/scratch/lami2026/personal/jimin_2782/` 내부에서만 생성/수정.
  공유 ~/.bashrc, ~/.cache/huggingface, 타 personal 디렉토리는 읽기만.
- ★ 노드: **iREMB-C-03 / iREMB-C-07 만** 사용. sbatch만, --exclusive 금지, smoke→pilot→본런.
  점유 확인: `squeue -w iREMB-C-07` / `-w iREMB-C-03`.
- sudo / chmod777 / 공유 dotfile·cache 수정 / 시스템 pip 금지.

## B. env_check 실측 (job 86955, C-07/L40S)
- **C-07 (l40sq)**: L40S 46GB ×8, 점검 시 7장 free → 현재 다GPU 학습 현실적 노드.
- **C-03 (h200q)**: H200 141GB ×4, 단 점검 시 공용 계정 잡이 3장 점유(1장만 free). 메모리 여유 크나 가용 장수 변동.
- 파티션: h200q=C-01~06(gpu:4), l40sq=C-07~08(gpu:8). (우린 03/07만)
- verl_new(분석 env): torch2.10, transformers4.57.6, vllm0.19, flash-attn2.8.3 등 — **trl 없음**.
- 캐시 존재: Qwen3-8B(16G), OPSD 데이터셋 `siyanzhao___openthoughts_math_30k_opsd`.

## C. ★OPSD 학습 env (전용 env, 사용자 승인)
- verl_new 안 건드림. `opsd_src/environment.yml` 고정 스택으로 신규 생성:
  trl==0.26.0, torch==2.8.0, transformers==4.57.1, **vllm==0.11.0**, accelerate==1.11.0,
  deepspeed==0.18.2, peft==0.17.1, datasets==3.6.0, wandb, +flash-attn==2.8.3.
- **Miniforge personal prefix batch 설치**(~/.bashrc·시스템 conda 미수정): `miniforge3/` (설치됨).
  pkgs/pip 캐시 = `cache/conda_pkgs`, `cache/pip`.
- 설치 스크립트: `src/OPSD_Curriculum/training/setup_opsd_env.sh` (idempotent, 이어받기 가능).
- env 경로: **`envs/opsd`** → python `envs/opsd/bin/python`.
- 상태: 직전 세션 백그라운드 실행(PID 3145279), 로그 `runs/setup_opsd_env.log`. **완료 미확인**.

## D. ★다음 세션 첫 행동
1. `tail -n 50 runs/setup_opsd_env.log` → `==== DONE` + `[OK] trl.experimental.gold import OK` 확인.
   - 실패 시 의심: vllm0.11 ↔ torch2.8/flash-attn ABI, deepspeed 빌드. 재실행 = 같은 스크립트.
   - conda는 PATH에 없음 → `source miniforge3/etc/profile.d/conda.sh` 또는 `envs/opsd/bin/python` 직접.
2. OPSD 코드 통독(미완): `src/OPSD_Curriculum/training/opsd_src/`의
   `opsd_train.py`(데이터 로딩), `opsd_trainer.py`, `data_collator.py`, `scripts/run_opsd_1b.sh`, `eval/evaluate_math.py`.
3. 데이터 조인키 정합성(선행 핸드오프 §G.6): OPSD sample 식별자 ↔ `stages_arm3_excludeOther.json` / problem_id(sha1) 매핑.
4. **사용자 커리큘럼 디렉션 수령** 후 구현.

## E. 산출물 (training/ 신규)
- `setup_opsd_env.sh`, `sbatch/env_check.sh` (둘 다 재사용 가능).
- ⚠ `curriculum/curriculum_schedule.py`, `curriculum_trainer.py`, `train_opsd_curriculum.py`
  = 직전 세션 placeholder, **미검증/미완. 디렉션 받고 재작성. 신뢰 금물.**

## F. 미완 체크리스트
- [x] opsd env 설치 완료·검증 (D.1)
- [x] OPSD 코드 통독 + 데이터 조인키 (match_rate=1.0; opsd_data.py 블로커 해결)
- [x] verify_schedule.py + 실행 (Phase-0 회귀 게이트 + 양 arm 드라이런 PASS)
- [ ] collator passthrough 패치 (upstream 직접수정 금지 → fork) — 디렉션 후
- [ ] curriculum_monitor.py + reward_proxy.py — 디렉션 후
- [ ] configs yaml — 디렉션 후
- [ ] 1.7B smoke + wandb 배관 — 디렉션 후

- [ ] 8B short smoke → step time(T)·L40S OOM 확정 (8B+LoRA+vllm colocate 46GB 가능여부 미확정)
- [ ] full ②-A + ③-A
- [ ] eval (AIME24/25, HMMT25)

## G. 주의 (선행 핸드오프 승계)
- 커리큘럼 1차 입력 = `stages_arm3_excludeOther.json` (16-stage, C1~C4 × D1~D4).
- C3{Algebra,Prealgebra} difficulty confound 인지. L8 단독결론 금지. Prealgebra L5+ 없음.
- upstream `opsd_src/` 직접 수정 금지 → 새 디렉토리 fork.
