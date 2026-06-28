# OPSD 커리큘럼 러닝: Overfitting 발견 및 향후 계획

**작성일**: 2026-06-23
**관련 job**: 87867 (cond2_diff), 87868 (cond3_ours_C2) — H200 x2, 진행 중

---

## 1. 원래 질문: OPSD 에폭 = 샘플을 몇 번 보는가?

### OPSD 원본 논문/코드의 학습 설정
- **데이터셋**: OpenThoughts-Math-30K (29,434문제)
- **에폭**: 1 epoch (각 문제를 1번씩만 학습)
- **배치사이즈**: B_glob = 32 (global effective batch)
- **총 step 수**: ceil(29,434 / 32) ≈ **920 step**

### 핵심 발견: "에폭 1" ≠ "많은 학습"
직관적으로 "30K 데이터를 1 epoch"이라면 많은 학습이라고 생각할 수 있지만, **OPSD는 on-policy self-distillation**이므로:
- 매 step마다 새로운 trajectory를 생성 (vLLM rollout)
- 하지만 **teacher는 step 0에서 고정** (fixed teacher mode)
- loss는 teacher의 정답-조건부 분포에 student를 매칭 (JSD)

### OPSD 원본의 피크 step (README/논문 기준)
| 모델 | 피크 step | 피크 후 동향 |
|---|---|---|
| Qwen3-1.7B | ~100 step | 이후 성능 하락 |
| Qwen3-8B (non-think) | **~50 step** (49.7%) | step 100에서 38.3%로 하락 |

즉, OPSD는 **50-150 step 사이에서 피크**를 찍고, 그 이후에는 **teacher collapse / overfitting**으로 성능이 하락합니다. 30K를 1 epoch(920 step) 돌리는 게 아니라, **920 step 중 극히 일부(50-150 step)에서만 효과**가 있는 것입니다.

---

## 2. 현재 커리큘럼 설정과의 비교

### 현재 설정 (stages_cond3_ours_C2.json)
- **5 stages, 총 28,771 unique 문제** (전체 29,434 중 98%)
- `curriculum_passes=1` → 각 문제 1번씩
- `B_glob=32`, `max_steps=900` (T=900)
- stage별 step 분포:

| Stage | 문제 수 | step 범위 | 누적 step |
|---|---|---|---|
| 0 | 5,301 | 0→165 | 165 |
| 1 | 6,144 | 165→357 | 357 |
| 2 | 5,759 | 357→537 | 537 |
| 3 | 5,933 | 537→723 | 723 |
| 4 | 5,634 | 723→899 | 899 |

### 문제점: 900 step은 OPSD 피크의 6~18배
- OPSD 원본 피크: 50-150 step
- 현재 커리큘럼: 900 step
- stage 0 끝(165 step)에서 이미 원본 피크 영역 통과
- stage 1-4(358-899 step)는 원본에서 성능 하락하던 구간

### Loss 관찰 (cond3_ours, 실시간)
slurm 로그에서 추출한 loss 추이:

| 구간 (추정 step) | loss |
|---|---|
| 초기 (~2-10) | 0.0119, 0.0104, 0.0137 |
| 중반 (~26-50) | 0.0055, 0.0031, 0.0043 |
| 후반 (~52-64) | 0.0018, 0.0012, 0.0003 |
| 최근 (~66-74) | 0.0007, 0.0005, 0.0004 |

**loss가 0.01 → 0.0004로 25배 감소** = student가 teacher 분포에 거의 완벽히 매칭. 이는 **overfitting 신호**입니다.

---

## 3. 체크포인트 문제

### 현재 설정
- `save_total_limit: 3` → HuggingFace Trainer가 최신 3개 checkpoint만 유지
- `save_steps: 30`

### 결과: stage별 checkpoint 대부분 삭제됨
현재 남아있는 checkpoint:
- **cond3_ours**: checkpoint-750, 780, 810 (전부 stage 4)
- **cond2_diff**: checkpoint-780, 810, 840 (전부 stage 4)

stage 0-3 경계의 checkpoint(166, 358, 538, 723)는 **이미 삭제됨**.
OPSD 원본 피크 구간(50-150 step)의 checkpoint도 전부 삭제됨.

### 영향
- stage별 성능 변화를 추적할 수 없음
- overfitting 패턴을 checkpoint별 eval로 입증 불가
- 남은 late-stage checkpoint만으로는 커리큘럼 효과 분석 어려움

---

## 4. 현재 run의 의의

현재 진행 중인 900 step run은 **"1-pass 전체 학습 시 overfitting된다"는 증거**로 활용:
1. loss가 0.0004까지 단조 감소 → teacher 분포에 과적합
2. final adapter eval → 성능이 base 대비 향상되었는지 확인
3. 향후 step을 줄인 run과 비교 → 최적 step 수 탐색의 baseline

---

## 5. WandB 로그

### cond3_ours (job 87868)
- **Run name**: `full_cond3_ours_C2_manifest_once_h200_T900_bs32`
- **Run ID**: `14aqp4t6`
- **URL**: https://wandb.ai/acatwithasword_1010/OPSD_Curriculum/runs/14aqp4t6
- **로컬**: `wandb/wandb/run-20260623_001137-14aqp4t6`

### cond2_diff (job 87867)
- **Run name**: `full_cond2_diff_manifest_once_h200_T900_bs32`
- **Run ID**: `mqqb2jp6`
- **URL**: https://wandb.ai/acatwithasword_1010/OPSD_Curriculum/runs/mqqb2jp6
- **로컬**: `wandb/wandb/run-20260622_233048-mqqb2jp6`

- **Project**: https://wandb.ai/acatwithasword_1010/OPSD_Curriculum

---

## 6. 향후 계획

### 6.1 현재 run 완료 후 (즉시)
- [ ] job 87867, 87868 완료 대기 (건드리지 않음)
- [ ] 기존 eval 스크립트로 final adapter eval (math500, non-think, val_n=12)
  - `eval_cond3_ours_nonthink_math500_l40s.sh`
  - `eval_cond2_diff_nonthink_math500_l40s.sh`
- [ ] eval 결과를 base Qwen3-8B 성능과 비교

### 6.2 다음 run을 위한 코드 수정
- [ ] **`save_total_limit` 제거 또는 대폭 증가** (모든 checkpoint 보존)
- [ ] **`save_steps`를 stage 경계에 맞춤** 또는 더 빈번하게 (예: 10 step)
- [ ] **`max_steps` 축소**: 300-500 step (OPSD 피크 영역 커버)
- [ ] **stage별 샘플 축소**: 각 stage를 ~1,500-2,000문제로 서브샘플링
  - 총 ~7,500-10,000문제, step ≈ 250-310
  - 또는 각 stage에서 상위 N step만 학습하도록 max_steps 제한

### 6.3 Early stopping (선택적)
- [ ] loss threshold 기반 early stop callback 추가
  - 예: 최근 N step loss 평균이 0.001 이하면 중단
- [ ] 또는 stage별 early stop: 각 stage 끝에서 loss가 임계값 이하면 다음 stage로

### 6.4 Stage별 eval
- [ ] 각 checkpoint를 개별 eval할 수 있도록 eval 스크립트 수정
  - `--checkpoint_dir`을 특정 checkpoint 경로로 지정 가능하도록
- [ ] stage 경계 checkpoint(166, 358, 538, 723, 899) 우선 eval
- [ ] checkpoint별 성능 곡선으로 피크 step 탐지

### 6.5 권장 step 수 (가설)
OPSD 원본 피크(50-150 step)를 참고:
- **stage 0**: 50-100 step (원본 피크 영역)
- **stage 1-4**: 각 30-50 step씩 (커리큘럼 효과 탐색)
- **총**: 200-350 step

이렇게 하면:
1. stage 0에서 기본 distillation 효과 확보
2. 이후 stage에서 커리큘럼(난이도/subject 변화)이 추가 gain을 주는지 확인
3. overfitting 방지

---

## 7. 파일 위치 참고

### 코드
- 학습 엔트리: `src/OPSD_Curriculum/training/curriculum/train_opsd_curriculum_manifest_once.py`
- 스케줄 빌더: `src/OPSD_Curriculum/training/curriculum/curriculum_schedule_manifest_once.py`
- 모니터: `src/OPSD_Curriculum/training/curriculum/curriculum_monitor_manifest_once.py`
- Trainer: `src/OPSD_Curriculum/training/curriculum/curriculum_trainer.py`
- Config: `src/OPSD_Curriculum/training/curriculum/configs/full_8b_h200.yaml`

### 데이터
- Stage manifest: `src/OPSD_Curriculum/training/stages_tiered_20260622/stages_cond3_ours_C2.json`
- Row table: `src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet`

### Checkpoint
- `checkpoints/opsd_curriculum/full_8b/full_cond3_ours_C2_manifest_once_h200/`
- `checkpoints/opsd_curriculum/full_8b/full_cond2_diff_manifest_once_h200/`

### Eval 스크립트
- `src/OPSD_Curriculum/training/curriculum/sbatch/eval_cond3_ours_nonthink_math500_l40s.sh`
- `src/OPSD_Curriculum/training/curriculum/sbatch/eval_cond2_diff_nonthink_math500_l40s.sh`

### Slurm 로그
- `runs/slurm-opsd_cur_cond3_ours_C2_manifest_once_h200.87868.iREMB-C-03.{out,err}`
- `runs/slurm-opsd_cur_cond2_diff_manifest_once_h200.87867.iREMB-C-03.{out,err}`

---

## 요약

OPSD는 "30K 1 epoch"이라지만 실제 효과는 **50-150 step**에 집중. 현재 커리큘럼은 900 step으로 overfitting 위험이 크며, loss 로그도 이를 시사함. 향후 run에서는:
1. **step 수를 200-350으로 축소**
2. **checkpoint를 빈번히 저장** (save_total_limit 제거)
3. **각 checkpoint를 개별 eval**하여 피크 step 탐지
4. (선택) **loss 기반 early stopping**