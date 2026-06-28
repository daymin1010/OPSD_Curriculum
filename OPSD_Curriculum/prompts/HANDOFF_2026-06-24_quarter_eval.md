# OPSD 커리큘럼 실험 — 다음 세션 핸드오프 (2026-06-24)

## 현재 상황 (2026-06-24 00:42 기준)

### 진행 중인 Slurm job
- **87995**: `quarter_cond3_ours_q4_h200` (1/4 스케일 ours arm, N=7193, T=225) — 제출됨, 실행 대기/진행 중
- **87996**: `quarter_cond2_diff_q4_h200` (1/4 스케일 diff arm, N=7193, T=225) — 제출됨, 실행 대기/진행 중
- **87867/87868**: 원본 900-step full run (cond2_diff/cond3_ours) — 완료 또는 완료 임박
- **87979/87994**: step900 eval (cond2_diff/cond3_ours) — pending (AssocGrpCpuLimit)
- **87966/87969**: step810 eval (cond2_diff/cond3_ours) — C-07에서 running

### 1/4 스케일 실험 개요
원본 28,771문제(OpenThoughts-Math-30K Set-A)에서 공통으로 7,193개(25%, seed=42)를 추출.
두 arm(cond2_diff vs cond3_ours)이 **정확히 동일한 7,193문제**를 학습하되, stage 배치만 다름.
→ 공정한 A/B 비교. stage 순서 효과만 분리하여 측정.

OPSD 원본 피크(50-150 step)가 1/4 스케일의 stage 1-2에 위치 → 피크 구간 정밀 커버.

## 생성된 파일 목록
1. `src/OPSD_Curriculum/training/curriculum/make_quarter_manifests.py` — 공통 universe 1/4 샘플링 스크립트
2. `src/OPSD_Curriculum/training/stages_tiered_20260622/stages_cond2_diff_q4.json` — 1/4 diff manifest (N=7193)
3. `src/OPSD_Curriculum/training/stages_tiered_20260622/stages_cond3_ours_C2_q4.json` — 1/4 ours manifest (N=7193)
4. `src/OPSD_Curriculum/training/curriculum/configs/quarter_8b_h200.yaml` — 1/4 config (save_steps=10, save_total_limit=30)
5. `src/OPSD_Curriculum/training/curriculum/sbatch/quarter_cond3_ours_q4_h200.sh` — ours arm sbatch
6. `src/OPSD_Curriculum/training/curriculum/sbatch/quarter_cond2_diff_q4_h200.sh` — diff arm sbatch
7. `src/OPSD_Curriculum/training/curriculum/verify_schedule_manifest_once.py` — `--expect_universe none` 옵션 추가 (수정)

## Stage 경계 (1/4 스케일, B_glob=32)

| stage | diff step 범위 | ours step 범위 | diff n | ours n |
|---|---|---|---|---|
| 0 | 0→42 | 0→41 | 1370 | 1326 |
| 1 | 42→86 | 41→88 | 1412 | 1520 |
| 2 | 86→134 | 88→132 | 1522 | 1410 |
| 3 | 134→180 | 133→179 | 1485 | 1479 |
| 4 | 180→224 | 179→224 | 1404 | 1458 |

## 학습 설정
- Qwen3-8B, H200 ×2, B_glob=32 (pd2×ga8×ws2)
- LoRA r=64, alpha=128, LR=5e-6
- save_steps=10, save_total_limit=30 (대부분 checkpoint 보존)
- curriculum_passes=1, within_stage_order=shuffle, tail_policy=partial

## 검증 상태
- `verify_schedule_manifest_once.py` dry-run: **ALL CHECKS PASSED ✓**
- manifest 공정성: 두 arm universe MD5 hash 동일 (`1df57719219c`), N=7193, 중복 0
- 원본 universe(28,771)의 정확히 25% 부분집합

## 다음 세션에서 할 작업

### 1. job 상태 확인
```bash
squeue -u lami2026 --format="%.10i %.30j %.8T %.10M %R"
```

### 2. 1/4 학습 완료 후 checkpoint별 eval 스크립트 작성
- checkpoint는 `checkpoints/opsd_curriculum/quarter_8b/quarter_cond{2,3}_*_q4_h200/` 아래에 step 10,20,...,220 저장됨
- 기존 eval 스크립트(`eval_cond3_ours_nonthink_math500_l40s.sh` 등)를 템플릿으로 사용
- 각 checkpoint에 대해 math500 non-think eval 실행
- 핵심 checkpoint: step 10, 20, 30, 40, 50, 60, 80, 100, 130, 160, 190, 220
- 결과로 step별 성능 곡선 작성 → 피크 step 탐지

### 3. 원본 900-step run eval 결과 수집
- job 87979/87994(step900 eval) 완료 대기
- job 87966/87969(step810 eval) 결과 확인
- base Qwen3-8B 성능과 비교

### 4. 디스크 정리 (나중에)
- checkpoints/opsd_curriculum/full_8b/ 아래 checkpoint 6개 (step 750-900) 중 불필요한 것 정리
- eval_ckpts/ 아래 중복 어댑터 정리

## 참고 문서
- `src/OPSD_Curriculum/training/curriculum/STATUS_OVERFITTING_AND_PLAN_2026-06-23.md` — overfitting 분석 + 향후 계획
- `src/OPSD_Curriculum/training/stages_tiered_20260622/REPORT_stagebuild_2026-06-22.md` — 스테이지 설계 문서
- `paper/25580_Neuron_Aware_Data_Select.pdf` — 논문 PDF

## 환경
- conda env: `/scratch/lami2026/personal/jimin_2782/envs/opsd`
- 활성화: `source /scratch/lami2026/personal/jimin_2782/miniforge3/etc/profile.d/conda.sh && conda activate /scratch/lami2026/personal/jimin_2782/envs/opsd`
- PYTHONPATH: `src/OPSD_Curriculum/training/opsd_src:src/OPSD_Curriculum/training/curriculum`