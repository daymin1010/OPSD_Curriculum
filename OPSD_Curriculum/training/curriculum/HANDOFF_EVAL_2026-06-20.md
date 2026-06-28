# OPSD 커리큘럼 — eval 단계 인계 프롬프트 (2026-06-20)

작업 디렉토리: `/scratch/lami2026/personal/jimin_2782`. opsd env: `envs/opsd/bin/python`. 본인 JOBID/NAME(87xxx)만 조작.

## 정본 문서 (순서)
1. `src/OPSD_Curriculum/reasoning_pivot/activation/analysis/HANDOFF_CURRICULUM_LEARNING_2026-06-18.md` (클러스터·stages)
2. `src/OPSD_Curriculum/training/HANDOFF_OPSD_ENV_SETUP_2026-06-18.md` / `…b.md` (env·GPU·B_glob)
3. **이 프롬프트 = 최신 정본(학습 완료→eval 단계).**

## 운영/안전 (엄수)
- `lami2026` 공용계정. **sbatch만**, `--exclusive` 금지, GPU≤4, **L40S 우선**. 노드: iREMB-C-07(L40S, partition `l40sq`) / iREMB-C-03(H200, `h200q`).
- sudo·chmod777·시스템 pip·공유 dotfile 금지. upstream `opsd_src/` 직접 수정 금지(이미 fork됨).
- **추측 코드 금지**, 큰 로그 전체 출력 금지(grep/tail). `list_files` 재귀 남용 금지.

## 현재 상태 (2026-06-20)
- **학습 2런 모두 정상.**
  - ②-A **diff-only (job 87068) = COMPLETED**. 어댑터: `checkpoints/opsd_curriculum/full_8b/full_diffonly_h200/` (최종 step480이 top-level, +checkpoint-420/450/480). LoRA r=64, alpha=128.
  - ③-A **main (job 87069) = RUNNING**, 마지막 확인 step 176/480, ETA ~7h(완료 예상 06-20 밤). loss 정상(-0.011), monitor gate(stage_respected) ABORT 없음. 산출: `checkpoints/opsd_curriculum/full_8b/full_main_h200/`.
- **eval diff-only (job 87400) = 큐 적재(PD, l40sq)**. 스크립트: `src/OPSD_Curriculum/training/curriculum/sbatch/eval_diffonly_l40s.sh`.
  - OPSD 정본 `opsd_src/eval/evaluate_math.py` 호출. base=`Qwen/Qwen3-8B`, `--checkpoint_dir`=diffonly 어댑터, **thinking ON, val_n=12(Avg@12), temp=1.0**, TP=2(L40S×2), gpu_mem_util=0.9, datasets=aime24·aime25·hmmt25.
  - 결과 JSON: `outputs/eval_opsd_curriculum/diffonly/{ds}_diffonly_thinking_valn12.json` (wandb 아님, JSON 저장).
  - feas 잔재(checkpoint-8, feas_8b_l40s)는 정리 완료.

## eval 조건이 OPSD 원본과 동일함 (대조 완료)
`opsd_src/eval/run_eval.sh` + `evaluate_math.py` 디폴트 대비: val_n=12 / temperature=1.0 / enable_thinking=True / top_p=auto0.95 / top_k=-1 / min_p=0.0 / max_new_tokens=38912 / max_model_len=auto40960 / gpu_mem_util=0.9 / LoRA `--checkpoint_dir`(LoRARequest) — **전부 동일**.
의도된 결과-중립 차이 2개: ① `tensor_parallel_size` 4→2 (L40S 2장, 점수 무관), ② dataset을 aime24 단일→**aime24·aime25·hmmt25 3종**(과제 스펙). 원본 run_eval.sh의 base baseline은 미포함(원하면 별도 잡).

## 가장 먼저 — 87400 결과 게이트 체크 (1줄)
```
cd /scratch/lami2026/personal/jimin_2782
E=$(ls -t runs/slurm-opsd_cur_eval_diffonly_l40s.87400.*.err 2>/dev/null|head -1); O=${E%.err}.out
squeue -j 87400 -o "%.8i %.30j %.2t %R" || echo "(끝남)"
echo FAIL:; grep -ciE "Error|Traceback|No space|out of memory|OSError|ConnectionError" "$E"
echo RESULT:; grep -hE "Average@|Pass@|Majority Vote@|FINAL RESULTS|ALL EVAL DONE" "$O" | tail -20
```
- green 판정: FAIL=0 + 3개 데이터셋 모두 `Average@12`/`Pass@12` 출력 + `ALL EVAL DONE`.

## 87400 실패 시 분기
- **HF 다운로드 에러**(aime25=`yentinglin/aime_2025`, hmmt25=`MathArena/hmmt_feb_2025`, trust_remote_code 필요): 로그인 노드에서
  `HF_HOME=$PWD/cache/huggingface envs/opsd/bin/python -c "from datasets import load_dataset; load_dataset('yentinglin/aime_2025',split='train',trust_remote_code=True); load_dataset('MathArena/hmmt_feb_2025',split='train',trust_remote_code=True)"`
  로 사전 캐시 후 재제출. (aime24=`HuggingFaceH4/aime_2024`)
- **L40S OOM(40960 ctx, TP=2)**: `--gpu_memory_utilization 0.85`로 낮추거나 데이터셋별 잡 분리(점수 무관).
- **LoRA**: r=64 == 스크립트 `max_lora_rank=64`로 OK. 로그에서 `Successfully created LoRA request` 확인(없으면 base-only로 빠진 것).

## green이면 — main(③-A) eval
1. 87069 완주 확인(`squeue`, `full_main_h200/`에 `adapter_model.safetensors` 존재).
2. `eval_diffonly_l40s.sh` 복사→`eval_main_l40s.sh`: `CKPT`를 `…/full_main_h200`로, job-name/OUTDIR를 main으로 변경(나머지 동일). `bash -n` 후 `sbatch`.
3. (선택) base baseline: `--checkpoint_dir` 생략 동일 조건 잡으로 Qwen3-8B 원본 점수 — 3-way 비교용.

## 결과 집계
- 3 데이터셋 × {diffonly, main(, base)}의 Avg@12·Pass@12 표. JSON 키: `average_at_n_pct`, `pass_at_n_pct`, `majority_vote_at_n_pct`, `format_rate`.
- 핵심 비교: **diff-only vs main**(커리큘럼 arm) on AIME24/AIME25/HMMT25, thinking ON.

## wandb
- 학습 잡만 `WANDB_PROJECT=OPSD_Curriculum`로 게이트/loss 로깅. **eval은 wandb 미사용(JSON 저장)** → 키 불필요. 학습 추가/재개로 wandb UI 필요 시 그때 키/엔티티/`WANDB_MODE=offline` 지시.

## 참고 경로
- 학습 sbatch: `…/curriculum/sbatch/full_{diffonly,main}_h200.sh`, config: `…/configs/full_8b_h200.yaml`
- eval sbatch: `…/curriculum/sbatch/eval_diffonly_l40s.sh`
- 정본 eval 코드(수정금지): `…/opsd_src/eval/evaluate_math.py`
