#!/bin/bash
# 미뤄진 8B cliff 재평가 4종 (2026-07-13 사용자 결정으로 보류) — C-03 여유 있을 때 실행.
# 옛 cond2_diff ckpt들(현행 cliff와 구조 동일 검증됨)을 val_n=12로 재채점 → 8B 데이터-분량 곡선의 cliff 참조점.
# 사용: bash postponed_cliff_revals.sh   (직렬 체인으로 제출, 동시 1 job=2 GPU)
set -euo pipefail
cd "$(dirname "$0")/../mainphase_20260709"
CK=/scratch/lami2026/personal/jimin_2782/checkpoints/opsd_curriculum
EVD=/scratch/lami2026/personal/jimin_2782/outputs/eval_opsd_curriculum
PREV=""
submit(){ local name=$1 ckpt=$2
  local dep=""; [ -n "$PREV" ] && dep="--dependency=afterany:$PREV"
  PREV=$(RUN_PREFIX=reval8b CKPT_BASE=$ckpt OUTDIR=$EVD/reval8b_${name}_nonthink \
    sbatch --parsable --export=ALL $dep --job-name reval8b_${name} eval_cliff8b_h200.sh ${name})
  echo "reval8b_${name}: $PREV"
}
submit cliff_full $CK/full_8b/full_cond2_diff_manifest_once_h200
submit cliff_d25  $CK/quarter_8b/quarter_cond2_diff_q4_h200
submit cliff_d11  $CK/mini100_8b/mini100_cond2_diff_h200
submit cliff_d06  $CK/mini50_8b/mini50_cond2_diff_h200
