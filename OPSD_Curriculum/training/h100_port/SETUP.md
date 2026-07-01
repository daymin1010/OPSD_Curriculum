# H100 서버 세팅 & 실행 runbook (4B 커리큘럼 실험)

맨바닥 H100(4장) 서버에서 4B 커리큘럼 실험을 돌리기 위한 포팅 가이드. **직접 실행(SLURM 아님).**

---

## 0. 필요한 것
- H100 ×4(GPU 0-3만 사용), conda/miniforge, 인터넷(HF 다운로드·git).
- 이 repo(`OPSD_Curriculum`)만 있으면 됨 — 필요한 데이터파일(row table)도 이제 git에 포함.

## 1. 환경 (conda)
```bash
# miniforge 설치 후
conda create -p ./envs/opsd python=3.10 -y
conda activate ./envs/opsd
pip install -r OPSD_Curriculum/training/h100_port/requirements_pip.txt
```
핵심 버전(검증됨): torch 2.8.0 · vllm 0.11.0 · transformers 4.57.1 · trl 0.26.0 · accelerate 1.11.0 · peft 0.17.1 · flash-attn · deepspeed 0.18.2.

## 2. repo + 코드
```bash
git clone <repo_url> $REPO_PARENT     # 또는 git pull
# REPO = OPSD_Curriculum 의 상위 디렉토리
export REPO=$REPO_PARENT
```
git pull로 들어오는 것: `opsd_src/`, `curriculum/`(configs·trainer, **context_scaling 패치 포함**), `stages_cliff4b_20260630/`(manifest + build 스크립트).

## 3. 별도 전송 파일 — 이제 불필요 ✅
- **`OPSD_Curriculum/training/outputs/join_setA_rows.parquet`** (~1MB, problem_id → opsd_index 매핑, 스케줄 빌드 필수)은 **git에 추적됨** → `git pull`로 자동 반영. 별도 scp 불필요.

## 4. 데이터·모델 (인터넷 자동 다운로드)
- 학습 데이터: `siyanzhao/Openthoughts_math_30k_opsd` (HF, opsd_src가 `load_dataset`으로 자동).
- 모델: `Qwen/Qwen3-4B` (실험), `Qwen/Qwen3-8B`(필요시). 첫 실행 시 HF가 받음. `export HF_HOME=$WORK/hf`로 캐시 위치 지정.

## 5. accelerate / GPU
- 스크립트는 `--num_processes 4`(H100 4장)로 호출. `opsd_src/accelerate.yaml`의 deepspeed/그래디언트 설정 확인(필요시 4장에 맞게).
- config `configs/full_4b_cliff.yaml`: Qwen3-4B, B_glob 32, **`context_scaling: true`** → stage별 생성 max_new_tokens 1024→4096 램프(패치 적용됨, teacher/loss 무관). max_completion 4096=상한.

## 6. 실행 (학습)
```bash
export REPO=...; export ENV_PY=$(which python); export WORK=$REPO/_run
export CUDA_VISIBLE_DEVICES=0,1,2,3      # H100 0-3번만
cd $REPO/OPSD_Curriculum/training/h100_port
# 지금 우선순위: shuffle, cliff_P (나머지 arm은 필요 시)
for ARM in shuffle cliff_P; do
  ./train_cliff4b.sh $ARM              # 4장 직접 실행, 순차
done
# subject 검정 arm (2-seed): SEED 인자 지정
#   ./train_cliff4b.sh cliff_subjgeo 42
#   ./train_cliff4b.sh cliff_subjgeo 43
#   ./train_cliff4b.sh cliff_subjrand_s0 42
#   ./train_cliff4b.sh cliff_subjrand_s1 43
```
체크포인트: `$WORK/checkpoints/full_4b_cliff/cliff4b_<ARM>/checkpoint-*` (config output_dir를 $WORK 기준으로 바꾸거나 그대로 두고 경로 확인).

## 7. 실행 (eval)
```bash
for ARM in shuffle cliff_P; do
  ./eval_cliff4b.sh $ARM "100 400 650 900"
done
# subject arm은 run_config에 seed 접미사가 붙으므로 CKPT_BASE 지정:
#   CKPT_BASE=$WORK/checkpoints/full_4b_cliff/cliff4b_cliff_subjgeo_s42 \
#   OUTDIR=$WORK/outputs/eval/cliff4b_cliff_subjgeo_s42_nonthink \
#   ./eval_cliff4b.sh cliff_subjgeo "100 400 650 900"
```
결과 json: `$WORK/outputs/eval/cliff4b_<ARM>_nonthink/`. 이 json들을 git push(또는 scp)로 가져오면 분석 가능.

## 8. 결과 회수 → 분석
- eval json(작음)만 push/scp. 체크포인트(큼)는 두고 옴.
- 분석은 메인 서버에서: pooled AIME24+25+HMMT25 avg@12로 arm 비교.

---
## arm 요약
| arm | 구성 | 역할 |
|---|---|---|
| shuffle | 무커리큘럼(랜덤) | 바닥 baseline |
| diff | tight level 밴드(겹침×) | 난이도-only |
| cliff_P | 절벽 슬라이딩창(겹침), 과목 무작위 | 난이도 메인 / 과목 대조 |
| cliff_subjgeo | cliff + 과목 기하 co-move | **풀 메인 / treatment** |
| cliff_subjrand_s0/s1 | cliff + 과목 무작위(재추첨) | 과목 null 대조 |

전부 4B · clean 28,743 · partition(각 문제 1번) · context_scaling ON.
(구 subj_V1/subj_shuf는 폐기 — 중간선 cliff_subjgeo로 대체.)
