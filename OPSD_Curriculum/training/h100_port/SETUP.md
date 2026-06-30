# H100 서버 세팅 & 실행 runbook (4B 커리큘럼 실험)

맨바닥 H100(4장) 서버에서 4B 커리큘럼 실험을 돌리기 위한 포팅 가이드. **직접 실행(SLURM 아님).**

---

## 0. 필요한 것
- H100 ×4, conda/miniforge, 인터넷(HF 다운로드·git).
- 이 repo(`OPSD_Curriculum`) + 아래 **별도 전송 파일 1개**.

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
git pull로 들어오는 것: `opsd_src/`, `curriculum/`(configs·trainer), `stages_cliff4b_20260630/`(manifest 5종, .json 추적됨).

## 3. 별도 전송 파일 ⚠️ (gitignore라 git에 없음)
- **`OPSD_Curriculum/training/outputs/join_setA_rows.parquet`** (~1MB)
  - problem_id → opsd_index 매핑. 스케줄 빌드에 필수.
  - scp로 옮기거나, repo의 `.gitignore`에 `!OPSD_Curriculum/training/outputs/join_setA_rows.parquet` 한 줄 추가해 git으로 관리 권장(작음).
```bash
scp <thisserver>:.../training/outputs/join_setA_rows.parquet \
    $REPO/OPSD_Curriculum/training/outputs/
```

## 4. 데이터·모델 (인터넷 자동 다운로드)
- 학습 데이터: `siyanzhao/Openthoughts_math_30k_opsd` (HF, opsd_src가 `load_dataset`으로 자동).
- 모델: `Qwen/Qwen3-4B` (실험), `Qwen/Qwen3-8B`(필요시). 첫 실행 시 HF가 받음. `export HF_HOME=$WORK/hf`로 캐시 위치 지정.

## 5. accelerate / GPU
- 스크립트는 `--num_processes 4`(H100 4장)로 호출. `opsd_src/accelerate.yaml`의 deepspeed/그래디언트 설정 확인(필요시 4장에 맞게).
- config `configs/full_4b_cliff.yaml`: Qwen3-4B, max_completion 4096(전 arm 공통=공정), B_glob 32. (stage별 context ramp는 v2 — manifest의 `context_per_stage` 필드 사용 시 trainer wrapper 패치 필요.)

## 6. 실행 (학습)
```bash
export REPO=...; export ENV_PY=$(which python); export WORK=$REPO/_run
cd $REPO/OPSD_Curriculum/training/h100_port
for ARM in shuffle diff cliff_P subj_V1 subj_shuf; do
  ./train_cliff4b.sh $ARM            # 4장 직접 실행, 순차
done
```
체크포인트: `$WORK/checkpoints/full_4b_cliff/cliff4b_<ARM>/checkpoint-*` (config output_dir를 $WORK 기준으로 바꾸거나 그대로 두고 경로 확인).

## 7. 실행 (eval)
```bash
for ARM in shuffle diff cliff_P subj_V1 subj_shuf; do
  ./eval_cliff4b.sh $ARM "100 400 650 900"
done
```
결과 json: `$WORK/outputs/eval/cliff4b_<ARM>_nonthink/`. 이 json들을 git push(또는 scp)로 가져오면 분석 가능.

## 8. 결과 회수 → 분석
- eval json(작음)만 push/scp. 체크포인트(큼)는 두고 옴.
- 분석은 메인 서버에서: ours/diff/cliff/subject 비교표.

---
## arm 요약
| arm | 구성 | 비교 |
|---|---|---|
| shuffle | 무커리큘럼(랜덤) | 바닥 |
| diff | tight level 밴드 | 표준 |
| cliff_P | smooth 절벽 밴드(partition) | 메인 |
| subj_V1 | 과목 시퀀싱(톱니) | subject-primary |
| subj_shuf | 과목 블록 랜덤순서 | subj 대조 |

전부 4B · clean 28,743 · partition(각 문제 1번) · 동일 compute.
