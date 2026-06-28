# 🧭 Session Hand-off — Activation Extraction & Label-Driven Analysis

작성일: 2026-05-30
작성자: Jimin (jimin_2782)
상태: **labeling phase 종료, activation extraction phase 시작 직전**

---

## 0. 한 줄 요약 & 다음 세션 Mission

- **방금 마친 일**: OpenThoughts-30K 의 29,434 문제 전체에 `gpt-4.1-mini` 로 `(subject, level)` 라벨 부여. 100% 성공, $9.
- **다음 세션 mission**: 사용자가 줄 *추출 방향(direction)* 에 따라 Qwen3-8B 로 **activation 을 새로 추출**하고, 이번에 만든 라벨/pilot universe 로 분석.
- **이 세션 AI 의 책임**: 추출 스펙 (어떤 layer, 어떤 토큰 위치, thinking 여부, 어떤 단위) 을 **사용자가 정해서 줄 때까지 임의로 결정하지 말 것**. 미리 정해두면 폐기 비용 큼.

---

## 1. ⚠️ 운영 환경 — **반드시 먼저 읽기**

### 1.1 공동 계정 & 작업 디렉토리
- `lami2026` 은 **여러 사람이 공유하는 user 계정**입니다.
- `/home/lami2026/`, `/scratch/lami2026/` 의 대부분은 다른 사용자와 공유 자원.
- **모든 작업물은 `/scratch/lami2026/personal/jimin_2782/` 내부에서만** 생성/수정.
- `~/.bashrc`, 공유 `~/.cache/huggingface/`, 다른 personal 디렉토리 (`/scratch/lami2026/personal/<other>`) 절대 수정 금지 (읽기만 OK).

### 1.2 GPU 노드 정책
| 용도 | 노드 | 자원 | sbatch flag |
|---|---|---|---|
| **추론 (이번 단계: activation 추출, pass-rate)** | `iREMB-C-07` | L40s 48GB × 4 | `--nodelist=iREMB-C-07 --gres=gpu:4` |
| **학습 (이후 단계: NAIT-curriculum SFT/RL)** | `iREMB-C-03` | H200 80GB × 2 | `--nodelist=iREMB-C-03 --gres=gpu:2 --partition=h200q` |

- **다른 노드 (`iREMB-C-02/04/05/06` 등) 절대 사용 금지.**
- `--exclusive` 금지. 필요한 GPU 수만 정확히 요청.
- TIME LIMIT 짧게: smoke ETA × 2 를 상한선으로. 무한 점유 절대 금지.

### 1.3 GPU 호출 방식
- **VS Code 터미널에서 직접 `python ... --device cuda` 실행 금지** (=login/compute 외부 노드).
- 모든 GPU 작업은 **`sbatch <script.sh>` 만**.
- sbatch 스크립트 헤더에 항상 `--nodelist`, `--gres`, `--time`, `--output`, `--error` 명시.
- MPS daemon 필요 시 sbatch 안에서만:
  ```bash
  unset ROCR_VISIBLE_DEVICES
  nvidia-cuda-mps-control -d || true
  trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT
  ```

### 1.4 다른 사용자에 대한 배려 (safety checklist)
**매 GPU job 제출 전 5개 확인:**
1. `[ ]` 출력 디렉토리가 새 경로인가 (기존 산출물 덮어쓰지 않는가)
2. `[ ]` sbatch 헤더에 `--nodelist`, `--gres`, `--time` 모두 있는가
3. `[ ]` `squeue -w iREMB-C-07` 로 다른 user job 점유율 확인했는가
   - 다른 user 가 GPU 4개 모두 점유 중이면 대기, 강제 인터럽트 금지
   - L40s 4 GPU 중 2개만 비어 있으면 `--gres=gpu:2` 로 축소해서 양보
4. `[ ]` smoke run (≤30 sample, ≤15분) 먼저 통과했는가
5. `[ ]` 새 dataset/모델 다운로드 위치가 `/scratch/lami2026/personal/jimin_2782/cache/` 인가 (공유 `~/.cache/huggingface` 오염 금지)

### 1.5 절대 하지 말 것 (one-strike list)
- `sudo`, `chmod 777`, `~/.bashrc` 또는 공유 `~/.cache` 수정
- `srun --pty` 로 interactive GPU 잡 30분 이상 점유
- VS Code 터미널에서 GPU 직접 호출
- 다른 personal user 디렉토리 read 외 접근
- `iREMB-C-02/04/05/06` 사용
- `pip install` 시스템 python 에 (반드시 `envs/verl_new/bin/pip` 안에)
- `wandb` 가 공유 `~/.config/wandb` 에 쓰는 것 (envvar `WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb` 명시)

---

## 2. 이번 세션 산출물 (요약)

### 2.1 데이터
| 파일 | 설명 |
|---|---|
| `src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet` | **최종 정규화 본** 29,434 × 22 cols. `problem_id` (sha1 16hex) unique. subject 8-canonical. |
| `src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet` | **pilot 3,000** stratified by (subject, level), seed=42. activation 추출 입력 후보. |
| `src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels.csv` | raw API output (보존) |
| `src/OPSD_Curriculum/labeling/outputs/smoke200_labels.csv` | smoke run 보존본 |

### 2.2 검증 (모두 PASS)
- rows: **29,434 / 29,434**, errors=0, truncation=0, retries=0, JSON valid=100%
- ρ(level, r1_cot_token_count) = **0.61** (Spearman) — 강한 난이도 신호
- problem_id 중복 0
- 비용 실측 **$9** (no-cache 추정 $20.24, prompt caching ~55% 절감)
- OOV subject 66건 (0.22%) → `Other` 로 매핑, 원본은 `subject_raw` 컬럼

### 2.3 코드 / 리포트
- `src/OPSD_Curriculum/labeling/label_openthoughts_30k.py` — async API 라벨러
- `src/OPSD_Curriculum/labeling/analyze_full.py` — sanity 12-point
- `src/OPSD_Curriculum/labeling/postprocess.py` — 정규화/id/pilot 추출
- `src/OPSD_Curriculum/labeling/outputs/REPORT_full.md` — 분포 & 신호
- `src/OPSD_Curriculum/labeling/outputs/REPORT_pilot.md` — pilot stratification 진단
- `src/OPSD_Curriculum/labeling/HANDOFF_LABELING_DONE.md` — labeling sub-task 보고

---

## 3. 다음 세션 첫 행동 (순서)

1. **이 hand-off** 통독 (1–2분).
2. (선택) 이전 컨텍스트:
   - `src/OPSD_Curriculum/labeling/HANDOFF_LABELING_DONE.md`
   - `src/OPSD_Curriculum/analysis_qwen3_8b/HANDOFF_THINKING_MODE.md` (이전 thinking-mode 미완 task)
   - `src/OPSD_Curriculum/analysis_qwen3_8b/nait/MASTER_SUMMARY_qwen3_8b.md`
3. pilot universe 확인:
   ```bash
   PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
   $PY -c "
   import pandas as pd
   p = pd.read_parquet('src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet')
   print(p.shape, p.columns.tolist())
   print(p[['problem_id','subject','level']].head())
   print('subj x level:'); print(pd.crosstab(p['subject'],p['level']))
   "
   ```
4. **사용자에게 추출 방향(direction)을 받기**. 받기 전엔 절대 추출 스크립트 작성/실행하지 말 것.
   - 예시 질문 항목: 어떤 layer 집합? 어떤 token 위치 (prompt 마지막? generation 첫 토큰? thinking block 안? `</think>` 직후?)? thinking-mode on/off? batch=1 / batched?
   - pilot 도 그대로 3,000 쓸지, 다른 universe 쓸지 확인.
5. 받은 spec 으로 **smoke (10–30 sample) → pilot (3,000) → 분석** 단계 강제 (스킵 금지).

---

## 4. 권장 디렉토리 구조 (다음 세션에서 만들 것)

```
src/OPSD_Curriculum/
├── HANDOFF_2026-05-30_activation_extraction.md   ← 이 파일
├── labeling/                                      ← 이번 세션 (보존)
├── analysis_qwen3_8b/                             ← 이전 세션 (보존)
└── activation_v2/                                 ← ★ 다음 세션에서 새로 만듦
    ├── extract_<spec_name>.py                     ← 사용자 dir 반영
    ├── sbatch/
    │   ├── run_smoke.sh
    │   └── run_pilot_chunk{0..N}.sh
    ├── outputs/
    │   ├── shifts/{problem_id}.pt
    │   └── meta.json
    └── analysis/
        ├── inspect_signal.py
        └── REPORT_*.md
```

**원칙:** 이름에 spec / date / mode 를 박아둘 것. 기존 `analysis_qwen3_8b/activation/outputs/` 절대 덮어쓰지 말 것.

---

## 5. 환경 / 자주 쓰는 명령

```bash
# Python 가상환경
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python

# 모델 캐시 위치 (공유 ~/.cache 오염 방지)
export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb

# 노드 점유 확인
squeue -w iREMB-C-07     # 추론용 L40s ×4
squeue -w iREMB-C-03     # 학습용 H200 ×2
squeue -u $USER          # 내 job

# 내 job kill
scancel <jobid>
scancel -u $USER         # 내 전체 (긴급용)

# 라벨 / pilot 로드 sanity
$PY -c "
import pandas as pd
f = pd.read_parquet('src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet')
p = pd.read_parquet('src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet')
print('full :', f.shape, '| pilot:', p.shape)
print('pilot subj x level:'); print(pd.crosstab(p['subject'],p['level']))
"
```

### 5.1 권장 sbatch 헤더 (L40s 추론용)
```bash
#!/bin/bash
#SBATCH --job-name=act_smoke
#SBATCH --nodelist=iREMB-C-07
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00                  # smoke: 짧게!
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

set -euo pipefail
trap 'echo "[exit] $(date)"; echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT

export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb

PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
cd /scratch/lami2026/personal/jimin_2782
$PY src/OPSD_Curriculum/activation_v2/extract_<spec>.py --smoke --n 20
```

### 5.2 권장 sbatch 헤더 (H200 학습용, **나중 단계**)
```bash
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-03
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
```

---

## 6. 절대 수정 금지 / 보존 파일

이번 / 이전 세션 산출물 (재생성 비용 큼):
- `src/OPSD_Curriculum/labeling/outputs/*` (라벨, ~$9)
- `src/OPSD_Curriculum/labeling/label_openthoughts_30k.py` (prompt_sha 보존)
- `src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts/*.pt` (이전 non-thinking ΔA)
- `src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet`
- `src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/prompt_act/*.pt`
- `src/OPSD_Curriculum/analysis_qwen3_8b/nait/MASTER_SUMMARY_qwen3_8b.md` 및 모든 `REPORT*.md`
- `src/4.6_Task2/` 전체 (이전 1.5B 분석, import 만 OK)

새 작업은 모두 **새 디렉토리** (`activation_v2/`, `outputs_<spec>/` 등) 에서.

---

## 7. Quirks / 잊지 말 것

1. **MATH level convention**: 1=쉬움 ↔ 8=어려움. `ρ(pass_rate, level)` 은 음수가 정상.
2. **subject 8-canonical**: `Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus, Other`. 원본 LLM 출력 잔존은 `subject_raw` 에.
3. **problem_id**: `sha1(problem_text)[:16]`. 전체 29,434 unique 보장됨. cross-tool key 로 이걸 써야 안전.
4. **level 8 희소**: 전체에서 66개뿐 (0.22%). 일부 subject 에서 0. 분석 시 L8 단독 cell 결론 내리지 말 것 (n 부족).
5. **Prealgebra dead at L5+**: 정의상 정상.
6. **CSV embedded newlines**: `wc -l` 결과 ≠ row count. 항상 pandas/parquet 로 검증.
7. **OPENAI key**: `LAMI_OPENAI_API_KEY` envvar 로. 새 tmux 세션에서는 inheritance 확인 (`echo "${LAMI_OPENAI_API_KEY:0:7}"`).
8. **GPU OOM 발생 시**: 즉시 `scancel` 후 batch size 1로 축소 / `max_new_tokens` 줄여서 재시도. 무한 retry 금지.

---

## 8. 연구 컨텍스트 breadcrumb (생략 가능)

- 본 프로젝트는 **NAIT 기반 self-distillation curriculum** 설계 연구.
- 이전 세션: Qwen3-8B non-thinking ΔA 추출 → PC1 단일 축 신호 부족 → thinking-mode 재추출 가설 (`analysis_qwen3_8b/HANDOFF_THINKING_MODE.md`).
- 이번 세션: 라벨링 데이터 baseline 을 OpenThoughts-30K (29k) 로 확장.
- 다음 세션: 사용자가 정한 새 추출 방향으로 activation 재수집 + 위 라벨로 (subject, level, r1_cot_token_count, pass_rate 등) 분석.

연관 문서:
- `paper/25580_Neuron_Aware_Data_Select.pdf` — NAIT 원 논문
- `src/4.6_Task2/RESEARCH_NARRATIVE_PROMPT.md` — 전체 narrative
- `src/4.6_Task2/activation/analysis/full_final/MASTER_SUMMARY.md` — 1.5B 분석 결과
