# 🧭 Session Hand-off: Qwen3-8B Thinking-mode Activation Re-extraction

작성일: 2026-05-25
이전 hand-off: `src/OPSD_Curriculum/analysis_qwen3_8b/HANDOFF_NEXT_SESSION.md`

---

## 0. ⚠️ 운영 환경 — **반드시 먼저 읽기**

### 0.1 공동 user 계정
- **`lami2026` 은 공동 user 계정입니다.** `/home/lami2026/`, 그리고 system-wide 경로 (`/scratch/lami2026/`, 공용 outputs 등) 가 여러 사람과 공유됨.
- **모든 작업물은 반드시 `/scratch/lami2026/personal/jimin_2782/` 내부에서만** 생성/수정.
  - 특히 새 파일은 `/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/analysis_qwen3_8b/` 하위에만.
- `~/` (= `/home/lami2026/`) 의 파일, 공유 `~/.cache`, 공유 `~/.bashrc` 같은 건 **읽기만**. 절대 수정/덮어쓰기 금지.
- 다른 사용자의 파일 (예: `/scratch/lami2026/personal/other_user/...`) 절대 건드리지 말 것.

### 0.2 GPU 노드 제한
**반드시 다음 두 노드만 사용:**
- **`iREMB-C-03`** : H200 80GB × 2 (partition `h200q`, `--gres=gpu:2`)
- **`iREMB-C-07`** : L40s 48GB × 4 (partition는 클러스터 기본, `--gres=gpu:4`)

다른 노드 (`iREMB-C-02`, `04`, `05`, `06` 등) 는 **사용 금지**.
모든 sbatch 스크립트에 다음을 명시:
```bash
#SBATCH --nodelist=iREMB-C-03      # 또는 iREMB-C-07
#SBATCH --gres=gpu:2               # 또는 gpu:4
```

### 0.3 GPU 호출 방식
- **VS Code 터미널에서 직접 `python ... --device cuda` 실행 금지** (compute 노드 외부).
- 모든 GPU 작업은 **`sbatch <script.sh>` 만** 사용.
- MPS daemon 필요 시: `unset ROCR_VISIBLE_DEVICES && nvidia-cuda-mps-control -d` 를 sbatch 스크립트 내부에 포함.

---

## 1. 연구 컨텍스트 (요약)

- **주제**: NAIT 기반 self-distillation을 위한 curriculum 설계.
- **Track A** (pass rate, model-internal difficulty): Qwen3-8B non-reasoning, 8-rollout, 2666 pilot — **완료**.
- **Track B** (ΔA = h_tK − h_t1, activation shift): Qwen3-8B non-reasoning, 36 layer × 12288, 2666 pilot — **완료**.
- **NAIT 3-phase (Phase 0/A/Critical)**: 완료. → `nait/MASTER_SUMMARY_qwen3_8b.md`

## 2. 현재 결과 핵심

### 2.1 Non-thinking 모드에서의 발견
- ΔA(non-thinking) 의 PC1 / supervised LDA / ridge 결과:
  - subject F1 ≈ 0.40 ± (보고서 참조), level F1 ≈ 0.30, pass_rate R² 작음
- Critical baseline `A_prompt` (= prompt만 forward 한 후 마지막 prompt 토큰의 36-layer hidden activation):
  - 위 모든 metric 에서 ΔA보다 약간 **더 좋음** (gap 모두 −0.01 ~ −0.03)
- → **non-thinking ΔA 가 A_prompt 위에 정보를 거의 추가하지 못함**.

### 2.2 가설
사용자의 prior 경험에서: **thinking mode 로 ΔA 를 추출했을 때는 PC1 단일 축에 level·subject 가 둘 다 잡혔음.**
→ **non-thinking 모드가 ΔA 신호를 죽인 것이 결정적 원인** 이라는 가설.

이유:
- non-thinking 은 생성이 짧음 (median 수백 토큰) → ΔA 가 "답안 포맷 잔차" 수준
- thinking 은 수천 토큰의 reasoning trajectory 가 ΔA 에 누적 → 풍부한 model-internal 신호

## 3. 다음 세션 Task — Thinking-mode Activation Re-extraction

같은 2666 pilot universe 에 대해 Qwen3-8B 를 **CoT mode (`enable_thinking=True`)** 로 돌려 activation shift 재추출.

### 3.1 구체 step
1. **새 extraction script**: `activation/extract_activation_shifts_qwen3_8b_thinking.py`
   - 기존 `activation/extract_activation_shifts_qwen3_8b.py` 를 복사
   - `tokenizer.apply_chat_template(..., enable_thinking=True)` 로 변경
   - 출력 dir 분리: `activation/outputs_thinking/shifts/{id}.pt`
   - checkpoint/metadata 파일명도 분리 (예: `..._thinking_chunk{i}.json`)
   - generation 길이 길어지므로 `max_new_tokens` 충분히 (예: 8192 또는 16384) 확인 필요
2. **Sbatch 4-chunk 분할**: `activation/sbatch/run_pilot_thinking_chunk{0..3}.sh`
   - `--nodelist=iREMB-C-03 --gres=gpu:2` (H200 2×, tp=2)
   - chunk 당 wallclock 예상 3–5 h → 4 chunk 병렬이면 큐 형편에 따라 반나절~1일
   - non-thinking 대비 ~3–5배 느릴 것
3. **Smoke test 먼저** (10–20 sample) — chat template `<think>` block 정상 생성/종료 검증
4. **3-phase NAIT 재분석**
   - `nait/direction_calibrated.py`, `supervised_direction.py`, `analyze_critical.py` 를 thinking 데이터로 재실행
   - 출력: `nait/outputs_thinking/REPORT*.md`
5. **비교 리포트**: non-thinking vs thinking
   - 같은 metric (subject F1, level F1, pass_rate R², ΔA vs A_prompt gap)
   - PC1 단일 축에 level/subject 같이 잡히는지 정성 확인

### 3.2 재사용 가능한 자산 (재추출 불필요)
- **A_prompt** (`nait/outputs/prompt_act/*.pt`): prompt-only forward 이므로 thinking 여부 무관.
- **pass_rate** (`outputs/pass_rate_pilot_2666.parquet`): non-thinking 8-rollout 결과. curriculum 의 difficulty signal 로 그대로 사용.
- **pilot universe (id list)**: `_nait_common.load_metadata` 로 2666 그대로.

### 3.3 첫 행동 (다음 세션 시작 시)
1. 이 hand-off + `nait/MASTER_SUMMARY_qwen3_8b.md` 읽기.
2. `activation/extract_activation_shifts_qwen3_8b.py` 코드 훑기 (hook 구조, prompt build).
3. `extract_activation_shifts_qwen3_8b_thinking.py` 초안 작성 → smoke 10 sample sbatch.
4. smoke 확인 후 4-chunk pilot sbatch 제출.

---

## 4. 절대 수정 금지 파일

- `src/4.6_Task2/activation/extract_activation_shifts.py` (옛 1.5B 용)
- `src/4.6_Task2/activation/extract_h0.py`
- `src/4.6_Task2/activation/analysis/_nait_common.py` (단, import 만 OK)
- 옛 manifest, 옛 2,648 parquet, `src/4.6_Task2/training/data/*`, `training/manifests/*`
- 이전 세션 산출물:
  - `src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts/*.pt` (non-thinking, 보존)
  - `src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet`
  - `src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/prompt_act/*.pt`
  - `nait/MASTER_SUMMARY_qwen3_8b.md` 및 REPORT*.md 들

새 작업은 모두 `outputs_thinking/`, `nait/outputs_thinking/` 등 **별도 디렉토리에서**.

---

## 5. 환경 / 자주 쓰는 명령어

```bash
# Python (verl env, 모든 분석/추출에 사용)
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python

# Pilot ID set 확인 (CPU, 즉시)
$PY -c "
import sys; sys.path.insert(0,'src/4.6_Task2/activation/analysis')
from _nait_common import load_metadata, resolve_shift_dirs
from pathlib import Path
df = load_metadata(resolve_shift_dirs(None)+[Path('src/4.6_Task2/activation/full_shifts_l7l8')])
df = df[df['status'].isin({'completed','ok','ok (skipped)'})].drop_duplicates(subset='id')
print(len(df))   # → 2666
"

# Non-thinking shift sample 로드 (참고용)
$PY -c "
import torch
s = torch.load('src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts/00001.pt',
               map_location='cpu', weights_only=False)
print(list(s.keys())); print(s['shifts'][0].shape, s['shifts'][0].dtype)
"

# sbatch 제출 (예시)
sbatch src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_pilot_thinking_chunk0.sh
squeue -u $USER
```

### 5.1 기존 sbatch 헤더 예시 (재사용)
```bash
#!/bin/bash
#SBATCH --job-name=qwen3_8b_act_thinking_c0
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-03
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.out
#SBATCH --error=/scratch/lami2026/personal/jimin_2782/runs/slurm-%x.%j.%N.err

unset ROCR_VISIBLE_DEVICES
nvidia-cuda-mps-control -d || true
```

---

## 6. 파일 경로 인덱스 (Base = `/scratch/lami2026/personal/jimin_2782/`)

### 6.1 핵심 결과물 (보존)
- Pass rate: `src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet`
- Non-thinking ΔA: `src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts/{id}.pt`
- A_prompt: `src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/prompt_act/{id}.pt`
- Validation: `src/OPSD_Curriculum/analysis_qwen3_8b/validation/outputs/report.md`
- NAIT MASTER SUMMARY: `src/OPSD_Curriculum/analysis_qwen3_8b/nait/MASTER_SUMMARY_qwen3_8b.md`

### 6.2 코드 (재사용/참고)
- Non-thinking extractor (복사 대상): `src/OPSD_Curriculum/analysis_qwen3_8b/activation/extract_activation_shifts_qwen3_8b.py`
- Prompt-act extractor: `src/OPSD_Curriculum/analysis_qwen3_8b/nait/extract_prompt_activation.py`
- NAIT 분석: `src/OPSD_Curriculum/analysis_qwen3_8b/nait/{direction_calibrated, supervised_direction, analyze_critical}.py`
- 공용 utility: `src/4.6_Task2/activation/analysis/_nait_common.py` (import 만, 수정 금지)

### 6.3 새로 만들 작업 경로 (예시)
- Extractor: `src/OPSD_Curriculum/analysis_qwen3_8b/activation/extract_activation_shifts_qwen3_8b_thinking.py`
- Sbatch: `src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/run_smoke_thinking.sh`, `run_pilot_thinking_chunk{0..3}.sh`
- Shift 출력: `src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs_thinking/shifts/{id}.pt`
- 분석 출력: `src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs_thinking/REPORT*.md`

---

## 7. Quirks / 잊지 말 것

1. `_nait_common.load_metadata` 의 `status` 값은 `"ok"` / `"ok (skipped)"` (옛 `"completed"` 도 포함). 필터: `{"completed","ok","ok (skipped)"}`.
2. MATH `level` 은 1=쉬움 ↔ 8=어려움. `ρ(pass, level)` 음수가 정상.
3. Thinking mode에선 `<think>...</think>` block 이 생성에 포함됨. ΔA 계산 시 첫 번째 토큰(t1)/마지막 토큰(tK) 위치를 어떻게 정할지 결정 필요:
   - 가장 단순: 기존 코드 그대로 (chat template 출력의 첫 generation token / 마지막 token).
   - 또는: `<think>` 안만, `</think>` 이후만 등 변형 가능. **첫 구현은 단순 그대로 → 결과 보고 분기.**
4. Generation 길이가 길어지므로 OOM 주의: H200 80GB 면 보통 OK 지만, `max_new_tokens` 와 batch size 단일(1) 유지 권장.
5. Chat template 검증: prompt 안에 비어있지 않은 `<think>...</think>` 가 미리 들어가 있으면 안 됨 (Qwen3 의 enable_thinking 동작 확인).

---

## 8. 참고 문서
- `paper/25580_Neuron_Aware_Data_Select.pdf` (NAIT 논문)
- `src/4.6_Task2/RESEARCH_NARRATIVE_PROMPT.md` (전체 연구 narrative)
- `src/4.6_Task2/activation/analysis/full_final/MASTER_SUMMARY.md` (옛 1.5B 분석 결과)
- `src/OPSD_Curriculum/analysis_qwen3_8b/HANDOFF_NEXT_SESSION.md` (이전 세션 hand-off)

---

**다음 세션의 첫 행동**:
1. 이 문서 + `nait/MASTER_SUMMARY_qwen3_8b.md` 통독 → 1–2 줄로 이해 확인.
2. `activation/extract_activation_shifts_qwen3_8b.py` 코드 훑기.
3. `_thinking.py` 초안 작성 → smoke 10 sample sbatch (iREMB-C-03).
4. smoke 결과 (ΔA shape, <think> 출현 확인) 보고 후 4-chunk pilot sbatch 제출.
