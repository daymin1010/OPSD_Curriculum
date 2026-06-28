# 🧭 Session Hand-off: NAIT Self-Distillation Research (Qwen3-8B Track)

> Created: 2026-05-22
> Previous session: Pilot 2,666 measurement + validation (both Track A pass-rate & Track B activation-shifts) completed.
> Next session: Track C — NAIT analysis on Qwen3-8B (cluster discovery + curriculum stage definition).

## 0. Context (꼭 먼저 읽어줘)
- 연구 주제: **NAIT 기반 self-distillation을 위한 curriculum 설계**
- 핵심 신호: Qwen3-8B (non-reasoning) 의 **pass rate** (model-internal difficulty)
                + **activation shift ΔA = h_t1 - h_tK** (layer-wise residual)
- 분석 단계가 두 트랙 평행으로 끝났고, validation까지 통과한 상태.
- 다음 단계는 **NAIT analysis on Qwen3-8B (cluster discovery + curriculum stage definition)**.

## 1. Repository Map (절대 경로)
Base: `/scratch/lami2026/personal/jimin_2782/`

### 1.1 Pilot 2,666 결과물 (이번 세션 산출물)
- **Track A — Pass rate**:
  `src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet`
  (sample_id, pass_rate, pass_count, correct_indices, mean_response_length{,_correct,_incorrect},
   truncation_count, raw_responses[8], subject, level, ground_truth)
- **Track B — Activation shifts**:
  `src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts/`
   ├─ `{id}.pt` × 2666  → `shifts[layer]` ∈ ℝ^12288, bfloat16, 36 layers
   ├─ `shifts_metadata.jsonl`   ← row-level metadata
   └─ `shifts_checkpoint_chunk{0..3}.json`
- **Validation report**: `src/OPSD_Curriculum/analysis_qwen3_8b/validation/outputs/report.md`
  (TL;DR / Decision Sheet / plots / spot_check)

### 1.2 코드/스크립트
- Track A driver: `src/OPSD_Curriculum/analysis_qwen3_8b/pass_rate_measurement.py`
- Track B driver: `src/OPSD_Curriculum/analysis_qwen3_8b/activation/extract_activation_shifts_qwen3_8b.py`
- Validation: `src/OPSD_Curriculum/analysis_qwen3_8b/validation/validate_pilot_2666.py`
- sbatch:
   `src/OPSD_Curriculum/analysis_qwen3_8b/sbatch/{run_smoke_test_h200,run_pass_rate_pilot_h200,run_smoke_test_l40s,run_pass_rate_pilot_l40s}.sh`
   `src/OPSD_Curriculum/analysis_qwen3_8b/activation/sbatch/{run_smoke_test,run_pilot}.sh`

### 1.3 NAIT 분석 기존 utils (재사용; 절대 수정 금지)
- `src/4.6_Task2/activation/analysis/_nait_common.py`
   → `load_metadata(dirs)`, `resolve_shift_dirs(None)` 등
- `src/4.6_Task2/activation/extract_activation_shifts.py` (옛 모델용; 수정 금지)
- `src/4.6_Task2/activation/extract_h0.py` (수정 금지)
- 옛 manifest, 옛 2,648 parquet, `training/data/*`, `training/manifests/*` (수정 금지)

### 1.4 환경
- Python: `/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python`
  (verl 0.8.0 + vllm 0.19.0, torch 2.x, scipy, sklearn, pandas, pyarrow 모두 OK)
- HF 모델: `Qwen/Qwen3-8B` (이미 캐시됨)
- GPU 정책: VS Code 터미널에서 GPU 직접 호출 금지 → 반드시 sbatch
  · H200 2× (partition `h200q`, node `iREMB-C-03`, gres `gpu:2`, tp=2)
  · L40s 4× (node 07, gres `gpu:4`, tp=4)
  · MPS daemon 필요: `unset ROCR_VISIBLE_DEVICES && nvidia-cuda-mps-control -d`

## 2. 현재 상태 (Where we are)

### 2.1 Pilot universe (2,666 sample)
- 기존 NAIT 분석에서 정의된 unit-stratified pilot (subject × level 59 units).
- 로딩 방법 (이미 검증됨):
  ```python
  import sys; sys.path.insert(0, "src/4.6_Task2/activation/analysis")
  from _nait_common import load_metadata, resolve_shift_dirs, BASE_DIR
  from pathlib import Path
  df = load_metadata(
      resolve_shift_dirs(None) + [Path("src/4.6_Task2/activation/full_shifts_l7l8")]
  )
  ok = {"completed", "ok", "ok (skipped)"}
  df = df[df["status"].isin(ok)].drop_duplicates(subset="id")  # → 2666 rows
  ```

### 2.2 Pilot Validation 결과 (`validation/outputs/report.md`)
- Overall: **GO** ✅ (FAILs none, WARN B3는 휴리스틱 threshold 이슈)
- §5.7 Hybrid: **HYBRID=LAYERED**  (pass=0 ratio = 22.7%)
- A3 Signal: **ACCEPTABLE_A3=YES**  (Spearman ρ(pass, level) = **-0.507**, p<1e-173)
- X2 Cross-track: **X2_SIGNAL=STRONG**  (ρ(pass, mean_norm) = -0.091, ANOVA F=11.95 p=2.8e-13)
- §8.A Subject axis: **SUBJECT=GPT_LABEL**  (silhouette ≈ -0.015 → 활성화엔 subject 구조 약함)
- Track C 진입: **WAIT (사람 검토)**
- 40K 확장: **GO**

핵심 통계:
- pass=0 count = 606 (22.7%) → comprehension threshold 후보
- pass=1.0 count = 831 (31.2%) → trivial
- truncation 전체 5.1%, pass=0 bucket에서 32.2% 집중
- `<think>` tag 출현 0건 → non-reasoning chat template (enable_thinking=False) 정상
- Activation: mean ||ΔA|| ≈ 51, 36 layers 모두 nonzero, NaN/Inf 없음

### 2.3 알려진 quirks / 의사결정 포인트 (다음 세션에서 까먹지 마)
1. `_nait_common.load_metadata`의 `status` 값은 `"ok"` / `"ok (skipped)"` (옛날 `"completed"` 아님).
   필터는 `{"completed","ok","ok (skipped)"}` 로 해야 함.
2. `mean_response_length_correct`는 pass=0 sample에서 NaN, `_incorrect`는 pass=1 sample에서 NaN.
   구조적으로 정상이므로 null check에서 제외.
3. MATH `level`은 1=쉬움 ↔ 8=어려움. ρ(pass, level)이 **음수**여야 정상 신호 → `abs(rho)` 기준 판정.
4. B3 layer-norm threshold (validate script의 `<100` 휴리스틱)는 12288-dim 활성화에선 의미 없음 (실측 ~51). 무시 가능.

## 3. 다음 세션의 Task

### 3.1 (최우선) Track C — NAIT analysis on Qwen3-8B pilot
검증된 Track A + Track B 결과를 NAIT 분석 파이프라인에 통과시켜 **cluster discovery & curriculum stage 정의**.

#### 구체적 step:
1. **Load 통합 DataFrame**
   - row 단위: 2666 sample
   - 컬럼: `id, subject, level, pass_rate, truncation_count, mean_response_length, shifts_path`
   - 추가 활성화 derived feature:
     - per-layer norm (36-dim vector)
     - mid-layer mean (layers 15/20/25) ∈ ℝ^12288
     - 또는 layer-pooled embedding (e.g., last-token mean across all 36 layers)

2. **NAIT 분석 재사용**
   - 참고: `src/4.6_Task2/activation/analysis/nait_*.py` 파일들
     (e.g., `nait_within_unit_analysis.py`, `nait_unit_prototype_analysis.py`,
       `nait_per_level_residual.py`, `nait_stage_clustering.py`,
       `nait_make_curriculum.py`, `nait_linear_probe.py`, `nait_channel_separation.py`)
   - 새 디렉토리 `src/OPSD_Curriculum/analysis_qwen3_8b/nait/` 만들고 거기서 작업
   - 옛 파이프라인 그대로 import 하되 input dir 만 새 Qwen3 결과로 swap

3. **Curriculum stage definition**
   - 입력: pass_rate (model-internal difficulty) + activation cluster
   - 출력: stage 1~5 manifest (각 stage 별 sample id list + 의도)
   - 저장 위치: `src/OPSD_Curriculum/analysis_qwen3_8b/curriculum/stage_{i}_manifest.parquet`

4. **Sanity & Comparison vs. 옛 1.5B**
   - 옛 distilled-1.5B에서 도출된 stage와 Qwen3-8B에서 도출된 stage의 cluster overlap
   - C5 outlier (옛 ill-posed 18 sample, `full_final/C5_outlier_samples.json`) 가 pass=0 cluster에 모이는지 검증
     - 이번 session validation X2에서 일부 확인됨 (보고서 X2 섹션 참고)

### 3.2 (가능하면) Track A 40K 확장
Validation Decision Sheet에서 40K=**GO** 였음. 단:
- 우선순위는 Track C (NAIT) 완료 후
- 40K = full L2 dataset (40,315 sample) → 2666 → 40315 scaling
- 예상 wallclock: 2666 sample이 H200 2× 에서 ~3시간이었으므로 40K는 ~45시간 단일 job → chunk 분할 필수
- 사전 작업: chunked sbatch + resume-from-checkpoint 로직 (이미 Track B에 있는 4-chunk 방식 참조)

### 3.3 (보류) Track C 본 실행 전 점검 항목
Validation report에서 "WAIT (사람 검토)" 였던 이유:
- B3 layer-norm 휴리스틱 false WARN (실제론 PASS)
- 사람이 한 번 봐줘야 할 것: NAIT method가 Qwen3-8B 같은 더 큰 모델에서도 같은 가정 (layer windowing, mid-layer 신호)을 그대로 쓸 수 있는지
- → 다음 세션 시작 시 사용자가 "GO/NO-GO" 결정해주면 진행

## 4. 운영 규칙 (다시 확인)
- **GPU 작업은 sbatch만** (`sbatch <script.sh>`). VS Code 터미널에서 직접 GPU 호출 금지.
- 큐 경쟁 심함 → 가능한 적은 GPU. CPU-only로 가능한 분석은 CPU에서.
- **수정 금지** (재확인):
  - `src/4.6_Task2/activation/extract_activation_shifts.py`
  - `src/4.6_Task2/activation/extract_h0.py`
  - `src/4.6_Task2/activation/analysis/_nait_common.py`
  - 옛 manifest, 옛 2,648 parquet, `training/data/*`, `training/manifests/*`
- 새 작업은 항상 새 subdir (`src/OPSD_Curriculum/analysis_qwen3_8b/...` 안에서).

## 5. 첫 행동
새 세션을 시작하면:
1. 이 hand-off (`src/OPSD_Curriculum/analysis_qwen3_8b/HANDOFF_NEXT_SESSION.md`) 를 다 읽고 이해했음을 1-2줄로 확인.
2. `src/OPSD_Curriculum/analysis_qwen3_8b/validation/outputs/report.md` 의 TL;DR + Decision Sheet 를 한 번 읽어서 현재 상태 파악.
3. **§3.1 Track C — NAIT analysis 진입**을 위한 implementation plan을 짧게 (5-8줄) 정리해서 제시. 사용자 승인 받고 작업 시작.

## 6. 자주 쓰는 명령어
```bash
# Python (verl env)
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python

# Validation 빠른 재실행 (CPU, ~1min)
$PY src/OPSD_Curriculum/analysis_qwen3_8b/validation/validate_pilot_2666.py

# Pilot ID set 확인
$PY -c "
import sys; sys.path.insert(0,'src/4.6_Task2/activation/analysis')
from _nait_common import load_metadata, resolve_shift_dirs
from pathlib import Path
df = load_metadata(resolve_shift_dirs(None)+[Path('src/4.6_Task2/activation/full_shifts_l7l8')])
df = df[df['status'].isin({'completed','ok','ok (skipped)'})].drop_duplicates(subset='id')
print(len(df))
"

# parquet 미리보기
$PY -c "
import pandas as pd
df = pd.read_parquet('src/OPSD_Curriculum/analysis_qwen3_8b/outputs/pass_rate_pilot_2666.parquet')
print(df.shape); print(df.head(2)); print(df['pass_rate'].describe())
"

# Track B sample 한 개 로드
$PY -c "
import torch, json
from pathlib import Path
fp = next(Path('src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts').glob('*.pt'))
s = torch.load(str(fp), map_location='cpu', weights_only=False)
print('keys:', list(s.keys()))
print('layer0 shape/dtype:', s['shifts'][0].shape, s['shifts'][0].dtype)
print('num layers:', len(s['shifts']))
"
```

## 7. 참고 문서 (background)
- `paper/25580_Neuron_Aware_Data_Select.pdf` (NAIT 논문)
- `src/4.6_Task2/RESEARCH_NARRATIVE_PROMPT.md` (전체 연구 narrative)
- `src/4.6_Task2/HANDOFF_2026-05-17_v{2,3,4}.md` (옛 hand-off 문서들)
- `src/4.6_Task2/activation/analysis/full_final/MASTER_SUMMARY.md` (옛 1.5B 분석 결과)

---

준비됐으면 §5 첫 행동부터 시작해줘.
