# Hand-off — Phase 0 DONE → Phase 1 (pooled) 시작 직전

작성: 2026-06-12 / reasoning_pivot activation, pooled(pilot1+pilot2) 분석 트랙

## 필독 선행 문서 (우선순위 순)
1. **`src/OPSD_Curriculum/HANDOFF_2026-05-30_activation_extraction.md`** — 운영환경·공용계정·GPU 가이드 **원본** (읽기 전용, 절대 수정 금지). 아래 §A에 핵심을 인라인 복제해 두었지만, 분쟁/모호 시 이 원본이 기준.
2. **`analysis/N_AUDIT.md`** — Phase 0 수치 (N/L8/분포) 정본.
3. **`analysis/MASTER_SIMILARITY_FINDINGS.md`** + `compare_pilot1_pilot2.py` 산출물(`REPORT_pilot2_comparison.md`) — 이전 유사도 분석.
4. 연구 narrative: `paper/25580_Neuron_Aware_Data_Select.pdf` (NAIT 원논문), `src/4.6_Task2/RESEARCH_NARRATIVE_PROMPT.md`, `src/OPSD_Curriculum/analysis_qwen3_8b/nait/MASTER_SUMMARY_qwen3_8b.md`, `analysis_qwen3_8b/HANDOFF_THINKING_MODE.md`.

## 0. 한 줄 요약
- Phase 0 audit 완료. **canonical N = 3025 (finite), non-finite = 0**.
- **L8 = 66개 존재** (전부 pilot1, pilot2=0) → 임계 20 초과 → **L40s GPU L8 추출 분기 미발동(불필요)**.
- 다음 세션 mission: **Phase 1 pooled 통합 분석** (CPU only, GPU 안 씀).

## A. 운영 환경 — 반드시 먼저 (원본: HANDOFF_2026-05-30…md)
**공용 계정 / 디렉토리**
- `lami2026` 은 **여러 사람이 공유하는 user 계정**. `/home/lami2026/`, `/scratch/lami2026/` 대부분 공유 자원.
- **모든 작업물은 `/scratch/lami2026/personal/jimin_2782/` 내부에서만** 생성/수정.
- 공유 `~/.bashrc`, `~/.cache/huggingface/`, 타 personal 디렉토리 **수정 금지(읽기만 OK)**.

**GPU 노드 정책**
| 용도 | 노드 | 자원 | flag |
|---|---|---|---|
| 추론(activation 추출, pass-rate) | `iREMB-C-07` | L40s 48GB ×4 | `--nodelist=iREMB-C-07 --gres=gpu:N` |
| 학습(SFT/RL, 이후 단계) | `iREMB-C-03` | H200 80GB ×2 | `--nodelist=iREMB-C-03 --gres=gpu:2 --partition=h200q` |
- **`iREMB-C-02/04/05/06` 절대 사용 금지. `--exclusive` 금지.** 필요한 GPU 수만 정확히 요청.
- GPU 작업은 **`sbatch`만** (VS Code 터미널에서 `python --device cuda` 직접 호출 금지).
- sbatch 헤더 필수: `--nodelist --gres --time --output --error`. TIME LIMIT 짧게(smoke ETA×2 상한).

**매 GPU job 제출 전 5-점 safety checklist**
1. 출력 디렉토리가 새 경로인가(기존 산출물 안 덮나)
2. 헤더에 `--nodelist/--gres/--time` 모두 있는가
3. `squeue -w iREMB-C-07` 로 타 user 점유 확인(4장 다 차면 대기·강제 인터럽트 금지, 2장만 비면 `--gres=gpu:2` 로 양보)
4. smoke(≤30 sample, ≤15분) 먼저 통과했는가
5. 새 다운로드 위치가 `cache/huggingface` 인가(공유 `~/.cache` 오염 금지)

**one-strike 금지**: `sudo`, `chmod 777`, 공유 `~/.bashrc`·`~/.cache`·`~/.config/wandb` 수정, `srun --pty` GPU 30분+ 점유, 시스템 python 에 `pip install`(반드시 `envs/verl_new/bin/pip`), 타 personal 디렉토리 read 외 접근, 금지 노드 사용.

**환경 변수 / 경로**
```bash
PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_DIR=/scratch/lami2026/personal/jimin_2782/wandb
squeue -w iREMB-C-07   # 점유 확인 | scancel -u $USER (긴급)
```

## B. 연구 컨텍스트 breadcrumb
- 프로젝트: **NAIT 기반 self-distillation curriculum** 설계 연구.
- 흐름: (1.5B 분석 `src/4.6_Task2/`) → Qwen3-8B non-thinking ΔA 추출(PC1 신호 부족) → **thinking-mode 재추출 가설**(`analysis_qwen3_8b/HANDOFF_THINKING_MODE.md`) → OpenThoughts-30K 라벨링($9, `labeling/`) → **reasoning_pivot: thinking/faithful ΔA pilot1+pilot2 추출** → (현재) **pooled 3025 라벨기반 분석**.
- MATH level: 1=쉬움↔8=어려움 (ρ(pass_rate,level) 음수가 정상). subject 8-canonical. problem_id=sha1(text)[:16].

## C. 보존 금지 파일 (재생성 비용 큼)
- `labeling/outputs/*` (라벨 ~$9), `labeling/label_openthoughts_30k.py`
- `reasoning_pivot/activation/outputs/pilot*/shifts/*.pt` (pilot1 1608 + pilot2 1417 ΔA)
- 이 분석 산출물: `analysis/N_AUDIT.md`, `analysis/MASTER_SIMILARITY_FINDINGS.md`, `analysis/REPORT*.md`
- `analysis_qwen3_8b/.../*.pt`·`MASTER_SUMMARY*`, `src/4.6_Task2/` 전체(import만 OK)
- 새 작업은 전부 **새 파일명(spec/date 박기)**.


## 1. Phase 0 결과 (`analysis/N_AUDIT.md` 참조 — 절대 덮어쓰지 말 것)
| | raw .pt | non-finite | finite N |
|---|---|---|---|
| pilot1 | 1608 | 0 | 1608 |
| pilot2 | 1417 | 0 | 1417 |
| **pooled** | **3025** | **0** | **3025** |

- "1541 vs 1608": non-finite=0 이므로 1541은 6/3 이전 **이른 스냅샷**. 현재 3025가 canonical. (NaN 가설 아님)
- pilot1 mtime: 2026-06-03 00:57 ~ 06-04 12:12 (span 35h).
- L8 분포(pooled): Number Theory 23 / Other 13 / Intermediate Algebra 11 / Counting&Probability 10 / Geometry 9. **Algebra·Prealgebra·Precalculus = L8 0** (정상, handoff quirk #4).
- level 카운트: L1 335, L2 480, L3 480, L4 437, L5 420, L6 420, L7 387, L8 66.
- Prealgebra는 L4=17, L5+ = 0 (정의상 정상).

## 2. 데이터/로더 사실 (다음 세션이 그대로 쓸 것)
- 로더: `analysis/similarity_analysis.py` 의 `sa.load_pilot(shifts_dir, max_n)` → `(DAF, DAT, md)`.
  - `DAF` = FAITHFUL ΔA, `DAT` = THINKING ΔA. `md` 컬럼: `subject, level, unit` (+ 기타).
  - 로더는 content filter 없음. .pt 로드에 pilot1 ~101s, pilot2 ~동급. 둘 합쳐 ~3분.
- shifts 경로:
  - pilot1: `outputs/pilot/shifts/*.pt` (1608)
  - pilot2: `outputs/pilot2/shifts/*.pt` (1417)
- 재사용 메트릭 함수(검증됨): `sa.normalize_members, sa.centroids, sa.sim_matrix, sa.within_between, sa.perm_pvalue, sa.spearman, sa.MIN_N, sa.N_PERM`.

## 3. Phase 1 — pooled 통합 분석 (메인, canonical)
**설계 (replication과 반대 handling 임에 주의):**
- pooled = pilot1 ⊕ pilot2 (3025) 를 **전체 공통평균(μ_pooled)** 으로 centering.
- 모드 2개: THINKING(DAT), FAITHFUL(DAF).
- 그룹 3개: subject / level / unit. 각: within / between / **gap** / perm-p (+ level은 ordinality ρ).
- perm budget: subject·level 1000, unit 200 (compare_pilot1_pilot2.py 와 동일 관례).
- **L8 cell (n=66, 5/8 subject만 존재)**: 표에는 포함하되 **단독 결론 금지** (n 부족 + subject 불균형 명시). level grouping 시 L1–L8 전체 vs L1–L7 둘 다 보고 권장.
- 출력: `analysis/REPORT_pooled_3025.md` (canonical). 기존 REPORT 덮어쓰지 말 것.

**구현 팁:** `compare_pilot1_pilot2.py` 의 `metrics_for_pilot()` 를 거의 그대로 쓰되,
(a) 두 pilot을 concat 후 한 번만 centering, (b) LEVEL 필터를 L1–L8(또는 옵션) 로, (c) per-pilot self-mean이 아니라 pooled-mean 사용.

## 4. Phase 2 — replication 부록
- 이미 작성된 `compare_pilot1_pilot2.py` 실행(L1–L7 공통범위, 각 self-center). 산출 `REPORT_pilot2_comparison.md`.
- pilot2에는 L8 없음 → replication은 L1–L7만. pooled(Phase1)이 L8 포함 메인.

## 5. Phase 3 — 최종 통합 리포트
- canonical N 표기: **raw 3025 / finite 3025 / analysis-N(필터별)** 병기. "3000"은 별칭일 뿐.
- L8 출처(전부 pilot1) 명시.

## 6. 운영 제약 (변함없음)
- GPU 불필요(이 트랙 CPU). 만약 추후 GPU 필요해지면 `iREMB-C-07` L40s, sbatch만, smoke→본런, `squeue -w iREMB-C-07` 점유 확인.
- 작업물은 `/scratch/lami2026/personal/jimin_2782/` 내부, 기존 산출물(outputs/pilot*, REPORT*, N_AUDIT.md) 덮어쓰기 금지. 새 파일명에 spec 박기.
- PY=`/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python`.

## 7. 새 세션 첫 행동
1. 이 파일 + `analysis/N_AUDIT.md` 통독.
2. `analysis/audit_pooled_N.py`(Phase0)·`compare_pilot1_pilot2.py`(Phase2 베이스) 확인.
3. Phase 1 pooled 스크립트 작성(`analysis/pooled_analysis.py`) → 실행 → `REPORT_pooled_3025.md`.
