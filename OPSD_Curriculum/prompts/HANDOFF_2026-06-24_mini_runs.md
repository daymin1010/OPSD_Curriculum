# OPSD 커리큘럼 실험 — 핸드오프 (2026-06-24 03:06)

## 운영/안전 (엄수)
- 작업 디렉토리: `/scratch/lami2026/personal/jimin_2782`. opsd env: `envs/opsd/bin/python`. 본인 JOBID/NAME(2782)만 조작.
- `lami2026` 공용계정. **sbatch만**, `--exclusive` 금지, GPU≤4, **L40S 우선**. 노드: iREMB-C-07(L40S, `l40sq`) / iREMB-C-03(H200, `h200q`).
- sudo·chmod777·시스템 pip·공유 dotfile 금지. upstream `opsd_src/` 직접 수정 금지(fork됨).
- **추측 코드 금지**, 큰 로그 전체 출력 금지(grep/tail). 다른 사람 job scancel 절대 금지.

---

## 1. 현재 상황 (2026-06-24 03:06 기준)

### Slurm job 현황
```bash
squeue -u lami2026 --format="%.10i %.40j %.8T %.10M %R"
```

| Job | 이름 | 상태(예상) | 노드 | 비고 |
|---|---|---|---|---|
| 87995 | `opsd_quarter_cond3_ours_q4_h200` | RUNNING (~step 70/228) | C-03 | q4 ours, 진행 중 |
| 87996 | `opsd_quarter_cond2_diff_q4_h200` | RUNNING (~step 60/228) | C-03 | q4 diff, 진행 중 |
| 88005 | `opsd_mini50_cond3_ours_h200` | PENDING | C-03 대기 | mini50 ours |
| 88006 | `opsd_mini50_cond2_diff_h200` | PENDING | C-03 대기 | mini50 diff |
| 88007 | `opsd_mini100_cond3_ours_h200` | PENDING | C-03 대기 | mini100 ours |
| 88008 | `opsd_mini100_cond2_diff_h200` | PENDING | C-03 대기 | mini100 diff |
| 87999 | `qeval_ours900` | RUNNING/완료 | C-07 | full step900 eval (ours) |
| 88000 | `qeval_diff900` | RUNNING/완료 | C-07 | full step900 eval (diff) |

### 예상 타임라인
- **q4 완료**: ~6:00-6:15 AM (step 228, ~1.3 min/step)
- **mini50 완료**: ~7:40 AM (q4 종료 후 자동 시작, T≈52/53, ~1.4h)
- **mini100 완료**: ~10:05 AM (mini50 종료 후, T≈102/103, ~2.4h)
- **전체 완료**: 오늘 오전 10시경

> C-03은 GPU 4장 → q4(2+2 GPU) 종료 후 mini50 쌍(2+2) 동시 실행, mini50 종료 후 mini100 쌍 실행.

---

## 2. 이번 세션에서 만든 파일

### Manifest (이미 생성됨, `stages_tiered_20260622/` 아래)
- `stages_cond2_diff_mini50.json` / `stages_cond3_ours_C2_mini50.json` — N=1603, T≈52/53
- `stages_cond2_diff_mini100.json` / `stages_cond3_ours_C2_mini100.json` — N=3198, T≈102/103
- `stages_cond2_diff_q4.json` / `stages_cond3_ours_C2_q4.json` — N=7193, T≈227/228 (기존)
- `stages_cond2_diff.json` / `stages_cond3_ours_C2.json` — N=28771, T≈900/901 (기존)

### Config (`curriculum/configs/`)
- `mini50_8b_h200.yaml` — save_steps=10, save_total_limit=8
- `mini100_8b_h200.yaml` — save_steps=20, save_total_limit=8
- `quarter_8b_h200.yaml` (기존) — save_steps=10, save_total_limit=30
- `full_8b_h200.yaml` (기존)

### Training sbatch (`curriculum/sbatch/`)
- `mini50_cond3_ours_h200.sh` (88005, port 12981)
- `mini50_cond2_diff_h200.sh` (88006, port 12982)
- `mini100_cond3_ours_h200.sh` (88007, port 12983)
- `mini100_cond2_diff_h200.sh` (88008, port 12984)
- 각 sbatch는 시작 시 `verify_schedule_manifest_once.py`로 두 arm universe 일치 검증 후 학습 진입

### Eval sbatch (`curriculum/sbatch/`, 제출 보류)
- `eval_mini50_cond3_ours_l40s.sh` — step 10/20/30/40/50 × {aime24, aime25, hmmt25, math500}
- `eval_mini50_cond2_diff_l40s.sh` — 동일
- `eval_mini100_cond3_ours_l40s.sh` — step 20/40/60/80/100 × 4 datasets
- `eval_mini100_cond2_diff_l40s.sh` — 동일
- L40S(C-07) ×2, TP=2, non-thinking, val_n=3(aime/hmmt)/1(math500), temp=1.0
- checkpoint 없으면 SKIP 처리 (안전 장치 포함)

### 감사/분석
- `curriculum/audit_manifests.py` — manifest 공정성 검증 스크립트
- `prompts/MANIFEST_AUDIT_2026-06-24.md` — 감사 결과 보고서 (executive summary + 상세 데이터)

---

## 3. 검증 결과 요약

### A/B 공정성 (MANIFEST_AUDIT 확인)
- **각 rung 내부 diff vs ours**: 완전히 동일한 problem_id 집합 사용 (MD5 매칭, Jaccard=1.0, symdiff=0)
- **난이도/subject 분포**: q4/full≈0.24, mini100/q4≈0.44, mini50/q4≈0.22 — 균일 subsampling, 분포 보존 ✓
- **Nesting**: q4⊂full ✓, mini100⊂q4 ✓, mini50⊂q4 ✓

### mini50 ⊄ mini100 (주의사항)
- mini50과 mini100은 q4에서 **독립적** stratified 샘플링 → 서로 다른 문제 집합
- 교집합: 700/1603 (44%), Jaccard=0.17
- **rung 내부 A/B에는 무영향** (각 rung의 diff vs ours는 동일 문제 사용)
- **scale-ladder 비교 시 주의**: mini50@step50와 mini100@step50는 다른 문제를 본 것
- 보고서에 "mini50과 mini100은 q4의 독립 stratified subsample (44% 공유)"라고 명시

### OPSD 논문 기준 설정 확인 (타 LLM 검증)
- `teacher_thinking=True`, `student_thinking=False` → **논문 main 설정** (Table 5, §4.3.2, line 1101)
- `jsd_token_clip=0.06` → 논문이 τ를 튜닝하지 않았다고 명시 (line 1102-1104), 값 자체는 정당
- 논문 학습 예산: **100-step, 매 20 step eval, best 보고, Avg@12** (line 610/624/1104)
- full T=900은 논문 예산의 **9배** = over-training 구간 → final eval은 "over-train 끝점"으로만 해석

### Schedule 정확치 (실측, chat 요약 정정)
| rung | arm | T_total | stage 경계 (cum_T) |
|---|---|---|---|
| full | diff | 900 | 180→360→540→720→900 |
| full | ours | 901 | 166→358→538→724→901 |
| q4 | diff | 227 | 43→88→136→183→227 |
| q4 | ours | 228 | 42→90→135→182→228 |
| mini100 | diff | 102 | 19→39→61→82→102 |
| mini100 | ours | 103 | 19→41→61→82→103 |
| mini50 | diff | 52 | 10→20→31→42→52 |
| mini50 | ours | 53 | 10→21→31→42→53 |

> `tail_policy=partial`로 매 stage `ceil(n/32)` step → 5 stage×~1 extra ≈ +5

---

## 4. 학습 설정 (모든 rung 공통)

| 항목 | 값 |
|---|---|
| 모델 | Qwen3-8B + LoRA r=64, alpha=128 |
| GPU | H200 ×2 (pd=2, ga=8, ws=2 → B_glob=32) |
| LR | 5e-6, max_grad_norm=0.1 |
| max_completion_length | 1024 |
| temperature / top_p / top_k | 1.1 / 0.95 / 20 |
| beta / lmbda | 0 / 1 (forward KL) |
| fixed_teacher | True |
| teacher_thinking | True (논문 main 설정) |
| student_thinking | False |
| jsd_token_clip | 0.06 |
| curriculum_passes | 1 |
| within_stage_order | shuffle (seed=42) |
| tail_policy | partial |

---

## 5. 다음 세션에서 할 작업 (우선순위)

### 1순위: job 상태 확인
```bash
squeue -u lami2026 --format="%.10i %.40j %.8T %.10M %R"
```

### 2순위: mini50/mini100 학습 완료 확인 후 eval 제출
- checkpoint 위치: `checkpoints/opsd_curriculum/mini50_8b/mini50_cond{2,3}_*_h200/checkpoint-{10,20,...}`
- mini100: `checkpoints/opsd_curriculum/mini100_8b/mini100_cond{2,3}_*_h200/checkpoint-{20,40,...}`
- eval sbatch 제출:
```bash
cd src/OPSD_Curriculum/training/curriculum/sbatch
sbatch eval_mini50_cond3_ours_l40s.sh
sbatch eval_mini50_cond2_diff_l40s.sh
sbatch eval_mini100_cond3_ours_l40s.sh
sbatch eval_mini100_cond2_diff_l40s.sh
```
- L40S(C-07)에서 실행, GPU 2장 × 4 job (2개씩 직렬 가능)
- 예상 시간: mini50 5ckpt×4ds ≈ 3-4h, mini100 5ckpt×4ds ≈ 5-6h

### 3순위: q4 checkpoint별 eval
- q4 checkpoint: `checkpoints/opsd_curriculum/quarter_8b/quarter_cond{2,3}_*_q4_h200/checkpoint-{10,20,...,220}`
- 기존 `eval_quick_quarter_cond{2,3}_ours_l40s.sh`는 step 10/220만 커버
- 전체 step eval 필요 시 새 sbatch 작성 (step 10,20,30,40,50,60,80,100,130,160,190,220)

### 4순위: 원본 900-step run eval 결과 수집
- job 87979/87994(step900 eval) 완료 대기/확인
- job 87966/87969(step810 eval) 결과 확인
- base Qwen3-8B 성능과 비교

### 5순위: 결과 분석
- step별 성능 곡선 작성 (mini50, mini100, q4)
- 피크 step 탐지
- diff vs ours 비교 (같은 step에서)
- per-stage 학습 곡선 = H2 전제 검증 ("stage마다 학습 이득이 다른가")

### 6순위: 디스크 정리 (나중에)
- `checkpoints/opsd_curriculum/full_8b/` 아래 불필요 checkpoint 정리
- `eval_ckpts/` 아래 중복 어댑터 정리

---

## 6. 주의사항

1. **full T=900은 over-train 구간**: final adapter eval은 "deep over-training 끝점" 정보로만 취급. 신호는 ~100 step에 있으나 그 checkpoint는 `save_total_limit` 때문에 삭제됨.
2. **mini50 ⊄ mini100**: 두 rung은 독립 subsample. rung 내부 A/B는 공정하지만 rung 간 비교 시 문제-level noise 존재.
3. **eval val_n=3**: aime24/aime25는 30문제 × 3 = 90 trial → ±5% stderr. 두 arm 차이가 작으면 통계적 분리 어려울 수 있음.
4. **q4 진행 중**: 87995/87996이 완료되어야 mini50이 시작됨. 건드리지 말 것.

---

## 7. 참고 파일

| 파일 | 내용 |
|---|---|
| `prompts/MANIFEST_AUDIT_2026-06-24.md` | 감사 결과 (executive summary + 상세 표) |
| `prompts/HANDOFF_2026-06-24_quarter_eval.md` | 이전 핸드오프 (q4 기준) |
| `curriculum/STATUS_OVERFITTING_AND_PLAN_2026-06-23.md` | overfitting 분석 + 계획 |
| `stages_tiered_20260622/REPORT_stagebuild_2026-06-22.md` | 스테이지 설계 문서 |
| `paper/25580_Neuron_Aware_Data_Select.pdf` | OPSD 논문 PDF |
| `OPSD_original/scripts/run_opsd_8b_nonthink.sh` | OPSD 원본 8B nonthink 학습 스크립트 |
| `OPSD_original/scripts/run_opsd_8b.sh` | OPSD 원본 8B thinking 학습 스크립트 |

## 8. 환경
- conda env: `/scratch/lami2026/personal/jimin_2782/envs/opsd`
- 활성화: `source /scratch/lami2026/personal/jimin_2782/miniforge3/etc/profile.d/conda.sh && conda activate /scratch/lami2026/personal/jimin_2782/envs/opsd`
- PYTHONPATH: `src/OPSD_Curriculum/training/opsd_src:src/OPSD_Curriculum/training/curriculum`