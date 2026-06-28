# Hand-off — 클러스터 분할 완료 → 커리큘럼 러닝(OPSD) 본격 시작 직전

작성: 2026-06-18 / 트랙: reasoning_pivot, pooled(pilot1+pilot2) THINKING ΔA
선행 핸드오프: `HANDOFF_CLUSTERING_2026-06-17.md`(직전 정본·승계), `MASTER_REPORT_phase3_2026-06-12.md`(group-structure 정본), `REPORT_curriculum_materials_DETAILED_2026-06-13.md`, `REPORT_level_subject_similarity_pooled_N3025_2026-06-17.md`

---

## 0. 한 줄 요약 + 다음 미션
- **다음 미션: 확정된 subject-cluster × difficulty 분할(16-stage)을 바탕으로 OPSD self-distillation 위에 커리큘럼 러닝을 본격 적용한다.**
- **커리큘럼 적용의 구체 디렉션(어떤 arm을 main으로, stage budget/step, baseline 비교 설계, 데이터 스케줄링 방식, eval 프로토콜 등)은 사용자가 다음 세션에 직접 스펙을 준다.** 이 핸드오프는 분석 결론·재료·서버 주의점·학습 베이스 코드만 정리한다.
- subject·level은 각각 별도 축(marginal). **level(difficulty)이 1차 축, subject는 cluster mixing 축.**
- ⚠ 학습 베이스: **`src/OPSD_original/`** (On-Policy Self-Distillation, TRL GOLD Trainer). `src/4.6_Task2/`의 verl/GRPO/fastcurl과는 **무관** — 혼동 금지.

---

## A. 운영 환경 (엄수)
- `lami2026`은 **공유 user 계정**. 모든 작업물은 **`/scratch/lami2026/personal/jimin_2782/` 내부에서만** 생성/수정. 공유 `~/.bashrc`, `~/.cache/huggingface/`, 타 personal 디렉토리는 **읽기만**.
- **★ 노드 규칙(엄수): `iREMB-C-03`·`iREMB-C-07` 두 노드만 사용 가능. 나머지(`C-02 / C-04 / C-05 / C-06` 등) 전부 금지.**
  - GPU/학습 필요 시: **03 또는 07만**, `sbatch`만(직접 srun 점유 X), smoke→pilot→본런 순서, `squeue -w iREMB-C-07`(또는 `-w iREMB-C-03`)로 점유 확인. `--exclusive` 금지.
- one-strike 금지: `sudo`, `chmod 777`, 공유 dotfile/cache 수정, 시스템 python `pip install`.
- 환경 변수(분석용 — CPU):
  ```bash
  PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
  export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
  export HF_HUB_CACHE=$HF_HOME/hub
  ```
- OPSD 학습용 env는 다음 세션 첫 점검 항목(§F 참조).
- 새 산출물은 **새 파일명(spec/date 박기)**, 기존 산출물 덮어쓰기 금지.

---

## B. 확정 데이터 사실 (canonical)
| | raw .pt | non-finite | finite N |
|---|---|---|---|
| pilot1 | 1608 | 0 | 1608 |
| pilot2 | 1417 | 0 | 1417 |
| **pooled** | **3025** | **0** | **3025** |

- subject = **8-canonical**, level = **1–8** (1=쉬움 ↔ 8=어려움). `problem_id = sha1(text)[:16]`.
- **L8 단독결론 금지**(n=66, 전부 pilot1, 5/8 subject만). **Prealgebra L5+ 없음**(정의상). subject×level 빈 셀 자연 발생.
- length confound = **보류**(현 의사결정 미반영). 난이도 축은 정당한 신호로 사용. (3-method 검증: gen_len-balanced 매칭해도 level gap 88% 유지, 유의)

---

## C. ★직전 세션 산출(클러스터 분할 — FINAL): EXCLUDE 'Other' arm ③-main
스크립트: `analysis/clustering/subject_grouping_excludeOther_pooled.py` (CPU, seed=42, ~115s)
산출물(모두 신규, 보존, `analysis/clustering/` 아래):
- `REPORT_subject_grouping_excludeOther_N3025.md`
- **`stages_arm3_excludeOther.json`** ← 커리큘럼 stage 정의의 1차 입력
- `grouping_excludeOther_outputs.json`
- `dendro_subjgroup_excludeOther_{average,complete}.png`, `heatmap_subjS7_reordered_excludeOther.png`

**확정 결과:**
- 'Other' 제외한 7 subjects → **K=4 subject cluster** (average·complete linkage 모두 target 일치; cophenetic avg +0.720 / comp +0.697):
  - **C1{Intermediate Algebra, Precalculus}**
  - **C2{Counting & Probability, Number Theory}**
  - **C3{Algebra, Prealgebra}**
  - **C4{Geometry}**
- 난이도축 고정: **D1{1,2}, D2{3,4}, D3{5,6}, D4{7,8}**
- σ(클러스터 smooth 순서) = **C1→C4→C2→C3** (open path cost 3.139, headroom_mean=0.165 — Geometry 직교성으로 modest).
- **16-stage snake** (4 difficulty × 4 cluster, D-경계서 σ reversal, subject 연속성 OK) — `stages_arm3_excludeOther.json`에 확정.
- harness 디자인: **arm④** = 동일 16셀 subject-cluster 방문순서 셔플, **arm⑤** = difficulty별 4 subject-agnostic random parts(=baseline).
- Feasibility(Set-A, non-Other): 총 **28,771행** / N(Other)=663, 4×4 cell min=79, **empty=0**.
- **CONFOUND FLAG(게이트 아님, 문서화)**: C3{Algebra,Prealgebra}는 D1 Prealgebra-dominant → D4 Algebra-only(Prealgebra L4 초과 공백). cluster 구성이 난이도 따라 이동(D1 3703→D2 4084→D3 1860→D4 79). 커리큘럼 stage 설계 시 인지.
- Robustness(게이트 아님): pilot1/pilot2 모두 target 일치(S-corr +0.981/+0.977). gen_len-balanced(N=1260)는 {Algebra,Prealgebra} 분리로 NO — Prealgebra length/difficulty confound로 주석 처리.

---

## D. 분석 결론 요약 (커리큘럼 정당화)
- **LEVEL = 1차 축(강함)**: within−between gap +0.434, ρ(level)+0.84~0.90, out-of-sample ρ +0.937, Δlevel 단조성 ρ +0.893.
- **SUBJECT = marginal 신호 있으나 level 통제 시 약함**(marginal +0.353 → within-level −0.04). 'Other'만 독립 클러스터(그래서 main arm은 Other 제외).
- 비대칭: 같은 과목·다른 레벨은 많이 갈림(+0.227), 같은 레벨·다른 과목은 거의 안 갈림(−0.04). → **난이도 주축, subject는 cluster mixing**.

---

## E. 데이터 / 로더 (활성화 분석용 — 그대로 재사용)
- 로더: `analysis/similarity_analysis.py`의 `sa.load_pilot(shifts_dir, max_n)` → `(DAF, DAT, md)`. DAT=THINKING ΔA(primary). md: `subject, level, unit, gen_len, is_correct`.
- shifts: pilot1 `reasoning_pivot/activation/outputs/pilot/shifts/*.pt`(1608), pilot2 `.../pilot2/shifts/*.pt`(1417).
- 커리큘럼 stage 입력: **`stages_arm3_excludeOther.json`**. 보조: `currmat_artifacts.npz`(ridge_level 점수·easy→hard 정렬).
- ※ subject×level 셀별 요약표(8×8 n/정답률/gen_len)가 필요하면 `analysis/curriculum_subjectlevel_cells.py`(아직 없음, md만 쓰면 .pt 로드 불필요해 빠름) 생성 권장.

---

## F. ★학습 베이스 코드: `src/OPSD_original/` (OPSD = On-Policy Self-Distillation)
커리큘럼 러닝은 **이 OPSD 코드 위에** 적용한다. `src/4.6_Task2/`(verl/GRPO/fastcurl)와 **무관**.

- **방법론**: OPSD = On-Policy Self-Distillation. 한 모델이 student(문제만)/teacher(정답 solution도) 두 역할을 하며, student의 on-policy trajectory에서 token-level JSD distribution matching. **TRL GOLD Trainer 기반**. arXiv 2601.18734.
- **핵심 파일** (`src/OPSD_original/`):
  - `opsd_trainer.py` — `OPSDTrainer` (core self-distillation trainer)
  - `data_collator.py` — self-distillation용 data collator
  - `opsd_train.py` — OPSD 학습 entry point
  - `sft_train.py` / `grpo_train.py` — SFT / GRPO baseline entry point
  - `accelerate.yaml` — multi-GPU accelerate config
  - `scripts/run_opsd_{1b,4b,8b}.sh`, `*_nonthink.sh`, `run_sft.sh`, `run_grpo.sh` — 런처
  - `eval/evaluate_math.py`, `eval/run_eval.sh`, `eval/run_eval_nonthink.sh` — vLLM eval (AIME24/25, HMMT25 등)
- **환경**: `environment.yml` → conda env `opsd` + `flash-attn==2.8.3`. (기존 `envs/verl_new`로 될지 / 별도 opsd env가 필요할지 다음 세션 첫 점검. 새 env 설치 시 personal 디렉토리 내, 시스템 pip 금지.)
- **모델**: Qwen3-1.7B(quick start, 4×H100 ~15min/100step), 4B, 8B. thinking / non-thinking 모드 둘 다 지원.
- **주요 OPSD 인자**: `--fixed_teacher`(LoRA로 teacher step0 고정, 메인 결과), `--use_tinker_loss`(sampled-token PG, 메모리절약), `--max_completion_length`(메인=1024), `--beta`(JSD; 0=forward KL, 1=reverse KL), `--jsd_token_clip`(per-token KL clip, default 0.05 — style token 'wait'/'think' dominance 완화), `--reason_first`, `--run_config`(출력/WandB suffix).
- ⚠ **커리큘럼 주입 방식 미정**: OPSD는 단일 데이터셋 self-distillation 구조. subject-cluster × difficulty 16-stage를 어떻게 주입할지(stage별 데이터셋 분할 후 순차 학습 / 단일 run 내 data scheduling / 기타)는 **사용자 디렉션 대기**.
- ⚠ **데이터 정합성 점검 필요**: OPSD 학습 데이터(문제 + ground-truth solution)와 우리 분석의 `problem_id`(sha1) · `stages_arm3_excludeOther.json` sample 식별자 간 **조인키 매핑이 존재하는지** 다음 세션 첫 점검 항목 — `opsd_train.py` / `scripts/run_opsd_1b.sh`에서 데이터 로딩 방식 먼저 확인.
- ⚠ **`src/OPSD_original/`는 upstream 원본 → 직접 수정 금지.** 커리큘럼용 변경은 **새 디렉토리**(예: `src/OPSD_Curriculum/training/` 또는 spec/date 박은 경로)로 fork하여 생성.

---

## G. 미해결 / 주의
1. length confound = 보류(현 의사결정 미반영). 재검은 옵션.
2. C3 Prealgebra-difficulty confound(§C) — stage 설계 시 인지.
3. L8 단독결론 금지 / Prealgebra L5+ 없음.
4. with-Other K=5 변형은 직전 세션에 재생성 안 함(main = exclude-Other K=4).
5. **커리큘럼 러닝 디렉션(커리큘럼 주입 방식·arm 선택·budget·baseline·eval) 미정 — 사용자 스펙 대기.**
6. **OPSD 학습 데이터 ↔ problem_id / stage sample 식별자 조인키 정합성 미확인** — 학습 데이터 빌드 전 최우선 점검 (`opsd_train.py` 데이터 로딩 방식 확인).

---

## H. 새 세션 첫 행동
1. 이 파일 + `REPORT_subject_grouping_excludeOther_N3025.md` + `stages_arm3_excludeOther.json` 통독.
2. **`src/OPSD_original/` 통독**: `opsd_train.py`(data loading 방식), `opsd_trainer.py`, `data_collator.py`, `scripts/run_opsd_1b.sh`, `eval/evaluate_math.py`.
3. **§G.6 데이터 정합성 점검**: OPSD 학습 데이터 포맷 및 sample 식별자 ↔ `stages_arm3_excludeOther.json` sample 식별자 매핑 존재 여부.
4. **OPSD env 점검**: `environment.yml` 기준 conda env `opsd` 가 이미 있는지(`conda env list`), 없으면 personal 디렉토리 내 생성 계획.
5. **사용자로부터 커리큘럼 러닝 디렉션 수령.**
6. 디렉션에 맞춰 구현. 신규 산출물 = **새 파일명/새 디렉토리(spec/date)**, GPU/학습은 **03/07 노드 + sbatch + smoke→pilot→본런**.
