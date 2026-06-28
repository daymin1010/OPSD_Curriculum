# Hand-off — Curriculum 클러스터 분할 직전 (subject·level marginal 트랙)

작성: 2026-06-17 / 트랙: reasoning_pivot, pooled(pilot1+pilot2) THINKING ΔA
선행 핸드오프: `HANDOFF_CURRICULUM_SUBJECTxLEVEL_2026-06-13.md`(직전 정본·승계), `HANDOFF_PHASE1_POOLED.md`(Phase0/1 원본), `MASTER_REPORT_phase3_2026-06-12.md`(group-structure 정본), `REPORT_curriculum_materials_DETAILED_2026-06-13.md`(커리큘럼 재료 해석)

---

## 0. 한 줄 요약 + 이번 세션 합의사항
- **다음 미션: pooled N=3025 분석 결과를 바탕으로 커리큘럼을 만들기 위해 "클러스터를 나누는" 작업을 한다.**
- **클러스터 분할의 구체 디렉션(클러스터 개수 / 분할 축(level 주축·subject 보조) / 경계 기준 / sparse 셀 처리 등)은 사용자가 다음 세션에 직접 스펙을 준다.** 이 핸드오프는 그 작업에 필요한 컨텍스트·서버 주의점·데이터/로더 컨텍스트·재료 인덱스만 정리한다.
- 여전히 **subject·level을 각각 별도 축(marginal)으로** 본다. joint unit(subject×level 57셀)은 1차 설계축으로 쓰지 않는다(보조 참고만).
- **length confound: 판단 보류(현재 신경쓰지 않음).** 어려운 문제일수록 추론·서술이 길어지는 건 자명하므로 활성화 난이도 신호가 "length 때문에 나온 가짜"라고 볼 수 없다. (이번 세션 N=3025 검증으로 더 강화됨 → §C.4 참조). `REPORT_unsup_difficulty_lengthgate.md`의 GATE=FAIL/"GPT level fallback" 권고는 현 단계 의사결정에 미반영(보류, 단정 철회 아님). 지금은 활성화 난이도 축을 정당한 난이도 신호로 사용한다.

---

## A. 운영 환경 (원본: `src/OPSD_Curriculum/HANDOFF_2026-05-30_activation_extraction.md` — 읽기전용)
- `lami2026`은 **공유 user 계정**. 모든 작업물은 **`/scratch/lami2026/personal/jimin_2782/` 내부에서만** 생성/수정. 공유 `~/.bashrc`, `~/.cache/huggingface/`, 타 personal 디렉토리는 **읽기만**.
- **이 트랙은 CPU only (GPU 불필요).**
- **★ 노드 사용 규칙(엄수): `iREMB-C-03` 과 `iREMB-C-07` 두 노드만 사용 가능. 나머지 노드(`iREMB-C-02 / C-04 / C-05 / C-06` 등)는 전부 사용 금지.**
  - 만약 추후 GPU/sbatch가 필요해지면: **03 또는 07 노드만**, `sbatch`만 사용(직접 srun 점유 X), smoke→본런 순서, `squeue -w iREMB-C-07`(또는 `-w iREMB-C-03`)로 점유 확인. `--exclusive` 금지.
- one-strike 금지: `sudo`, `chmod 777`, 공유 dotfile/cache 수정, 시스템 python `pip install`(반드시 `envs/verl_new/bin/pip`).
- 환경 변수:
  ```bash
  PY=/scratch/lami2026/personal/jimin_2782/envs/verl_new/bin/python
  export HF_HOME=/scratch/lami2026/personal/jimin_2782/cache/huggingface
  export HF_HUB_CACHE=$HF_HOME/hub
  ```
- 새 산출물은 **새 파일명(spec/date 박기)**, 기존 산출물 덮어쓰기 금지.

---

## B. 확정된 데이터 사실 (canonical)
| | raw .pt | non-finite | finite N |
|---|---|---|---|
| pilot1 | 1608 | 0 | 1608 |
| pilot2 | 1417 | 0 | 1417 |
| **pooled** | **3025** | **0** | **3025** |

- subject = **8-canonical**, level = **1–8** (1=쉬움 ↔ 8=어려움).
- is_correct non-null = **3024** (pilot1 1607 + pilot2 1417), 전체 1-shot 정답률 = **0.818**.
- `problem_id = sha1(text)[:16]`.
- **level 카운트(pooled)**: L1 335, L2 480, L3 480, L4 437, L5 420, L6 420, L7 387, L8 66.
- **L8 caveat (필독)**: L8=66 **전부 pilot1**(pilot2 L8 없음). 5/8 subject만: Number Theory 23 / Other 13 / Intermediate Algebra 11 / Counting&Probability 10 / Geometry 9. **Algebra·Prealgebra·Precalculus = L8 0**(정상 quirk). → **L8 단독 결론 금지**(n 부족 + subject 불균형). level 보고 시 L1–L8과 L1–L7 둘 다 권장.
- Prealgebra: L4=17, L5+ = 0 (정의상 정상).
- canonical N 표기 규칙: **raw 3025 / finite 3025 / analysis-N(필터별)** 병기. "3000"은 별칭.

---

## C. 커리큘럼 정당화 근거 (기존 리포트 수치 인용)
**출처: `MASTER_REPORT_phase3_2026-06-12.md`, `REPORT_curriculum_materials_DETAILED_2026-06-13.md`, `REPORT_level_subject_similarity_pooled_N3025_2026-06-17.md`**

### C.1 LEVEL 축 (강함 — 커리큘럼의 1차 축)
- THINKING level gap (within−between) = **+0.434**, perm-p ≤ 0.001. ordinality ρ(level) = **+0.84~0.90**.
- 과목을 고정한 within-subject / between-LEVEL gap = **+0.227** (p=0.005) → 같은 과목 안에서도 난이도로 뚜렷이 갈림.
- Δlevel 단조성: same-subject·diff-level 코사인이 Δ=1 +0.734 → Δ=3 +0.019 → Δ=5 −0.467, **ordinality ρ(cos,−Δ)=+0.893** → 활성화 공간에 연속적·단조적 난이도 축 실재.
- 난이도 축(ridge_level, pilot1 fit → pilot2 test) out-of-sample ρ(level)=**+0.937**.

### C.2 SUBJECT 축 (marginal로는 신호 있으나 level 통제 시 약함)
- subject marginal gap = **+0.353**, perm-p = 0.001 → 과목 자체로는 분리 신호 있음.
- 난이도를 고정하면(within-level / between-SUBJECT) gap = **−0.0418** (p=0.005) → 거의 0. 같은 난이도 안에서는 과목으로 거의 안 갈림.
- 예외: **'Other' 과목만** 두 레이어 뷰에서 독립 클러스터 형성(subject-entropy 0) → 난이도와 무관한 고유 활성화 패턴.
- 지도 subject LDA(pilot1→pilot2): macro-F1 = 0.681 (chance≈0.125) — 정보는 있으나 representation 약함.

### C.3 비대칭 핵심 (직관)
> "같은 과목·다른 레벨"은 activation이 **많이** 달라지고(+0.227), "같은 레벨·다른 과목"은 **거의 안** 달라진다(−0.04).
> ⇒ 활성화 1차 결정자는 **LEVEL**. SUBJECT 기여는 약하고 'Other'에 국한.
> **클러스터 함의**: 난이도(level)를 주 분할 축으로, subject는 클러스터 내부의 mixing/balancing 축으로 두는 것이 데이터와 정합. (단 최종 분할 디렉션은 사용자 스펙 우선.)

### C.4 ★이번 세션 신규: Length-confound 3-method 검증 (POOLED N=3025)
- 스크립트 `levsubj_length_confound_pooled.py` → `LENGTHCONF_pooled3025.txt`, `lengthconf_pooled_outputs.json`. μ_pooled(per-layer) centering, seed=42, CPU only, ρ(level,gen_len)=+0.709.
- 세 방법(중복 아님): ① Mantel(거리행렬 상관+perm p) ② residual survival(gen_len±log/±quad/+level 회귀 잔차에서 centroid-cosine 생존율) ③ gen_len-balanced(5분위 매칭 부분표본 gap 재현).
- **SUBJECT → content-driven (PASS)**: Mantel r≈+0.03 (p=0.62~0.82, 비유의); residual min +0.958(layeravg)/+0.974(mid); balanced gap +0.353→+0.201 (유지 57%, p=0.005, N_bal=1200).
- **LEVEL → Mantel/residual은 length 동행이나 balanced 유지**: Mantel r≈+0.86 (p≈0.0001); residual min +0.624/+0.668(부분붕괴); **balanced gap +0.435→+0.384 (유지 88%, p=0.005, N_bal=142)**.
- **결론**: 난이도는 길이와 강하게 동행하지만 길이를 매칭해도 gap이 88% 유지(유의) → length만으로 설명 불가. 어려운 문제일수록 추론이 길어지는 건 자명하므로 **활성화 난이도 축을 정당한 신호로 계속 사용**(length confound 보류 유지).

---

## D. 데이터 / 로더 사실 (다음 세션이 그대로 쓸 것)
- 로더: `analysis/similarity_analysis.py`의 `sa.load_pilot(shifts_dir, max_n)` → `(DAF, DAT, md)`.
  - `DAF` = FAITHFUL ΔA, `DAT` = THINKING ΔA(primary). `md` 컬럼: `subject, level, unit` (+ `gen_len`, `is_correct` 등).
  - content filter 없음. 로드 시간 pilot1+pilot2 합쳐 ~3분(.pt I/O).
- shifts 경로:
  - pilot1: `reasoning_pivot/activation/outputs/pilot/shifts/*.pt` (1608)
  - pilot2: `reasoning_pivot/activation/outputs/pilot2/shifts/*.pt` (1417)
- 재사용 메트릭 함수(검증됨): `sa.normalize_members, sa.centroids, sa.sim_matrix, sa.within_between, sa.perm_pvalue, sa.spearman, sa.MIN_N, sa.N_PERM`.
- 레이어 뷰: **layeravg**(feat dim 12288, silhouette가 더 좋음) / **midL11-15**(feat dim 61440). 둘 다 동일한 난이도 정렬을 줌(robustness).
- 난이도 점수축: **ridge_level**(채택). pooled centering은 per-layer μ_pooled.

---

## E. 다음 세션이 바로 쓸 재료 / 산출물 인덱스
**클러스터 작업의 핵심 입력:**
- `currmat_artifacts.npz` — 라벨/클러스터 배정/ridge_level 점수/cluster easy→hard 정렬 (45KB). **← 클러스터 분할의 1차 입력**
- 덴드로그램: `dendro_layeravg_currmat.png`, `dendro_midL11-15_currmat.png` (계층 구조 시각).

**기존 아티팩트 (모두 보존, 덮어쓰기 금지):**
- `unsupdiff_artifacts.npz` — 비지도 PCA 축·잔차화 결과 (length 보류 트랙).
- `REPORT_pooled_3025.md`, `pooled_analysis.py` — Phase1 canonical group-structure.
- `compare_pilot1_pilot2.py` → `REPORT_pilot2_comparison.md` — replication.

**★이번 세션 신규 산출물 (보존):**
- `levsubj_length_confound_pooled.py` (3-method length-confound 스크립트, CPU)
- `LENGTHCONF_pooled3025.txt`, `lengthconf_pooled_outputs.json` (N=3025 결과)
- `REPORT_level_subject_similarity_pooled_N3025_2026-06-17.md` → "Length-confound robustness — 3-method 일치 (POOLED N=3025)" 섹션 추가됨.
- `level_subject_similarity_pooled.py`, `subject_similarity_gate.py`, `subject_length_confound.py` (관련 게이트/유사도 스크립트)

**아직 없는 것 (필요 시 생성 권장):**
- **subject×level 셀별 요약표(8×8)**: 셀별 n, 1-shot 정답률, 평균 gen_len, (선택) ridge_level 평균점수. 신규 스크립트 후보: `analysis/curriculum_subjectlevel_cells.py` (CPU only, md만 쓰면 .pt 로드 불필요해 빠름). sparse 셀(예: Geometry|L8 n=9) 판단에 유용.

---

## F. 미해결 / 주의
1. **length confound = 보류**(§0, §C.4). 현 의사결정에 미반영. 재검은 옵션.
2. **L8 단독결론 금지**(n=66, 전부 pilot1, 5/8 subject만). level 보고는 L1–L8 / L1–L7 병기.
3. **Prealgebra L5+ 없음**(정의상). subject×level 빈 셀 자연 발생.
4. **sparse 셀 / 작은 클러스터 처리 방침 미정** — 사용자 클러스터 분할 스펙 대기(흡수/제외/별도 처리 중 선택).
5. **unit(57셀) 비사용** — joint 분석은 이미 `curriculum_materials.py`에 있음. 이번 트랙은 marginal(subject, level) 우선.

---

## G. 새 세션 첫 행동 (클러스터 분할)
1. 이 파일 + `MASTER_REPORT_phase3_2026-06-12.md` + `REPORT_curriculum_materials_DETAILED_2026-06-13.md` + `REPORT_level_subject_similarity_pooled_N3025_2026-06-17.md` 통독.
2. `currmat_artifacts.npz`(클러스터 배정·ridge_level·easy→hard 정렬) + 덴드로그램 PNG 로드/확인.
3. **사용자로부터 클러스터 분할 디렉션 수령** (클러스터 개수 / 분할 축(level 주축·subject 보조) / 경계 기준 / sparse 셀 처리 등).
4. 디렉션에 맞춰 클러스터 분할 구현. 신규 스크립트/산출물은 **새 파일명(spec/date)**, CPU only, 노드 필요 시 **03/07만**.
