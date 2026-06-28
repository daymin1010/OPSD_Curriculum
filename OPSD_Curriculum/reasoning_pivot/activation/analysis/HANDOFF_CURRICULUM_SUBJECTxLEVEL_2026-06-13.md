# Hand-off — Curriculum 본격 설계 직전 (subject·level marginal 트랙)

작성: 2026-06-13 / 트랙: reasoning_pivot, pooled(pilot1+pilot2) THINKING ΔA
선행 핸드오프: `HANDOFF_PHASE1_POOLED.md`(Phase0/1 원본), `MASTER_REPORT_phase3_2026-06-12.md`(group-structure 정본), `REPORT_curriculum_materials_DETAILED_2026-06-13.md`(커리큘럼 재료 해석)

---

## 0. 한 줄 요약 + 이번 세션 합의사항
- **다음 미션: subject·level을 각각 별도 축(marginal)으로 보고 커리큘럼을 본격 설계한다. joint unit(subject×level 57셀)은 1차 설계축으로 쓰지 않는다**(필요 시 보조 참고만).
- **커리큘럼 구체 설계(stage 개수/경계/schedule/혼합비 등)는 사용자가 다음 세션에 직접 스펙을 준다.** 이 핸드오프는 그 설계에 필요한 데이터·근거·로더 컨텍스트만 정리한다.
- **length confound: 판단 보류(현재 신경쓰지 않음).**
  - 근거: 어려운 문제일수록 추론·서술이 길어지는 건 자명하므로, 활성화 난이도 신호가 "length 때문에 나온 가짜"라고 볼 수 없다.
  - 따라서 `REPORT_unsup_difficulty_lengthgate.md`의 GATE=FAIL / "GPT level fallback" 권고는 **현 단계 의사결정에 반영하지 않는다(보류)**. 단정적으로 철회한 것은 아니며, 추후 필요 시에만 재검 옵션으로 남겨둔다. 지금은 활성화 난이도 축을 정당한 난이도 신호로 사용한다.

---

## A. 운영 환경 (원본: `src/OPSD_Curriculum/HANDOFF_2026-05-30_activation_extraction.md` — 읽기전용)
- `lami2026`은 **공유 user 계정**. 모든 작업물은 **`/scratch/lami2026/personal/jimin_2782/` 내부에서만** 생성/수정. 공유 `~/.bashrc`, `~/.cache/huggingface/`, 타 personal 디렉토리는 **읽기만**.
- **이 트랙은 CPU only (GPU 불필요).** 만약 추후 GPU가 필요해지면: 추론은 `iREMB-C-07`(L40s 48GB×4), **sbatch만**, smoke→본런, `squeue -w iREMB-C-07` 점유 확인. `iREMB-C-02/04/05/06` 사용 금지, `--exclusive` 금지.
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

---

## C. 커리큘럼 정당화 근거 (기존 리포트 수치 인용)
**출처: `MASTER_REPORT_phase3_2026-06-12.md`, `REPORT_curriculum_materials_DETAILED_2026-06-13.md`**

### C.1 LEVEL 축 (강함 — 커리큘럼의 1차 축)
- THINKING level gap (within−between) = **+0.434**, perm-p ≤ 0.001. ordinality ρ(level) = **+0.84~0.90**.
- 난이도를 고정하지 않은 marginal level 분리 외에도, **과목을 고정한** within-subject / between-LEVEL gap = **+0.227** (p=0.005) → 같은 과목 안에서도 난이도로 뚜렷이 갈림.
- Δlevel 단조성: same-subject·diff-level 코사인이 Δ=1 +0.734 → Δ=3 +0.019 → Δ=5 −0.467, **ordinality ρ(cos,−Δ)=+0.893** → 활성화 공간에 연속적·단조적 난이도 축 실재.
- 난이도 축(ridge_level, pilot1 fit → pilot2 test) out-of-sample ρ(level)=**+0.937**. (gen_len 동행 +0.745는 §0대로 **보류**.)

### C.2 SUBJECT 축 (marginal로는 신호 있으나 level 통제 시 약함)
- subject marginal gap = **+0.353**, perm-p = 0.001 → 과목 자체로는 분리 신호 있음.
- 그러나 **난이도를 고정하면**(within-level / between-SUBJECT) gap = **−0.0418** (p=0.005) → 거의 0. 즉 **같은 난이도 안에서는 과목으로 거의 안 갈림.**
- 예외: **'Other' 과목만** 두 레이어 뷰에서 독립 클러스터를 형성(layeravg C7 / mid C3, subject-entropy 0) → 'Other'는 난이도와 무관한 고유 활성화 패턴.
- 지도 subject LDA(pilot1→pilot2): macro-F1 = 0.681 (chance≈0.125) — 정보는 있으나 representation 약함(대조용).

### C.3 비대칭 핵심 (직관)
> "같은 과목·다른 레벨"은 activation이 **많이** 달라지고(+0.227), "같은 레벨·다른 과목"은 **거의 안** 달라진다(−0.04).
> ⇒ 활성화 1차 결정자는 **LEVEL**. SUBJECT 기여는 약하고 'Other'에 국한.
> **설계 함의**: 난이도(level)를 주 staging 축으로, subject는 각 stage 내부의 mixing/balancing 축으로 두는 것이 데이터와 정합. (단 최종 설계는 사용자 스펙 우선.)

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
**기존 아티팩트 (모두 보존, 덮어쓰기 금지):**
- `currmat_artifacts.npz` — 라벨/클러스터 배정/ridge_level 점수/cluster easy→hard 정렬 (45KB).
- `unsupdiff_artifacts.npz` — 비지도 PCA 축·잔차화 결과 (length 보류 트랙).
- `REPORT_pooled_3025.md`, `pooled_analysis.py` — Phase1 canonical group-structure.
- `compare_pilot1_pilot2.py` → `REPORT_pilot2_comparison.md` — replication.
- 덴드로그램: `dendro_layeravg_currmat.png`, `dendro_midL11-15_currmat.png`.

**아직 없는 것 → 다음 세션 첫 작업으로 생성 권장:**
- **subject×level 셀별 요약표(8×8)**: 셀별 표본수 n, 1-shot 정답률, 평균 gen_len, (선택) ridge_level 평균점수.
  - 신규 스크립트 후보: `analysis/curriculum_subjectlevel_cells.py` (CPU only, md만 쓰면 .pt 로드 불필요해 빠름).
  - 이 표가 있어야 stage별 표본 충분성·과목 balancing·sparse 셀 처리(예: Geometry|L8 n=9)를 사용자 설계 스펙에 맞춰 판단 가능.

---

## F. 미해결 / 주의
1. **length confound = 보류**(§0). 현 설계 의사결정에 미반영. 재검은 옵션.
2. **L8 단독결론 금지**(n=66, 전부 pilot1, 5/8 subject만). level 보고는 L1–L8 / L1–L7 병기.
3. **Prealgebra L5+ 없음**(정의상). subject×level 표에서 빈 셀 자연 발생.
4. **sparse 셀 처리 방침 미정** — 사용자 커리큘럼 스펙 대기(흡수/제외/별도 처리 중 선택).
5. **unit(57셀) 비사용** — joint 분석은 이미 `curriculum_materials.py`에 있음. 이번 트랙은 marginal(subject, level) 우선.
6. canonical N 표기 규칙: **raw 3025 / finite 3025 / analysis-N(필터별)** 병기. "3000"은 별칭.

---

## G. 새 세션 첫 행동
1. 이 파일 + `MASTER_REPORT_phase3_2026-06-12.md` + `REPORT_curriculum_materials_DETAILED_2026-06-13.md` 통독.
2. `analysis/curriculum_subjectlevel_cells.py` 작성 → subject×level 8×8 요약표 산출(CPU, md 기반).
3. **사용자로부터 커리큘럼 설계 스펙 수령** 후 구현(stage 정의·혼합비·schedule 등은 사용자 결정).
