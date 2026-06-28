# Phase 3 — MASTER 통합 리포트 (reasoning_pivot pooled ΔA group-structure)

작성: 2026-06-12 / 트랙: reasoning_pivot thinking·faithful ΔA, pilot1+pilot2

> **목적.** "ΔA(활성화 시프트)가 subject/level/unit 구조를 담는가?"에 대한 우리(=OPSD)의 진단.
> group-centroid · layer-averaged cosine + 라벨-퍼뮤테이션 검정. NAIT 논문의 PCA-direction
> scoring과는 다른 진단(그건 이후 curriculum-direction 단계, Track C supervised).

---

## 0. 한 줄 결론
세 그룹(subject/level/unit) 모두 within > between 이 **통계적으로 유의**(perm-p ≤ 0.005)하게 성립.
ΔA는 난이도(level)·주제(subject)·세부 unit 구조를 실제로 담고 있으며, **THINKING 모드가
FAITHFUL보다 일관되게 더 잘 분리**하고 **level은 강한 순서성(ρ≈0.84–0.90)** 을 보인다.

## 1. Canonical N (항상 raw / finite / analysis-N 병기)
- raw .pt (pilot1+pilot2) = **3025**  (pilot1=1608, pilot2=1417)
- non-finite ΔA dropped = **0**
- **finite pooled N = 3025** (canonical; "3000"은 별칭)
- analysis-N은 필터별: level L1–L8 = 3025, level L1–L7(L8 drop) = 2959, unit(n≥MIN_N) = 3025.
- 출처: `analysis/N_AUDIT.md`(Phase0 정본)와 완전 일치.

### L8 caveat (필독)
- L8 total = **66, 전부 pilot1** (pilot2에는 L8 없음).
- L8은 5/8 subject만: Number Theory 23 / Other 13 / Intermediate Algebra 11 / Counting&Probability 10 / Geometry 9.
- Algebra·Prealgebra·Precalculus = L8 0 (handoff quirk #4).
- ⇒ **L8 단독 결론 금지**(n 부족 + subject 불균형). level은 L1–L8과 L1–L7 둘 다 보고해 L8 영향 분리.

---

## 2. Phase 1 — POOLED (메인, canonical)
설계: pilot1 ⊕ pilot2 = 3025 를 **단일 pooled 글로벌 평균(μ_pooled, per layer)** 으로 centering.
퍼뮤테이션은 pooled 전체에서 라벨 셔플. (per-pilot self-center + within-pilot perm 은 Phase 2 replication.)
N_PERM: subject 1000 / level 1000 / unit 200. metric = `similarity_analysis.py` 함수(동일 방법).
산출: `analysis/REPORT_pooled_3025.md` (canonical, 본런 36분, BLAS-vectorized).

within / between / gap(=within−between) / perm-p / level ordinality ρ:

| MODE | grouping | G | analysis-N | within | between | **gap** | perm-p | level ρ |
|---|---|---|---|---|---|---|---|---|
| THINKING | subject | 8 | 3025 | +0.232 | −0.121 | **+0.353** | 0.0010 | — |
| THINKING | level (L1–L8) | 8 | 3025 | +0.324 | −0.110 | **+0.434** | 0.0010 | +0.841 |
| THINKING | level (L1–L7) | 7 | 2959 | +0.312 | −0.121 | **+0.434** | 0.0010 | +0.896 |
| THINKING | unit (subj×level, n≥MIN_N) | 57 | 3025 | +0.430 | −0.010 | **+0.440** | 0.0050 | — |
| FAITHFUL | subject | 8 | 3025 | +0.141 | −0.121 | **+0.263** | 0.0010 | — |
| FAITHFUL | level (L1–L8) | 8 | 3025 | +0.282 | −0.125 | **+0.407** | 0.0010 | +0.826 |
| FAITHFUL | level (L1–L7) | 7 | 2959 | +0.301 | −0.130 | **+0.430** | 0.0010 | +0.859 |
| FAITHFUL | unit (subj×level, n≥MIN_N) | 57 | 3025 | +0.359 | +0.003 | **+0.356** | 0.0050 | — |

읽을거리:
1. **세 그룹 모두 perm-p ≤ 0.005** → 구조는 우연 아님.
2. **THINKING > FAITHFUL** (subject gap +0.353 vs +0.263 등 전 그룹). thinking-mode 재추출 가설 지지.
3. **level 순서성 ρ = +0.83~+0.90** (매우 강함) → 난이도가 연속 축으로 임베딩됨(커리큘럼 설계에 유리).
4. **L8 robustness**: L8 포함/제외 비교 시 THINKING level gap 동일(+0.434), FAITHFUL은 포함 시 약간 낮음(+0.407 vs +0.430). 결론은 L8에 의존하지 않음.
5. unit은 between≈0 이지만 within이 높아(+0.36~0.43) gap 최대 — 세부 unit 단위로도 분리됨.

---

## 3. Phase 2 — REPLICATION (부록, robustness)
설계(Phase1과 반대 handling): 각 pilot을 **개별 self-center**, **L1–L7 공통 범위**(pilot2 L8 없음),
within-pilot 라벨 퍼뮤테이션. 두 pilot에서 동일 방향(THINKING>FAITHFUL, level ordinality>0)이
재현되는지 확인하는 견고성 점검.
스크립트: `analysis/compare_pilot1_pilot2.py` (CPU only) → 산출 `analysis/REPORT_pilot2_comparison.md`.
N_PERM: subject 1000 / level 1000 / unit 200.

> 상태: 본 세션에서 full 실행이 진행 중(백그라운드, CPU). 완료 시 `REPORT_pilot2_comparison.md`가
> 생성되며, 위 Phase1의 정성적 패턴(세 그룹 유의 분리, THINKING>FAITHFUL, level 양의 ρ)이
> 양 pilot에서 각각 재현되는지 대조하면 된다. (방향 불일치가 나오면 그때 재검토.)

---

## 4. 산출물 인덱스 (모두 보존 — 덮어쓰기 금지)
- Phase0: `analysis/N_AUDIT.md`, `analysis/audit_pooled_N.py`
- Phase1(canonical): `analysis/REPORT_pooled_3025.md`, `analysis/pooled_analysis.py`
- Phase2(replication): `analysis/compare_pilot1_pilot2.py` → `analysis/REPORT_pilot2_comparison.md`
- 이전 단일 pilot 분석: `REPORT_similarity_pilot.md`, `REPORT_similarity_pilot2.md`, `REPORT_residualize_pilot.md`
- ΔA 원천(.pt): `reasoning_pivot/activation/outputs/pilot/shifts/*` (1608), `outputs/pilot2/shifts/*` (1417)

## 5. 다음 단계 (별도 트랙)
이 group-structure 신호가 확인됐으므로, 다음은 NAIT-inspired **curriculum-direction 추출**
(Track C supervised direction)으로 진행. 추출에 GPU가 필요하면 `iREMB-C-07`(L40s), sbatch만,
smoke→본런, `squeue -w iREMB-C-07` 점유 확인 절차 준수.
</content>
<task_progress">- [x] Phase 1 pooled 본런 완료 + 검증
- [x] Phase 2 replication full 실행(백그라운드 진행 중) 확인
- [x] Phase 3 최종 통합 리포트 작성 (MASTER_REPORT_phase3_2026-06-12.md)</task_progress>
