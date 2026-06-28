# OPSD curriculum 재설계 — level-backbone + subject-residual slack (2026-06-24)

**method id**: `level_backbone_residual_subject_slack`  ·  **alpha = 2.0**  ·  **universe N = 28,771** (Set-A, OpenThoughts-Math-30K, include_other=False)

이 폴더는 기존 `stages_tiered_20260622/`(method `tiered_difficulty_backbone_residual_within_tier`, n_tiers=2)를 대체하는 새 "ours" 커리큘럼이다.

---

## 0. 한 줄 요약
난이도(level)를 단조·타이트한 backbone으로 고정하되, 같은 난이도대 안에서 **subject를 활성 기하(level-residual) 좌표 `g`로 재배치**한다(`score = level + α·g(subject)`, unit-atomic 분할). 난이도 회복(성능) + subject 2축 유지(차별성)를 동시에 달성.

---

## 1. 왜 다시 만들었나 (현 ours 실패 진단)
현 ours(`stages_cond3_ours_C2`)가 난이도-only baseline(`cond2_diff`)에 **짐**:
- full step900: aime24 avg@3 **55.6 vs 66.7**, math500 **79.6 vs 83.0** (diff 우위)
- MATH-500 subject/level 분해: ours가 **level 2–3에서 −8~−11%** 손해. level 5/Number Theory만 소폭 우위.
- **forgetting 아님** — 마지막 stage subject(Geometry)도 ours가 −2.4 손해.

**근본 원인 (코드 진단)**: `n_tiers=2` backbone.
- tier가 둘(L1–4 / L4–8)뿐 → tier 내 subject nearest-path가 난이도를 진동시킴.
- 5-stage equal-mass cut이 tier 경계를 가로질러 → ours stage2가 **level 1–8 광폭**(var 폭발), level 2–3이 stage0·1에 흩어져 **"level 2–3 마스터 stage"가 없음**.
- 게다가 인접 stage **전체표현 점프가 diff보다 큼**(0.394 vs 0.237, W_ALL) — 논문 주장("minimize representational jump")과 **정반대**.

진단 근거 수치(현 ours / diff):
| | 단조 min_diff | mean per-stage var | 전체표현 점프(W_ALL) |
|---|---|---|---|
| diff | +0.97 | 0.13 | 0.237 |
| **현 ours** | **−0.15** (dip) | **1.31** | **0.394** |

---

## 2. 설계
**score(problem) = level + α · g(subject)**, α = 2.0.

- **g(subject) ∈ [−0.5, +0.5]** = 난이도 직교 subject 기하축.
  - 출처: `stages_tiered_20260622/stagebuild_artifacts.npz :: residual_M_keep` (42×42 unit 코사인 유사도, **level group-mean 제거**).
  - 활성: **Qwen3-8B THINKING** (NAIT-thinking span), **faithful(DAF) 폐기**, pooled **pilot1+pilot2 N=3025**.
  - 계산: residual_M_keep classical-MDS leading axis(설명력 35.4%) → unit 좌표 → subject 평균 → C_disc>C_geo로 sign-orient → min-max [−0.5,0.5].
  - 값: `Precalculus −0.50, Inter.Algebra −0.45, Algebra −0.20, Geometry −0.12, Prealgebra +0.21, Number Theory +0.33, Counting&Prob +0.50`.
  - 의미: **g<0(C_alg) 이른 stage, g>0(C_disc) 늦은 stage, Geometry 중립.**
- **α=2.0**: subject가 ~1 level 분량만 이동 → backbone 단조·타이트 유지, level 밴드 내부만 subject로 재배치.
- **unit-atomic 분할**: unit(subject|level cell)을 score로 정렬 → unit을 쪼개지 않고 k·N/5에 가장 가까운 unit 경계에서 5분할. stage 크기는 자연스럽게 다름.

α는 프로토타입 sweep(α∈{0,0.5,1,1.5,2,3})에서 선택: α↑ = (점프↓, subject효과↑) vs (난이도밴드 넓어짐). α=2.0이 "점프<diff + 단조 + maxspan 3 + cond5 유의분리"를 만족하는 균형점.

---

## 3. 검증 결과 (build 출력 + measure_fulljump)
| 지표 | diff | 현 ours | **새 ours (α=2.0)** | 목표 |
|---|---|---|---|---|
| 단조 min_diff | +0.97 | −0.15 ✗ | **+0.86** ✓ | >0 |
| mean per-stage level var | 0.13 | 1.31 | **0.49** | 작게 |
| cond5 분리 dev (null≈750) | 2,968 | 36,490 | **28,174**, perm p=**0.005** | null≫, 유의 |
| **전체표현 점프 W_ALL** | 0.237 | 0.394 ✗ | **0.226** ✓ | < diff |
| universe md5 (=diff) | 3f54d1a51c71 | — | **3f54d1a51c71** ✓ | 동일 |

→ 새 ours는 **① 난이도 단조·타이트(성능 회복) ② cond5 통계 분리(subject 2축 입증) ③ 전체표현 점프 < diff(논문 주장 실현)**를 동시에 만족. 현 ours는 ①③ 위반.

전체표현 점프는 4개 윈도우 모두 동일 결론(현ours 큼, 새ours 작음): W_ALL 0.226 / W_SL 0.236 / W_SUBJ 0.293 / W_LEV 0.218 (vs diff 0.237/0.247/0.294/0.233). 전 레이어(W_ALL)에서도 깨끗.

---

## 4. 새 ours stage 구성 (α=2.0, unit-atomic)
| stage | 문제 수 | 평균 난이도 | level 범위 | unit 수 |
|---|---|---|---|---|
| 0 | 5,323 | 1.90 | L1–3 | 14 |
| 1 | 5,365 | 3.04 | L2–4 | 7 |
| 2 | 6,171 | 3.90 | L3–5 | 7 |
| 3 | 5,775 | 4.89 | L4–6 | 7 |
| 4 | 6,137 | 5.97 | L5–8 | 15 |

stage별 unit (subject|level: n):
- **s0** (L1–3): PreAlg L1(888)·L2(1460), Alg L1(165)·L2(1190), Precal L1(18)·L2(133)·L3(549), Geo L1(37)·L2(356), IntAlg L1(16)·L2(85)·L3(231), C&P L1(106), NumTh L1(89)
- **s1** (L2–4): Alg L3(1805), Geo L3(797), Precal L4(755), IntAlg L4(464), PreAlg L3(538), C&P L2(561), NumTh L2(445)
- **s2** (L3–5): Alg L4(1724), Geo L4(1317), C&P L3(961), NumTh L3(909), IntAlg L5(725), Precal L5(518), PreAlg L4(17)
- **s3** (L4–6): Geo L5(1389), Alg L5(1365)·L6(495), NumTh L4(1101), C&P L4(713), IntAlg L6(517), Precal L6(195)
- **s4** (L5–8): NumTh L5(1271)·L6(1033)·L7(565)·L8(23), C&P L5(720)·L6(480)·L7(308)·L8(10), Geo L6(870)·L7(491)·L8(9), IntAlg L7(240)·L8(11), Alg L7(79), Precal L7(27)

메커니즘 예 (L3 난이도가 subject 기하로 갈림): **Precal L3·IntAlg L3 → s0**, Alg/Geo/PreAlg L3 → s1, **C&P L3·NumTh L3 → s2**. 난이도는 단조, 같은 난이도대를 **대수→기하→이산/정수론** 순으로 배치.

---

## 5. 산출 파일
| 파일 | 내용 |
|---|---|
| `build_stages_subjslack.py` | 재현 빌드 스크립트 (CPU, artifacts.npz 재사용, 활성 재계산 X) |
| `stages_cond3_ours_subjslack.json` | **새 main "ours"** (deliverable) — stages[*].problem_ids |
| `stages_cond2_diff.json` | 난이도-only baseline (동일 universe) |
| `stages_cond5_diffmatched_seed{0,1,2}.json` | control: ours per-level stage count 매칭 + level내 subject 랜덤 |
| `g_subject_axis.json` | g(subject) 값 + 출처 |
| `manifest.json` | params + 검증 metrics |
| `measure_fulljump.py` | 전체표현 consecutive 점프 측정 (pooled 로드 필요) |

manifest 스키마: 학습 로더(`curriculum_schedule_manifest_once.py`)는 `stages[*].problem_ids`만 사용. `opsd_indices`/`items`는 호환·감사용 부가 필드.

---

## 6. 다음 단계
1. **mini50/100/q4 subsample**: 기존 `make_mini_manifests.py`/`make_quarter_manifests.py` 방식으로 새 ours·diff·cond5의 stratified subsample 생성 (rung 내 cross-arm universe 동일 유지).
2. **학습 재실행** (eval로 현 diff>ours 재현 확정 후): mini→q4→full, cond2_diff vs cond3_ours_subjslack (+ cond5).
3. **핵심 검증 비교**: 새 ours가 (a) diff를 따라잡/넘는가, (b) **cond5를 이기는가** ← subject 기하 효과 입증의 load-bearing.

> 안전: 공용계정. sbatch만, GPU≤4, L40S 우선(C-07 l40sq / C-03 h200q), 타인 job 미조작. upstream `opsd_src/` 미수정.
