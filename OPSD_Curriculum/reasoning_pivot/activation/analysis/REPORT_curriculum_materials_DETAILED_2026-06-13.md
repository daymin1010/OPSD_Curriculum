# Curriculum Materials — 상세 해석 보고서 (pooled THINKING ΔA, tag=currmat)

작성: 2026-06-13 / 트랙: reasoning_pivot, pooled(pilot1+pilot2) 분석
원천(기계 출력, 덮어쓰기 금지): `REPORT_curriculum_materials.md`
재현 아티팩트: `currmat_artifacts.npz` · 그림: `dendro_layeravg_currmat.png`, `dendro_midL11-15_currmat.png`
실행 로그: `runs/curriculum_materials_detached_20260613_034023.log` (03:40 시작 → 04:05 종료, ~25분, 에러 0)

> 이 문서는 기계 출력 리포트의 수치를 **해석**한 것이며, stage 경계·개수·schedule 같은 커리큘럼 확정 결정은 **사용자 review 후** 별도로 진행한다.

---

## 0. 실행 메타 / 방법론 disclaimer

| 항목 | 값 |
|---|---|
| pooled N | **3025** (pilot1=1608, pilot2=1417) |
| subjects | 8-canonical |
| levels | 1–8 |
| is_correct non-null | pilot1=1607, pilot2=1417, total=3024 |
| overall 1-shot 정답률(non-null) | **0.818** |
| 자원 | **CPU only** (GPU 미사용) |
| 신호 | **THINKING ΔA** (primary). FAITHFUL은 보조. |

⚠️ **방법론 경계**: 본 분석에서 쓴 group-similarity / centering / permutation 검정은 **우리의 자체 진단 도구**이지, NAIT 원논문의 PCA-scoring 절차가 아니다. 여기서 말하는 "difficulty axis"는 활성화 공간 위의 회귀/주성분 축이며, 모델이 내부적으로 쓰는 점수와 동일하다는 보장은 없다.

---

## 1. TASK 1 — 난이도 축 (difficulty axis)

**설계**: PCA는 pilot1(n=1608)에 fit하고 pilot2(n=1417)에 projection하여 **out-of-sample(정직한) 비교**를 한다. supervised ridge도 pilot1에서 5-fold CV로 α를 고른 뒤 pilot2 테스트에서 평가한다. 모든 ρ은 pilot2 test 기준 Spearman.

### 1.1 비지도 PC 후보 (pilot2 test)
| PC | EVR | ρ(level) | ρ(is_correct) | ρ(gen_len) |
|----|-----|----------|---------------|------------|
| PC1 | 0.4266 | **+0.709** | −0.378 | **+0.729** |
| PC2 | 0.0967 | +0.616 | +0.051 | +0.429 |
| PC3 | 0.0537 | +0.413 | +0.001 | +0.290 |

### 1.2 지도 ridge (dual; α=10000, pilot1 5-fold CV ρ=+0.941)
| axis | ρ(level) | ρ(is_correct) | ρ(gen_len) |
|------|----------|---------------|------------|
| ridge_level | **+0.937** | −0.280 | +0.745 |

### 1.3 채택 및 해석
- **채택 난이도 축 = `ridge_level`** (pilot2 test에서 |ρ(level)| 최대, +0.937). pilot1→pilot2 out-of-sample이므로 과적합이 아닌 일반화된 축이다.
- **PC1은 EVR 43%로 지배적이지만 난이도 전용 축이 아니다.** ρ(level)=+0.709와 ρ(gen_len)=+0.729가 거의 같다 → PC1의 변동은 상당 부분 **생성 길이(공통 성분)**가 끌고 있다. 어려운 문제일수록 답이 길어지는 자연스러운 상관 때문에 level과 길이가 얽혀 있다.
- **gen_len 동행 관련(보류)**: 채택한 ridge_level은 ρ(gen_len)=+0.745로 생성 길이와도 강하게 동행한다. 다만 **난이도와 생성 길이는 본질적으로 비례하는 관계**(어려운 문제일수록 추론·서술이 길어짐)로 보는 것이 자연스럽다. 따라서 이 동행을 곧바로 "length confound(난이도 축이 사실은 길이 축)"라고 단정하기는 이르다. 길이를 난이도의 정당한 부수 신호로 볼지, 별도 교란으로 분리할지는 **현 단계에서 판단을 보류**하고 추후 필요 시 재검한다.
- ρ(is_correct)가 음수(−0.28~−0.38)인 것은 정상: 어려운 문제일수록 1-shot 정답률이 떨어진다(부호 방향 sanity OK).
- [side sanity] pooled-all PC1: ρ(level)=+0.707, ρ(gen_len)=+0.739 — 위 비교와는 다른 샘플이지만 동일 패턴 재현(길이-난이도 동행).

---

## 2. TASK 2 — 두 축(subject / level) 분해

활성화 구조가 **난이도(level)**로 갈리는가, **과목(subject)**으로 갈리는가? unit(n≥10) = **57개**로 분석.

### 2.1 unit-centroid 코사인 (pair type별, 낮을수록 분리됨)
| pair type | mean cos | n_pairs | 해석 |
|---|---|---|---|
| same-level / diff-subject | **+0.456** | 181 | 같은 난이도 안에서 과목 간 거리. 여전히 높음 → 과목으로는 잘 안 갈림 |
| both-diff (baseline) | −0.104 | 1234 | 둘 다 다르면 거의 무상관~약한 반대 |

### 2.2 same-subject / diff-level (Δlevel별)
| Δlevel | mean cos | n |
|---|---|---|
| 1 | +0.734 | 49 |
| 2 | +0.401 | 41 |
| 3 | +0.019 | 33 |
| 4 | −0.300 | 25 |
| 5 | −0.467 | 18 |
| 6 | −0.485 | 11 |
| 7 | −0.387 | 4 |

- **ordinality ρ(cos, −Δ) = +0.893** → 난이도 거리가 멀수록 단조적으로 유사도가 떨어진다. 활성화 공간에 **연속적인 난이도 축**이 실재한다는 강한 증거.

### 2.3 조건부 분리도 (sample-level, block-restricted perm N=200)
| 비교 | gap | p | 해석 |
|---|---|---|---|
| within-level / between-**SUBJECT** | **−0.0418** | 0.0050 | 같은 난이도 안에서 과목 분리도 → **거의 없음(음수)** |
| within-subject / between-**LEVEL** | **+0.2274** | 0.0050 | 같은 과목 안에서 난이도 분리도 → **뚜렷이 분리** |

### 2.4 TASK2 결론
> **활성화 구조의 1차 결정자는 LEVEL(난이도)이다. SUBJECT(과목)의 기여는 매우 약하다.**
> within-LEVEL의 subject gap이 음수(−0.04)라는 것은, 난이도를 고정하면 과목으로는 사실상 갈라지지 않음을 의미한다. 반대로 과목을 고정해도 난이도로는 +0.23만큼 뚜렷이 갈린다. Δlevel 단조성(ρ=+0.893)이 이를 보강한다.

---

## 3. TASK 3+4 — 합동 클러스터링 & 과목 분기(subject branching)

두 레이어 뷰로 Ward(cosine) 클러스터링. unit=57, sparse(n<10)는 제외.

### 3.1 [layeravg] (feat dim=12288)
- silhouette 최적 **K=7 (sil=0.487)** — K별: 4→0.411, 5→0.438, 6→0.470, **7→0.487**, 8→0.465
- 그림: `dendro_layeravg_currmat.png`

cluster × level (대각 정렬이 강함 = level 띠):
```
level    1  2  3  4  5  6  7  8
C1       0  0  0  1  2  4  7  4   (고난도 L6-8)
C2       0  0  0  2  4  3  0  0   (중상 L4-6)
C3       0  0  1  1  0  0  0  0
C4       0  0  3  1  0  0  0  0
C5       7  7  1  0  0  0  0  0   (최易 L1-2)
C6       0  0  2  2  0  0  0  0
C7       1  1  1  1  1  0  0  0   ('Other' 전용, level 퍼짐)
```
- **subject-branching index = 90/181 = 0.497** (같은 난이도 unit쌍이 서로 다른 cluster에 들어가는 비율)
- 평균 cluster subject-entropy = 1.46 bits, level-entropy = 1.43 bits
- 특이: **C7은 전부 'Other' 과목**(subj_H=0)으로, level이 1–5에 퍼져도 하나의 cluster를 형성 → 'Other'만은 과목 정체성이 강하다.

### 3.2 [midL11-15] (feat dim=61440, 중간층 11–15)
- silhouette 최적 **K=6 (sil=0.391)** — K별: 4→0.369, 5→0.375, **6→0.391**, 7→0.385, 8→0.362
- 그림: `dendro_midL11-15_currmat.png`

cluster × level:
```
level    1  2  3  4  5  6  7  8
C1       0  0  0  1  4  2  0  0
C2       0  0  0  0  1  5  7  4   (고난도)
C3       1  1  1  1  1  0  0  0   ('Other' 전용)
C4       0  0  2  2  0  0  0  0
C5       7  7  2  0  0  0  0  0   (최易)
C6       0  0  3  4  1  0  0  0
```
- **subject-branching index = 83/181 = 0.459**
- 평균 cluster subject-entropy = 1.71 bits, level-entropy = 1.55 bits
- 여기서도 C3 = 전부 'Other'로 재현.

### 3.3 과목 분기 판정 (verdict)
- branching index: layeravg=0.497 vs mid-layer=0.459 → 둘 다 **약 0.5 (중간 수준)**.
- cluster×level 표가 두 뷰 모두 **강하게 대각**(level 띠) → 클러스터의 1차 구성 원리는 난이도.
- 단 'Other' 과목은 두 뷰 모두 **독립 cluster**를 형성(C7/C3) → 일부 과목은 난이도와 무관한 고유 활성화 패턴을 가진다.
> **판정: 구조는 주로 level-driven. subject 분기는 보조적이며 'Other'에 국한.** "같은 난이도가 과목별로 갈라진다"는 joint(novelty) 주장을 강하게 펴기엔 branching이 충분히 크지 않다. 정직하게 'level-driven + 일부 과목 특이성'으로 보고하는 것이 타당.

---

## 4. TASK 5 — 클러스터 난이도 정렬 (커리큘럼 재료)

`ridge_level` 축의 cluster 평균 점수로 easy→hard 정렬. **이는 재료일 뿐, 최종 stage가 아니다.**

### 4.1 [layeravg] easy→hard
| 순위 | cluster | score | levels | n_units |
|---|---|---|---|---|
| 1 | C5 | −2.446 | 1,2,3 | 15 |
| 2 | C7 | −0.852 | 1–5 ('Other') | 5 |
| 3 | C4 | −0.752 | 3,4 | 4 |
| 4 | C6 | −0.546 | 3,4 | 4 |
| 5 | C3 | −0.441 | 3,4 | 2 |
| 6 | C2 | +0.964 | 4,5,6 | 9 |
| 7 | C1 | +2.337 | 4,5,6,7,8 | 18 |

### 4.2 [midL11-15] easy→hard
| 순위 | cluster | score | levels | n_units |
|---|---|---|---|---|
| 1 | C5 | −2.349 | 1,2,3 | 16 |
| 2 | C3 | −0.852 | 1–5 ('Other') | 5 |
| 3 | C4 | −0.546 | 3,4 | 4 |
| 4 | C6 | −0.228 | 3,4,5 | 8 |
| 5 | C1 | +0.975 | 4,5,6 | 7 |
| 6 | C2 | +2.515 | 5,6,7,8 | 17 |

### 4.3 해석
- 두 뷰 모두 **단조적·일관된 난이도 정렬**: 최易(C5, L1-3) → 중간(L3-5) → 고난도(L6-8). 점수 간격이 양극(±2.x)에서 크고 중간대가 촘촘 → 자연스러운 3-band(쉬움/중간/어려움) 구조가 보인다.
- 'Other' 과목 cluster는 두 뷰 모두 점수상 **두 번째(약간 쉬움)**로 위치 — level이 퍼져 있어 난이도 해석은 주의.

---

## 5. 한계 및 다음 단계

1. **gen_len 동행 (판단 보류)**: 채택 난이도 축이 gen_len과 ρ=+0.745로 동행하나, **난이도와 생성 길이는 본질적으로 비례**(어려울수록 추론이 길어짐)하므로 이를 confound로 단정하지 않고 **현 단계에서는 보류**한다. 추후 필요 시 gen_len 잔차화 재검을 옵션으로 둔다. (참고 자산: `src/4.6_Task2/.../nait_length_confound.py`, `full_final/length_confound_report.md`)
2. **Sparse unit 처리**: Geometry|L8 (n=9, 1개 unit)은 클러스터링에서 제외됨. nearest-cluster 흡수 대상으로만 표기, 단독 결론 금지.
3. **L8 출처**: L8 66개는 전부 pilot1 출처(handoff §1). L8 단독 결론 금지(n 부족 + subject 불균형).
4. **Stage 확정은 사람 결정**: §4의 정렬은 재료. stage 개수/경계/schedule은 사용자 review 후 확정.
5. **레이어 뷰 선택**: layeravg(sil 0.487)가 mid-layer(0.391)보다 분리가 좋음 → 커리큘럼 재료로는 layeravg K=7을 1순위 후보로 권장하되, mid-layer와의 일관성(둘 다 동일 난이도 정렬)으로 robustness 확보됨.

---

## 6. 산출물 인덱스

| 파일 | 역할 |
|---|---|
| `REPORT_curriculum_materials.md` | 기계 출력 수치 리포트 (canonical raw, 덮어쓰기 금지) |
| `REPORT_curriculum_materials_DETAILED_2026-06-13.md` | **본 문서** — 해석/결론 |
| `currmat_artifacts.npz` | 재현용 행렬·라벨·클러스터 배정 (45KB) |
| `dendro_layeravg_currmat.png` | layeravg Ward 덴드로그램 (K=7) |
| `dendro_midL11-15_currmat.png` | mid-layer Ward 덴드로그램 (K=6) |
| `runs/curriculum_materials_detached_20260613_034023.log` | 실행 로그 |
| `curriculum_materials.py` | 분석 스크립트 (재실행: `--cond-perm N`) |

### 한 줄 요약
> pooled 3025 THINKING ΔA에서 활성화 구조는 **난이도(level) 주도**이며(within-subject between-level gap +0.227 p=0.005, Δlevel 단조성 ρ=0.893), 과목 분기는 'Other'를 제외하면 약하다. 난이도 축(ridge_level, out-of-sample ρ=0.937)은 생성 길이와도 동행(ρ=0.745)하나 난이도-길이 비례는 자연스러운 관계로 보아 **현 단계에서 confound 판단은 보류**한다. 클러스터 난이도 정렬은 두 레이어 뷰에서 일관적이며 커리큘럼 stage 설계의 재료로 사용 가능(확정은 사람 review).

---

## 부록 A. 직관 요약 / FAQ — "같은 레벨·다른 과목" vs "같은 과목·다른 레벨"

**Q. 같은 레벨인데 과목이 다르면 activation이 다른 경우가 많았나? / 같은 과목인데 레벨이 다르면 다른 경우가 많았나?**

**A. 비대칭적이다. "같은 과목·다른 레벨"에서 activation이 다른 경우는 매우 많았고, "같은 레벨·다른 과목"에서 다른 경우는 거의 없었다.**

| 조건 | activation이 "달라지는" 정도 | 근거 수치 |
|---|---|---|
| **같은 레벨 / 다른 과목** | **거의 안 달라짐** | same-level·diff-subject unit쌍 평균 cos = **+0.456**(여전히 유사). 조건부 within-level / between-SUBJECT gap = **−0.0418** (p=0.005, 거의 0) |
| **같은 과목 / 다른 레벨** | **많이 달라짐** | 조건부 within-subject / between-LEVEL gap = **+0.2274** (p=0.005). 레벨 차이가 클수록 단조적으로 더 달라짐: Δ=1 +0.734 → Δ=3 +0.019 → Δ=5 −0.467 (ordinality ρ=+0.893) |

**해석**
- 핵심은 두 gap의 대비다: 과목으로 가르는 힘(−0.04)은 사실상 0이지만, 난이도로 가르는 힘(+0.23)은 뚜렷하다. → activation을 결정하는 1차 요인은 **레벨(난이도)**이고, **과목은 거의 영향이 없다.**
- 레벨이 1~2칸 차이일 때는 같은 과목 unit이 여전히 비슷(cos>0)하지만, 4칸 이상 벌어지면 오히려 반대 방향(cos<0)으로 가버린다 → 난이도는 **연속적이고 단조적인 축**으로 작동.
- **유일한 예외는 'Other' 과목**: 레벨이 1~5에 퍼져 있어도 독립 클러스터를 형성(layeravg C7 / mid C3) → 'Other'만 과목 고유의 activation 패턴을 가진다. 나머지 7개 과목은 난이도 띠 안에 섞인다.
- (참고) 이 난이도 축은 생성 길이와도 동행하지만(§1.3), 난이도와 길이가 본질적으로 비례한다고 보면 위 결론(레벨 주도)을 약화시키지 않는다.
