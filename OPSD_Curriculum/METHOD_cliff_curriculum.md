# METHOD — Activation-Guided "Cliff" Curriculum (2026-06-30)

논문 Method 초안. 실험 모델 = **Qwen3-4B** (H100), 8B는 cond5 마무리만.

---

## 1. 동기
원래 main이던 level×subject 커리큘럼(ours/subjslack)은 난이도 baseline(diff)은 이기지만 **무작위-subject 대조(cond5)는 못 이김** → subject 순서 효과 ≈ null. 그러나 ours가 diff를 이긴 진짜 원인은 **혼합-레벨(겹치는) stage 구성**이었다. 본 method는 그 "혼합/완만함"을 **모델 내부 표현 기하(activation)** 로부터 원리적으로 재구성한다.

## 2. Activation 분석 (경험적 기반)
1. **추출**: 추론 span hidden-state shift `dA = A(t_K) − A(t_1)`, 36층. pooled 3025문제(pilot1+pilot2). 난이도 채널 = **층 16–32** 평균, pooled global-mean centering.
2. **레벨 기하**: 레벨별 centroid `c_L`(그 레벨 문제 평균). 레벨 유사도 `S[i,j]=cos(c_i,c_j)` (8×8).
3. **재현성**: pilot1 vs pilot2 행렬 상관 **r=0.996**. 8B·4B 모두 동일 구조(모델 불변).
4. **표현 gap**: `g_i = 1 − S[i,i+1]`. 4B = [.079,.232,**.672**,.393,.077,.113,.045].
5. **구조**: 두 클러스터(L1–3 / L5–8) + **L4 절벽(bridge)**. **L3–L4–L5 = 전체 표현거리의 64%**.
6. **누적 위치**: `pos(L)=Σ_{i<L} g_i`. 4B pos = [0,.079,.311,.983,1.376,1.453,1.566,1.611].

> 주의: 활성로 *난이도 자체*를 재정의하려는 시도는 실패(길이와 얽힘, 길이는 난이도의 증상). 따라서 **난이도 backbone = 사람 level 유지**, activation은 **레벨 사이 구조(gap)** 에만 사용.

## 3. 커리큘럼 구성 규칙
- **stage(슬라이딩 창)**: 너비-3, `S_k = {L_{k+1},L_{k+2},L_{k+3}}`, k=0..5. 인접 stage 2레벨 겹침 → 예/복습. 한 레벨이 여러 stage에 **반복** 등장(절벽 레벨 3×).
- **dwell(학습시간 배분)**: `dwell_k ∝ 0.5·(span_k/Σspan) + 0.5·(1/K)`, 하한 11% 후 정규화. `span_k = pos(maxL_k) − pos(minL_k)`. → 표현거리 큰 절벽 stage에 시간 tilt(약하게) + 클러스터 바닥 보호.
- **context scaling**: `c_k = f(S_k 최난도 레벨의 non-thinking 길이)`, 512 배수 올림. 1024→4096 ramp(쉬운 stage 짧게=저렴, 하드 stage 길게=reasoning 안 잘림).

## 4. 최종 스펙 (clean universe 28,743 · T=900 step 예시)
| stage | levels | 문제수 M_k | dwell | step | passes | context |
|---|---|---|---|---|---|---|
| S0 | L1–L3 | 11,319 | 13% | 120 | 0.34× | 1024 |
| S1 | L2–L4 | 16,093 | 23% | 206 | 0.41× | 1536 |
| S2 | L3–L5 | 17,858 | 25% | 229 | 0.41× | 2048 |
| S3 | L4–L6 | 15,661 | 16% | 143 | 0.29× | 2560 |
| S4 | L5–L7 | 11,281 | 11% | 102 | 0.29× | 3072 |
| S5 | L6–L8 | 5,351 | 11% | 99 | 0.59× | 4096 |

**문제 1개당 유효 노출(passes 합)**: L1:.34 L2:.75 **L3:1.16 L4:1.11 L5:.99** L6:1.17 L7:.88 L8:.59.
→ 절벽(L3–L5)≈1.1, 하드(L6–7)≈1.0, 쉬운(L1).34. 어느 레벨도 1.2× 초과 반복 없음(오버핏 안전).

> 레벨 분포 매우 불균등 (L1:1316 … L4:6090 … **L8:53**). L8은 0.2%뿐 → 별도 L8 보강 stage는 53문제 반복=오버핏이라 **제외**(L8은 S5에 포함). dwell 비율은 고정, T는 4B 실제 step수로 적용.

## 5. 실험 설계 (4B, clean셋, 동일 T, 동일 context 스케줄)

**cliff 두 형태**:
- **cliff-P (partition, 메인)**: 각 레벨을 슬라이딩-창 stage들에 *분배*(각 문제 1번, 노출 1× 균일 = diff와 동일 → 형평성 깔끔). stage = 3레벨 부드러운 혼합 + context ramp. 절벽 강조는 "절벽 stage가 큼(자연 분포)"으로만.
- **cliff-R (repetition, 변형)**: 절벽 레벨을 여러 stage에 *반복*(절벽 3× 노출). 절벽 강조 셈, 약한 오버핏 위험.

**arm 세트**:
| arm | 구성 | 노출 |
|---|---|---|
| diff | tight level 밴드 | 1× 균일 |
| cliff-P | smooth 3레벨 밴드+ctx, partition | 1× 균일 |
| cliff-R | 절벽 3× 반복+ctx | 절벽↑ |
| cliff-R-shuf | cliff-R 노출 매칭, 배정 셔플 | = cliff-R |
| subj-V1 | 과목 시퀀싱(톱니, subject-primary) | 1× 균일 |
| subj-shuf | 과목 블록 랜덤순서 | = subj-V1 |

**판정**: cliff-P>diff = 절벽 밴드 효과(공정) · cliff-R>cliff-P = 반복강조 추가효과 · **cliff-R>cliff-R-shuf = 절벽 *구조* 효과**(load-bearing) · subj-V1>diff&>subj-shuf = subject 1차축 부활(지면 종료).

**cliff-P 구성 (partition, 각 문제 1번)**:
| stage | 문제수 | context | 레벨 혼합 |
|---|---|---|---|
| S0 | 4,072 | 1024 | L1:1316 L2:1523 L3:1233 |
| S1 | 7,065 | 1536 | L2:2695 L3:2181 L4:2189 |
| S2 | 7,626 | 2048 | L3:2371 L4:2379 L5:2876 |
| S3 | 4,874 | 2560 | L4:1522 L5:1841 L6:1511 |
| S4 | 3,159 | 3072 | L5:1266 L6:1039 L7:855 |
| S5 | 1,947 | 4096 | L6:1039 L7:855 L8:53 |

**subj-V1 구성 (과목 blocked, 톱니 난이도)**: 대수(L1-4:7120, L5-8:4165) → 기하(2507, 2759) → 이산(7782, 4410). subject-primary(cond5 미검정 영역).

### 공정성/노출 노트
- 과거 diff·ours·cond5는 **레벨별 노출 완전 동일**(전부 partition·1 epoch, 하드 L5-8 = 11,341 동일) → ours 이득은 "하드 더"가 아니라 "순서/구조"임을 검증함.
- 새 cliff는 **의도적으로 노출을 재분배**(절벽↑) → 따라서 노출-매칭 **control(C)이 필수**. 모든 arm은 동일 compute(T) 사용.

## 6. 과거 ours와의 차이 (요약)
| | ours(subjslack) | cliff(new) |
|---|---|---|
| 혼합 stage 생성 | 레벨을 **분할**(subject g가 결정) | 레벨을 **반복**(슬라이딩 창) |
| 반복 | 없음(1×) | 있음(절벽 3×) |
| 배치 결정축 | subject 기하(null) | 난이도 표현거리(gap) |
| 시간 배분 | 균일 | gap-tilt dwell |
| context | 고정 1024 | 1024→4096 |
| subject | 핵심 | 제외 |

핵심: cliff = "ours에서 실제로 일한 혼합/완만함을, subject 대신 **activation 난이도 기하**로 원리적으로 만들고 + **절벽 강조(반복·dwell) + context**를 더한 후속작."

## 7. 데이터 위생
- eval(aime24/25·hmmt25·math500) ↔ 30K train 오염: AIME/HMMT 0, math500 진짜 중복 3 + 보수적 제거 **총 28개(0.10%)**. clean universe = 28,743. 4B는 clean셋 학습.

## 8. 안전/운영
- H100(신규): 직접 실행, env/모델/데이터 신규 세팅, 결과는 repo로 공유.
- 이 서버: 공용계정, C-03/C-07만, GPU 풀당4, upstream opsd_src 미수정.
