# Representation-Guided Curriculum for On-Policy Self-Distillation — 진행 보고 (2026-06-27)

> 목차 (전체 골격). 이번 갱신에서는 **0. 요약**과 **1. Motivation**을 채웁니다.
> 나머지 절은 다음 단계에서 작성합니다.
>
> - 0. 요약 (Executive Summary) — *작성됨*
> - 1. Motivation — 왜 representation-guided curriculum이 OPSD와 결합되어야 하는가 — *작성됨*
> - 2. Method (표현축 g, 2축 score, stage 구성, controls) — _작성 예정_
> - 3. Experimental Setup — _작성 예정_
> - 4. Results (분포 검증 / 최종 성능 / stage별 분석 / cond5) — _작성 예정_
> - 5. Interpretation & Framing (claim의 두 수준) — _작성 예정_
> - 6. Limitations & Risks — _작성 예정_
> - 7. Plan / Next steps — _작성 예정_
> - Appendix — _작성 예정_

---

## 0. 요약 (Executive Summary)

**한 줄 결과.** 모델의 internal representation에서 뽑아낸, 난이도와 독립적인 두 번째 축을 따라 OPSD 학습 순서를 다시 짰더니, 전체 데이터(N = 28,771)로 끝까지 학습한 최종 시점에서 AIME 2024·2025, HMMT 2025, MATH-500 네 개 벤치마크 **모두**에서 난이도만 사용한 기존 커리큘럼을 평균 **+4.1%p** 앞섰습니다.

이 결과는 "난이도 외에 표현 기하(representation geometry)에서 유도한 축이 OPSD 커리큘럼을 실제로 개선한다"는 핵심 주장을 뒷받침합니다. 다만 그 개선이 **subject(과목) 기하에 고유하게 귀속되는지**를 못박는 통제 실험(difficulty-matched random control)은 현재 학습 대기 중이며, 이 결과가 나오면 주장의 강도를 확정할 수 있습니다(5절 참조).

---

## 1. Motivation — 왜 representation-guided curriculum이 OPSD와 결합되어야 하는가

### 1-1. 교수님 지적의 핵심

교수님께서 "왜 representation-guided 커리큘럼이 하필 OPSD와 결합되어야 하는가"라는 질문을 주셨습니다. 현재 우리 스토리는 "표현이 비슷한 문제끼리 모아 매끄러운 순서로 학습한다"는 것인데, 이 논리는 일반적인 supervised fine-tuning이나 GRPO 같은 다른 학습 방법에도 그대로 적용됩니다. 즉 OPSD가 반드시 필요한 이유가 드러나지 않기 때문에, OPSD를 다른 학습법으로 바꿔 끼워도 이야기가 성립해버립니다. 이것이 motivation이 약하다는 지적의 핵심으로 보입니다.

이 지적에 답하려면, 우리가 커리큘럼 정렬에 쓰는 "표현 축"이 **OPSD라는 학습 방식에서만 특별한 의미를 갖는다**는 것을 보여야 합니다. 아래 두 아이디어는 각각 그 답의 절반씩을 담당합니다. 아이디어 2는 "표현 축이 왜 하필 OPSD에서 의미 있는가"를 **진단**하고, 아이디어 1은 그 진단을 활용해 OPSD와 커리큘럼을 **구조적으로 묶는 처방**을 제시합니다.

### 1-2. 먼저, OPSD의 작동 방식 (한 문단 정리)

OPSD(On-Policy Self-Distillation)에서는 하나의 모델이 **teacher와 student 두 역할**을 동시에 맡습니다. student는 문제만 보고 스스로 풀이(rollout)를 생성하고, teacher는 **같은 모델이지만 reference solution(정답 풀이)을 함께 본 상태**에서 student가 생성한 바로 그 토큰들에 대한 확률 분포를 다시 계산합니다. 학습 신호는 두 분포 사이의 JSD(Jensen-Shannon divergence)이며, student의 rollout 토큰 위에서 계산됩니다. 우리 설정에서 teacher는 base 모델로 고정(fixed teacher)됩니다.

여기서 핵심은 **teacher의 역할이 "정답을 알고 있는 상태에서 풀이를 합리화(rationalize)해 주는 것"**이라는 점입니다. teacher는 정답 풀이를 봤기 때문에, student가 어디로 가야 하는지를 더 일관되게 가리키는 목표 분포를 제공합니다. 따라서 **teacher가 어떤 문제를 얼마나 잘 합리화하느냐가 그 문제에서 student가 받을 수 있는 학습 신호의 질을 좌우**합니다. 이 "teacher의 합리화 능력"은 SFT(목표가 고정된 정답 텍스트)나 GRPO(신호가 스칼라 보상)에는 존재하지 않는, OPSD에만 있는 양입니다.

### 1-3. 아이디어 1 — 단계별 Teacher 갱신 (Stage-wise Teacher Upgrade)

표준 OPSD에서 teacher는 학습 내내 고정된 base 모델입니다. 그런데 base 모델은 어려운 문제일수록 정답 풀이를 잘 합리화하지 못할 수 있고, 그러면 정작 어려운 구간에서 학습 신호가 약해집니다. 커리큘럼은 이 문제에 자연스러운 해법을 제공합니다.

커리큘럼은 학습을 여러 stage로 나누어 쉬운 구간부터 차례로 통과합니다. 어떤 stage를 마치고 나면 모델은 그 구간에 대해 이미 향상된 상태입니다. 이때 **향상된 student를 다음 stage의 새로운 teacher로 승격**시키면, teacher의 합리화 능력이 커리큘럼을 따라 점진적으로 강해집니다. 즉 모델이 쉬운 stage를 오르며 쌓은 실력이, 바로 다음의 더 어려운 stage에서 teacher 신호의 질을 끌어올리는 데 재투자됩니다. base teacher가 가장 약했던 어려운 구간에서, teacher가 가장 강해진 상태로 신호를 주게 되는 것입니다.

이 메커니즘이 중요한 이유는, **OPSD와 커리큘럼이 둘 다 있어야만 정의된다**는 데 있습니다. teacher 역할이 없는 SFT나 GRPO에서는 "teacher를 갱신한다"는 개념 자체가 성립하지 않고, stage 구조가 없으면 "언제 갱신할지"의 스케줄이 없습니다. 커리큘럼의 stage 경계가 teacher 자기-개선의 자연스러운 시점을 제공하고, OPSD의 teacher 역할이 그 개선이 꽂힐 자리를 제공합니다. 따라서 "representation-guided curriculum + OPSD"의 결합은 임의적이지 않으며, 둘은 서로의 부품을 필요로 합니다.

### 1-4. 아이디어 2 — Rationalizability Alignment (표현–합리화가능성 정렬)

아이디어 1이 처방이라면, 아이디어 2는 그 처방을 정당화하는 진단입니다. 우리는 "teacher가 어떤 문제를 얼마나 잘 합리화하는가"를 **rationalizability(합리화가능성)**로 정의하고, 이를 정량화했습니다. 구체적으로, 문제만 주어진 prompt에서 teacher가 reference solution을 강제 디코딩(forced decoding)했을 때의 NLL(negative log-likelihood)을 측정합니다. NLL이 낮을수록 teacher가 그 정답 풀이를 더 자연스럽게 받아들인다는 뜻이고, 곧 더 잘 합리화한다는 뜻입니다.

핵심 관찰은 다음과 같습니다. 이 rationalizability(NLL)는 우리가 커리큘럼 정렬에 쓰는 **표현 축 $g$와 강하게 정렬**되어 있는 반면(상관계수 +0.253, p ≈ 6×10⁻⁴¹), **난이도 level과는 사실상 무관**했습니다(상관계수 +0.005, p ≈ 0.78). 즉 표현 축을 따라 커리큘럼을 정렬한다는 것은, 우연히도 **teacher가 잘 합리화하는 문제에서 잘 못하는 문제 순으로 정렬**하는 것과 같으며, 이 정렬은 단순한 난이도 정렬과는 별개의 정보입니다. (부호 방향: $g$가 큰 후반 구간일수록 NLL이 높아 — 즉 합리화가 더 어려워 — 더 늦게 배치됩니다.)

이 관찰이 교수님의 질문에 직접 답합니다. 우리가 쓰는 표현 축은 그저 "비슷한 문제 묶기"가 아니라, **OPSD의 학습 신호 그 자체인 teacher rationalizability를 예측하는 축**입니다. SFT에는 teacher 합리화라는 개념이 없고 GRPO도 마찬가지이므로, 그곳에서 $g$로 정렬하는 것은 특별한 의미가 없습니다. 오직 OPSD에서만 문제별 학습 신호의 질이 teacher의 합리화 능력에 달려 있고, $g$가 바로 그것을 예측합니다. 따라서 representation-guided curriculum은 OPSD와 결합될 때 비로소 "학습 신호 질의 순서로 학습한다"는 고유한 의미를 갖습니다.

### 1-5. 두 아이디어가 합쳐서 답하는 것

정리하면, OPSD를 다른 학습법으로 갈아 끼울 수 없는 이유는 두 가지입니다.

1. **진단(아이디어 2).** 커리큘럼의 정렬 축인 표현 $g$는 경험적으로 teacher rationalizability를 예측하는 축이며, 이 양은 OPSD에만 존재합니다. 그래서 $g$ 기반 정렬은 OPSD에서만 "학습 신호 질에 따른 정렬"이 됩니다.
2. **처방(아이디어 1).** 커리큘럼의 stage 구조는 teacher를 점진적으로 강화하는 자연스러운 스케줄을 제공하고, OPSD의 teacher 역할은 그 강화가 작동할 자리를 제공합니다. 둘 다 있어야만 stage-wise teacher upgrade가 정의됩니다.

OPSD를 SFT나 GRPO로 바꾸면 진단(합리화가능성이 신호)도, 처방(teacher 갱신)도 의미를 잃습니다. 이것이 representation-guided curriculum이 하필 OPSD와 결합되어야 하는 이유입니다.
