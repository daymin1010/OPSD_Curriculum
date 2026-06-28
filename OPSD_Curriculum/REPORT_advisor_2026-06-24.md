# Representation-aware Curriculum과 On-Policy Self-Distillation — 연구 보고 (2026-06-24)

## 0. 요약

수학 reasoning model을 학습할 때, problem을 **difficulty 순서뿐 아니라 subject 사이의 representation distance까지 반영한 순서**로 제시하면 성능이 향상되는지를 검증합니다. subject distance는 label이 아니라 **model 내부 activation에서** 추정합니다. 학습 방법으로 On-Policy Self-Distillation(OPSD)을 사용합니다.

---

## 1. Background

### 1-1. OPSD

한 model이 teacher와 student 역할을 동시에 맡습니다.

- **student**: problem만 입력으로 받아 solution을 생성합니다.
- **teacher**: 동일 model이지만 **reference solution을 입력에 포함한 상태**에서, student가 생성한 각 token에 대한 분포를 산출합니다.

teacher 분포와 student 분포의 차이가 distillation signal이며, 이 signal로 student를 학습시킵니다.

### 1-2. Curriculum의 필요성

OPSD 원 논문은 limitation을 명시합니다 — **teacher의 capability가 problem difficulty보다 낮으면, teacher가 reference solution을 rationalize하지 못해 distillation signal의 품질이 저하됩니다.** 고난도 problem에서 이 현상이 두드러집니다. easy-to-hard 순서로 학습하는 **curriculum**으로 이 문제를 완화하는 것이 목표입니다.

### 1-3. Research Question

difficulty는 label로 정렬할 수 있습니다. 본 연구는 **subject가 두 번째 정렬 축으로 유효한지**를 검증합니다. subject 사이의 distance는 **category label이 아니라 model 내부 representation에서** 추정합니다. 이 점이 difficulty-only curriculum과의 차이입니다.

---

## 2. Method

### 2-1. Subject geometry axis $g$

model activation에서 **difficulty와 직교하는 subject coordinate**를 추정합니다. problem $i$의 reasoning 중 activation shift $\Delta a_i$를 구한 뒤, **difficulty(level) 평균을 제거**합니다.

$$\tilde a_i = \Delta a_i - \mu_{\ell(i)}, \qquad \mu_{\ell} = \frac{1}{|\{j:\ell(j)=\ell\}|}\sum_{j:\ell(j)=\ell}\Delta a_j$$

동일 unit(subject × level cell)끼리 평균하고 normalize한 centroid를 $c_u$, unit 사이의 cosine similarity matrix를 $M_{uv}=\langle c_u, c_v\rangle$로 둡니다. $M$의 **leading axis**(classical MDS 1차원)를 subject별로 집계하고 $[-0.5, +0.5]$로 scaling한 값이 subject coordinate $g(s)$입니다.

$g<0$인 subject는 이른 stage, $g>0$인 subject는 늦은 stage로 배치합니다. 측정값은 다음과 같습니다.

| subject | $g$ |
| --- | --- |
| Precalculus | −0.50 |
| Intermediate Algebra | −0.45 |
| Algebra | −0.20 |
| Geometry | −0.12 |
| Prealgebra | +0.21 |
| Number Theory | +0.33 |
| Counting & Probability | +0.50 |

### 2-2. Two-axis score와 stage 구성

problem $x$에 score를 부여합니다. difficulty $\ell(x)$를 backbone으로 하고, subject coordinate $g$로 perturbation을 줍니다.

$$\rho(x) = \ell(x) + \alpha \cdot g(s(x))$$

$g \in [-0.5, 0.5]$이고 $\alpha=2.0$이므로, subject 항의 크기는 최대 약 1 level입니다. 따라서 stage별 평균 difficulty는 monotonic 증가를 유지하고, 동일 difficulty 구간 안에서만 subject가 재배치됩니다. $\rho$로 unit을 정렬한 뒤 5개 stage로 분할합니다.

### 2-3. 첫 설계와 수정

초기 설계(tiered)는 **subject 효과가 difficulty 순서와 구별되도록**, subject cluster를 강하게 분리했습니다. 이때 $\rho(\text{difficulty-only}, \text{ours})$의 Spearman 상관을 0.4–0.7 구간으로 통제했습니다. 차별화는 달성했으나 두 결과가 관측됐습니다.

1. 인접 stage 사이의 **representational jump가 difficulty-only보다 증가**했습니다 (full-representation 기준 0.394 vs 0.237). smooth transition이라는 의도와 반대 방향입니다.
2. difficulty backbone이 monotonic하지 않게 되어, level 2–3 구간을 집중 학습하는 stage가 사라졌습니다.

이에 2-2의 설계(subjslack)로 수정했습니다. **difficulty monotonicity를 회복하되 subject 분포는 통계적으로 분리**되도록 한 것입니다. 수정 목적은 "차별성을 유지하면서 representational jump를 감소시키는 것"입니다.

---

## 3. 현재까지의 결과

### 3-1. 성능 (tiered 설계, non-thinking 평가)

맞힌 시도 수 / 전체 시도 수로 정리합니다. AIME/HMMT는 30 problem × 3 trial = 90 trial, MATH-500은 500 problem × 1 trial입니다.

| model | AIME24 | AIME25 | HMMT25 | MATH500 |
| --- | --- | --- | --- | --- |
| base | 23/90 | 20/90 | 10/90 | 423/500 |
| difficulty-only | 60/90 | 38/90 | 21/90 | 415/500 |
| subject-aware (tiered) | 50/90 | 38/90 | 22/90 | 398/500 |

(full dataset 학습 종료 시점 기준입니다.)

- **AIME에서는 difficulty-only가 우위입니다.** 모든 학습 규모에서 일관됩니다.
- **HMMT에서는 subject-aware가 근소 우위입니다.**
- tiered 설계는 **중간 difficulty에서 손해, 극한 difficulty에서 이득**입니다. 2-3의 진단(difficulty backbone 손상)과 일치합니다. 수정 설계(subjslack)가 중간 difficulty 손해를 제거하는지가 현재 핵심 검증이며, 학습이 진행 중입니다.

### 3-2. Subject axis의 유효성 (예비 검증, 학습 불필요)

subject coordinate $g$가 의미 있는 변수인지 확인했습니다. base model에 problem만 입력하고 reference solution을 forced-decode했을 때의 평균 NLL(낮을수록 base에 친숙)을 측정한 뒤, $g$ 및 difficulty와의 상관을 계산했습니다.

| 상관 대상 | Spearman | p-value |
| --- | --- | --- |
| NLL ↔ subject coordinate $g$ | +0.25 | $6\times10^{-41}$ |
| NLL ↔ difficulty level | +0.01 | 0.78 |
| NLL ↔ $g$ (difficulty 통제 후) | +0.26 | $3\times10^{-42}$ |

- subject coordinate $g$는 base model의 solution familiarity를 유의하게 예측합니다. algebra 계열은 NLL이 낮고, combinatorics·number theory 계열은 NLL이 높습니다.
- **difficulty level은 이 familiarity와 무상관($p=0.78$)인 반면, $g$는 상관합니다.** difficulty 통제 후에도 효과가 유지됩니다. 즉 subject axis는 difficulty가 포착하지 못하는 정보를 제공합니다. two-axis curriculum의 직접 근거입니다.
- 단, 이 측정값은 forced-decode familiarity이며 teacher가 reference solution으로 정답을 *생성*하는 rationalizability와는 다릅니다. 후자는 후속 실험으로 확장합니다.

---

## 4. Motivation — "왜 OPSD인가"

### 4-1. 지적

교수님께서 "representation-aware curriculum이 왜 OPSD와 결합되어야 하는가"를 지적하셨습니다. "비슷한 representation의 problem을 모아 smooth하게 학습한다"는 논리는 supervised fine-tuning이나 GRPO에도 적용됩니다. 학습 방법을 교체해도 논리가 성립하므로 OPSD의 필연성이 약합니다. 아래 두 방향으로 이 공백을 보완합니다. 공통 근거는 1-2의 OPSD limitation(teacher가 reference solution을 rationalize하지 못하면 signal이 저하됨)입니다.

### 4-2. Idea 1 — Stage-wise Teacher Update (처방)

OPSD에서 teacher는 student의 특정 시점 snapshot입니다. 따라서 teacher의 capability는 고정 상수가 아니라 **선택 가능한 변수**입니다. curriculum은 stage 단위로 진행되므로, **각 stage 학습 종료 시 그 checkpoint를 다음 stage의 teacher 및 student 초기값으로 update**할 수 있습니다.

$$\text{stage } k \text{ 종료} \;\Rightarrow\; \text{checkpoint} = \text{stage } k{+}1 \text{ 의 teacher} + \text{student 초기값}$$

stage가 어려워질수록 teacher capability도 함께 상승하므로, 각 stage에서 teacher capability가 stage difficulty에 근접한 상태로 유지됩니다. 고난도 구간까지 rationalizability가 유지될 가능성이 생깁니다.

이 설계가 OPSD에 고유한 이유는 명확합니다 — "teacher를 stage별로 강화한다"는 *teacher가 reference solution을 보는* OPSD에서만 성립합니다. supervised fine-tuning에는 teacher가 없고, GRPO에는 answer-conditioned teacher가 없습니다.

risk도 명시합니다. teacher를 student로 반복 update하면 self-distillation이 iterative self-improvement로 성격이 전환되며, 한 stage의 error가 다음 teacher로 누적되는 collapse가 발생할 수 있습니다. 따라서 **update 시점과 빈도**가 별도의 research question입니다.

| 비교 | 측정 대상 |
| --- | --- |
| teacher 고정, difficulty-only vs subject-aware | representation 순서 효과 (진행 중) |
| teacher update, subject-aware vs teacher 고정, subject-aware | teacher update 이득 (핵심) |
| teacher update 시 subject-aware vs difficulty-only | 강화된 teacher 하에서의 순서 효과 |
| 위 항목을 고난도 stage에서 | OPSD limitation 해결 직접 증거 |

### 4-3. Idea 2 — Representation predicts Rationalizability (진단)

representation은 rationalizability의 원인이 아닙니다. rationalizability를 결정하는 것은 **teacher capability 대비 problem의 difficulty·domain 위치**이며, **representation은 그 위치를 측정하는 coordinate system**입니다. 인과 구조는 다음과 같습니다.

> representation에서 (difficulty·domain) 위치를 추정 → 그 위치가 teacher 대비 rationalizability를 결정 → 위치를 사전에 읽어 stage를 설계.

세 hypothesis로 분해합니다.

- **H1 (측정).** representation 위치로 stage별 rationalizability를 예측한다. (3-2가 1단계 — representation이 familiarity를 예측함을 확인.)
- **H2 (설명).** stage별 rationalizability가 그 stage의 학습 이득(checkpoint curve의 성능 증가분)을 예측한다.
- **H3 (처방).** rationalizability가 유지·상승하도록 stage를 배치하면(= 본 curriculum) 학습 이득이 증가한다.

핵심 측정 도구는 **checkpoint curve**입니다. 각 stage boundary에서 checkpoint를 저장해 개별 평가하면, stage별 성능 증가분을 직접 측정합니다. 이 curve는 H2의 종속변수이자 Idea 1의 update 지점이므로, 하나의 측정값이 두 idea를 동시에 지원합니다.

### 4-4. 통합

- **Idea 2 = 진단** — representation으로 rationalizability를 추정하고, 그것이 학습 이득을 설명함을 검증합니다.
- **Idea 1 = 처방** — rationalizability limitation을 teacher update로 완화합니다.
- 두 idea의 공통 종속변수는 **stage별 학습 이득(checkpoint curve)**입니다.

이 구조에서 narrative의 중심은 curriculum이 아니라 **rationalizability**가 되고, curriculum(특히 teacher update)은 rationalizability limitation에 대한 처방으로 위치합니다. 이것이 "왜 OPSD인가"에 대한 답입니다.

추가로, 수정 설계(subjslack)는 인접 stage 사이의 representational jump를 감소시켰습니다. jump가 작으면 update된 teacher가 representation 상 인접한 다음 stage만 담당하므로, teacher update의 collapse risk가 감소합니다. representation-aware 순서와 OPSD teacher update는 상호 보완 관계입니다.

---

## 5. 향후 계획

| 순서 | 내용 | 목적 |
| --- | --- | --- |
| 1 | 수정 설계(subjslack) full 학습 + checkpoint curve | AIME 손해 해소 여부, stage별 학습 이득 (H2) |
| 2 | difficulty-matched random(cond5) 대비 비교 | subject 효과가 difficulty 너머인지 확정 (H3) |
| 3 | teacher answer-conditioned rationalizability 측정 | H1 완성 |
| 4 | stage-wise teacher update 적용 + update 빈도 비교 | Idea 1 검증, main 승격 판단 |

운영 원칙입니다. 학습 step 수는 OPSD 논문 수치를 차용하지 않고 **checkpoint curve로 직접 결정합니다.** step 수 조절은 data를 줄이지 않고 max step으로만 합니다(조건 간 형평성). 평가는 non-thinking으로 수행합니다(고성능 model은 thinking 평가에서 ceiling effect로 조건 간 차이가 축소됩니다).
