## 0. 실험 결과 요약 (Executive Summary)

<aside>
💡

difficulty 기반 실험과 비교하여, **제 메인 실험인** **unit(levelxsubject) 유사도 기반 OPSD 커리큘럼 러닝의 결과가 네 benchmark(AIME 2024·2025, HMMT 2025, MATH-500) 모두에서 difficulty-only baseline보다 평균 +4.1%p 높은 성능을 보이는 것으로 나왔습니다.**

| Benchmark | difficulty-only | unit-similarity (ours) | Δ |
| --- | --- | --- | --- |
| AIME 2024 | 53.3 | 56.7 | +3.3 |
| AIME 2025 | 36.7 | 43.3 | +6.7 |
| HMMT 2025 | 27.8 | 28.9 | +1.1 |
| MATH-500 | 79.6 | 85.0 | +5.4 |
| **평균** | **49.3** | **53.5** | **+4.1** |

*full training 최종 시점(step 900), avg@n (%). AIME·HMMT는 val_n=3, MATH-500은 val_n=1, non-thinking(OPSD 논문 eval 기본 설정도 non-thinking).*

이로써 "difficulty 외에 representation geometry에서 유도한 axis로 curriculum을 개선할 수 있다"는 가설을 1차 검증했습니다. 

다만 이 이득을 subject도 커리큘럼 구성 기준에 포함시킨 영향으로 보려면 control(difficulty-matched random) 실험이 필요하고, 현재 training을 기다리고 있습니다.

또한 성능차가 아직 크지 않아 curriculum을 더 보완할 여지가 남아 있습니다.

</aside>

---

## 1. 연구 방향

<aside>
💡

현재 연구는 LLM 수학 추론을 OPSD(On-Policy Self-Distillation)로 학습할 때, 학습 순서를 정하는 curriculum을 어떻게 구성할지를 다룹니다. curriculum의 가장 자연스러운 기준은 난이도이지만, 현재 연구는 여기에 모델 자신의 internal representation에서 유도한 두 번째 축을 더합니다. 구체적으로 문제를 unit(subject×level) 단위의 representation 유사도로 묶어 학습 순서를 정합니다.

이 방향을 representation-guided curriculum이라 부르고, 난이도만 쓰는 difficulty-only baseline과 비교해 그 효과를 확인하는 것이 현재 연구의 목표입니다.

</aside>

---

## 2. Method

### 2-1. 설정과 표기, difficulty baseline

문제를 $x$, 난이도를 $\ell(x)\in{1,\dots,8}$, 과목을 $s(x)$라 하고, 둘을 묶은 단위를 unit $u=(s,\ell)$로 둡니다. unit에 속한 문제 수는 $n_u$입니다. curriculum은 전체 문제를 $K$개 stage로 나눈 순서이고, 학습은 stage 1부터 차례로 진행합니다($K=5$).

가장 단순한 기준은 난이도만 쓰는 difficulty baseline(diff)입니다.

$\text{score}_{\text{diff}}(x) = \ell(x)$

문제를 $\ell$ 오름차순으로 정렬해 같은 질량으로 $K$등분합니다. 이 기준으로는 쉬운→어려운 단조 순서를 얻지만, 같은 난이도 안의 과목은 구분되지 않습니다.

### 2-2. 공통 재료 — 난이도와 분리된 representation 축

두 방법에서 같은 representation 재료를 씁니다. 각 문제의 reasoning 중 neuron activation shift를 **thinking 모드로 추론한 probe 문제 집합(N = 3,025)에서** 추출하고(30K 전체 학습셋이 아님), 거기서 만든 unit centroid를 전체 unit에 적용합니다. 난이도 성분을 빼기 위해 같은 level 평균을 제거합니다.

$\tilde a_x = a_x - \mu_{\ell(x)}, \qquad \mu_{\ell} = \frac{1}{|{x:\ell(x)=\ell}|}\sum_{x:\ell(x)=\ell} a_x$

- $a_x$ : 문제 $x$의 activation shift 벡터
- $\mu_\ell$ : level $\ell$ 문제들의 평균 (난이도 성분)
- $\tilde a_x$ : 난이도를 제거한 잔차 — 난이도와 직교한 과목 표현

unit(subject와 level의 조합)별 평균을 정규화해 centroid $c_u$를 얻고, unit 사이 cosine 유사도 행렬 $M$을 만듭니다.

$c_u = \frac{\bar{\tilde a}_u}{\lVert \bar{\tilde a}u\rVert}, \qquad M{uv} = \langle c_u, c_v\rangle$

- $\bar{\tilde a}_u$ : unit $u$에 속한 문제들의 $\tilde a_x$ 평균
- $M_{uv}$ : unit $u$와 $v$의 표현 유사도 (1에 가까울수록 유사)

$M$에는 난이도를 제거한 뒤의 과목 기하(subject geometry)가 담겨 있습니다.

### 2-3. 첫 시도 — difficulty backbone + within-tier 재배치 (tiered)

처음에는 난이도를 **큰 구간(tier) 단위로만 고정**하고, 각 구간 안에서는 representation으로 순서를 섞었습니다. 절차는 세 단계입니다.

(1) unit을 $\ell$로 정렬해 같은 질량의 $T$개 **tier**로 나눕니다. **여기서 tier는 난이도로 크게 묶은 덩어리로, 최종적으로 쓴 $T=2$에서는 쉬운 절반(tier 1)과 어려운 절반(tier 2) 두 개가 됩니다.**

(2) 각 tier 안에서 cosine 거리 $d$가 가까운 unit을 잇는 greedy nearest path로 순서를 정합니다. 이 경로는 난이도 순서를 지키지 않습니다.

$d(u,v) = 1 - \langle c_u, c_v\rangle, \qquad u_{(i+1)} = \arg\min_{v \in \text{tier}} d(u_{(i)}, v)$

- $d(u,v)$ : 두 unit의 표현 거리 (가까울수록 작음)
- $u_{(i+1)}$ : 직전 unit과 표현이 가장 가까운 다음 unit

(3) tier를 쉬운 쪽에서 어려운 쪽으로 이어 붙여 전체 순서를 만들고, 그 순서를 같은 질량으로 $K=5$개 stage로 자릅니다.

즉 난이도 순서는 **두 tier 사이에서만** 보장되고(쉬운 절반 → 어려운 절반), 각 tier 안에서는 representation 순서를 따릅니다.

### 2-4. 첫 시도가 부족했던 이유

stage($K=5$)가 tier($T=2$)보다 잘다는 점이 문제였습니다. tier 하나가 여러 level을 통째로 덮는데, 그 안의 순서는 난이도가 아니라 representation 경로입니다. 그래서 여러 stage가 한 tier 안에 들어가면, stage 경계는 난이도가 아니라 **tier 내부의 representation 경로상 위치**를 따라 정해집니다. 이 경로를 따라가면 **난이도가 위아래로 진동**하므로, 한 stage가 여러 level에 걸쳐집니다.

그 결과 stage 평균 난이도의 단조성이 깨졌고, stage 내 난이도 분산이 diff의 약 12배로 커졌습니다(전 stage 평균 level 분산 1.31 대 0.11). 쉬운 난이도(level 2–3)를 집중적으로 다지는 stage가 사라졌고, 이것이 성능 저하로 이어졌습니다.

### 2-5. 메인 방법 — score 기반 2축 정렬

**원인이 "representation 경로가 난이도를 흔든다"에 있었으므로, 메인 방법에서는 난이도를 1차 정렬 키로 고정하고 과목을 그 위에 더하는 작은 보정으로 바꿨습니다.**

먼저 과목의 representation 좌표 $z(s)$를 만듭니다. unit 유사도 행렬 $M$을 이중 중심화한 뒤 leading 축(classical MDS)을 취합니다.

$B = JMJ,\quad J = I - \tfrac{1}{m}\mathbf{1}\mathbf{1}^\top,\qquad B = \sum_i \lambda_i v_i v_i^\top$

$z(s) = \text{scale}!\Big(\operatorname{mean}{u:,\text{subj}(u)=s}\ \sqrt{\lambda{\max}},[v_{\max}]_u\Big) \in [-\tfrac12, \tfrac12]$

- $B$ : 유사도 행렬 $M$을 이중 중심화한 행렬
- $\lambda_{\max}, v_{\max}$ : 가장 큰 고윳값과 고유벡터 — 과목 기하의 주축
- $z(s)$ : 과목 $s$의 좌표를 그 과목 unit들에 대해 평균내고 $[-\tfrac12,\tfrac12]$로 스케일한 값 (부호는 한 방향으로 고정)

문제별 score는 난이도에 이 좌표를 더해 정의합니다.

$\boxed{\ \text{score}_{\text{ours}}(x) = \ell(x) + \alpha \cdot z(s(x))\ }, \qquad \alpha = 2$

- $\ell(x)$ : 난이도 backbone (1차 정렬 키)
- $z(s(x))$ : 과목 보정 — 난이도와 직교한 representation 좌표
- $\alpha$ : 보정 세기. $z\in[-\tfrac12,\tfrac12]$이므로 보정 폭이 $|\alpha z|\le \alpha/2 = 1$ level로 제한됩니다

보정 폭이 1 level을 넘지 않아 난이도 단조는 유지되고, 같은 난이도대 안에서만 과목이 representation 순서로 앞뒤로 움직입니다.

마지막으로 unit을 score 순으로 정렬하고, unit을 쪼개지 않은 채 누적 질량이 $kN/K$에 가장 가까운 경계에서 잘라 $K$개 stage를 만듭니다.

$\text{cut}k = \arg\min{j}\Big|\textstyle\sum_{i \le j} n_{u_i} - \tfrac{kN}{K}\Big|, \qquad k = 1,\dots,K-1$

- $N$ : 전체 문제 수
- $\text{cut}_k$ : $k$번째 stage 경계 (질량 균형점)

### 2-6. 비교군 (control)

성능 차이를 representation 기반 과목 정렬에 귀속시키려면, 난이도 분포는 같고 과목 순서만 다른 비교군이 필요합니다. 이를 위해 난이도-정합 무작위 비교군(difficulty-matched random, 이하 random control)을 둡니다. 

이 비교군에서는 ours의 (level, stage)별 문제 수를 그대로 유지하고, 각 level 안에서 어느 문제가 어느 stage로 갈지는 무작위로 정합니다. ours가 random control을 이기면, 그 이득은 난이도 스케줄이 아니라 representation 기반 과목 정렬에서 온 것입니다.

---

## 3. Experimental Setup

### 3-1. 데이터 & 모델

subject와 level이 라벨링된 수학 문제 $N = 28{,}771$개를 학습셋으로 씁니다(7 subjects × 8 levels). representation 축은 §2-2대로 thinking 모드로 추론한 probe 문제 집합($N = 3{,}025$)에서 추출합니다. 모델은 Qwen3-8B이고 LoRA로 학습합니다(rank 64, $\alpha$ 128, target projection: q·k·v·o·gate·up·down).

diff, ours, random control 세 실험은 **같은 문제 집합**을 공유하며 stage 배정만 다릅니다(universe 동일성 확인 완료).

### 3-2. 커리큘럼 구성 (diff vs ours)

§2-5의 레시피로 실제로 나온 구성을 세 단계로 봅니다 — 먼저 난이도 구조, 다음 과목 구성, 마지막으로 과목을 cluster로 묶은 요약입니다.

**(A) stage별 난이도 구조**

| stage | diff 평균L (범위) | diff 문제수 | ours 평균L (범위) | ours 문제수 |
| --- | --- | --- | --- | --- |
| 1 | 1.76 (L1–2) | 5,549 | 1.90 (L1–3) | 5,323 |
| 2 | 3.00 (L3) | 5,790 | 3.04 (L2–4) | 5,365 |
| 3 | 4.00 (L4) | 6,091 | 3.90 (L3–5) | 6,171 |
| 4 | 5.00 (L5) | 5,470 | 4.89 (L4–6) | 5,775 |
| 5 | 6.22 (L5–8) | 5,871 | 5.97 (L5–8) | 6,137 |

diff는 stage마다 단일 level 밴드(L3, L4, L5 …)로 난이도를 좁게 끊습니다. ours는 인접 level이 겹치는 ~3 level 밴드이되, 평균 난이도는 똑같이 단조 증가합니다. ours가 밴드 경계를 의도적으로 겹친 이유는, 그 겹친 구간 안에서 과목을 representation 순서로 재배치할 여지를 만들기 위해서입니다.

**(B) stage별 subject 구성** — 문제수와 (stage 내 비율)

*diff:*

| stage | Alg | IntAlg | Precalc | Geo | Prealg | NumTh | C&P |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,355 (24%) | 101 (2%) | 151 (3%) | 393 (7%) | 2,348 (42%) | 534 (10%) | 667 (12%) |
| 2 | 1,805 (31%) | 231 (4%) | 549 (9%) | 797 (14%) | 538 (9%) | 909 (16%) | 961 (17%) |
| 3 | 1,724 (28%) | 464 (8%) | 755 (12%) | 1,317 (22%) | 17 (0%) | 1,101 (18%) | 713 (12%) |
| 4 | 1,365 (25%) | 725 (13%) | 0 (0%) | 1,389 (25%) | 0 (0%) | 1,271 (23%) | 720 (13%) |
| 5 | 574 (10%) | 768 (13%) | 740 (13%) | 1,370 (23%) | 0 (0%) | 1,621 (28%) | 798 (14%) |

*ours:*

| stage | Alg | IntAlg | Precalc | Geo | Prealg | NumTh | C&P |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,355 (25%) | 332 (6%) | 700 (13%) | 393 (7%) | 2,348 (44%) | 89 (2%) | 106 (2%) |
| 2 | 1,805 (34%) | 464 (9%) | 755 (14%) | 797 (15%) | 538 (10%) | 445 (8%) | 561 (10%) |
| 3 | 1,724 (28%) | 725 (12%) | 518 (8%) | 1,317 (21%) | 17 (0%) | 909 (15%) | 961 (16%) |
| 4 | 1,860 (32%) | 517 (9%) | 195 (3%) | 1,389 (24%) | 0 (0%) | 1,101 (19%) | 713 (12%) |
| 5 | 79 (1%) | 251 (4%) | 27 (0%) | 1,370 (22%) | 0 (0%) | 2,892 (47%) | 1,518 (25%) |

두 실험은 두 가지 면에서 다릅니다. 첫째, stage 평균 난이도는 거의 같지만(A) ours의 level 밴드가 더 넓게 겹쳐, 난이도 밴드의 모양이 다릅니다. 둘째, 같은 난이도대 안에서 과목 구성이 다릅니다(B). 따라서 과목 효과만 따로 떼어내는 비교는 diff가 아니라, 난이도 분포를 ours와 정확히 맞추고 과목만 무작위화한 random control(§2-6)이 담당합니다.

 Precalculus는 stage 1에서 diff 3% → ours 13%로 당겨지고, Number Theory는 stage 1에서 diff 10% → ours 2%로 빠지는 대신 stage 5에서 28% → 47%로 몰립니다.

**(C) cluster로 압축한 요약**

representation 좌표 $z(s)$를 기준으로 7개 과목은 세 무리로 나뉩니다.

- **algebra계** (Algebra, Intermediate Algebra, Precalculus) — $z<0$, 이른 배치 성향
- **geometry** (Geometry) — $z\approx0$, 중립
- **discrete계** (Prealgebra, Number Theory, Counting & Probability) — $z>0$, 늦은 배치 성향

각 cluster가 평균적으로 몇 번째 stage에 놓이는지로 이동을 한눈에 볼 수 있습니다(작을수록 이른 stage).

| cluster | diff 평균 stage | ours 평균 stage | 이동 |
| --- | --- | --- | --- |
| algebra계 | 3.04 | 2.60 | **−0.44 (앞으로)** |
| geometry | 3.48 | 3.48 | **±0.00 (불변)** |
| discrete계 | 2.78 | 3.33 | **+0.55 (뒤로)** |

ours에서 algebra계는 더 이른 stage로, discrete계는 더 늦은 stage로 이동하고, 중립인 geometry는 정확히 그대로입니다. 이 세 방향이 $z(s)$의 부호와 일치하며, representation 좌표가 실제 stage 배치를 그대로 만들어냈음을 보여줍니다.

한 가지 단서: discrete계의 Prealgebra는 문제가 거의 모두 쉬운 난이도라, 늦은 배치 성향에도 불구하고 난이도에 묶여 이른 stage에 남습니다. 그래서 stage 1의 discrete 비중은 여전히 높지만, level이 넓게 퍼진 Number Theory·Counting이 늦은 stage로 이동하면서 cluster 평균 stage는 뒤로 밀립니다.

**(D) 두 실험 구성의 통계적 차이**

| 항목 | 값 | 의미 |
| --- | --- | --- |
| Spearman ρ(diff, ours) stage 순위 | 0.889 | 전체 학습 순서는 강하게 유지됨 |
| stage가 바뀐 문제 비율 | 38.7% | 1/3 이상이 다른 stage로 이동 |
| 평균 |Δstage| | 0.405 | 이동 폭은 평균 0.4 stage |
| per-stage level 분산 (diff / ours) | 0.11 / 0.49 | ours 밴드가 더 넓음 — 위 "모양 차이"의 수치 |
| 과목 재배치 유의성 (ours vs random control) | permutation **p = 0.005** | 난이도를 고정해도 ours의 과목 배치는 무작위와 유의하게 다름 |

> 표 해석: 두 실험은 큰 순서(ρ = 0.89)는 공유하되 1/3 이상의 문제가 stage를 바꿉니다. 그 이동은 난이도 밴드를 넓히는 변화(분산 0.11 → 0.49)와 과목 재배치로 나뉘고, 과목 재배치는 난이도를 고정한 검정에서 무작위 대비 유의합니다(p = 0.005).
> 

### 3-3. 학습 설정

| 항목 | 값 |
| --- | --- |
| 모델 | Qwen3-8B |
| 학습 방식 | LoRA (rank 64, $\alpha$ 128, target q·k·v·o·gate·up·down) |
| 하드웨어 | H200 × 2 |
| batch 구성 | per-device 2 × grad accum 8 × world size 2 = **global 32** |
| learning rate | $5\times10^{-6}$ |
| max grad norm | 0.1 |
| 학습량 | 1 epoch ≈ 900 step |
| max completion length | 1,024 |
| max length | 20,000 |
| OPSD loss | JSD ($\beta = 0$, $\lambda = 1$) |
| JSD token clip | 0.06 |
| teacher | base 모델 고정 (fixed teacher), **thinking** 모드 |
| student | **non-thinking** 모드 |
| rollout sampling | temperature 1.1, top-p 0.95, top-k 20 |
| gradient checkpointing | on |
| checkpoint | 50 step마다 저장 (총 18개) |

### 3-4. 평가 설정

| 항목 | 값 |
| --- | --- |
| benchmark | AIME 2024, AIME 2025, HMMT 2025, MATH-500 |
| 추론 모드 | non-thinking |
| temperature | 1.0 |
| 지표 | avg@n, pass@n |
| val_n — step별 곡선 | AIME·HMMT 3, MATH-500 1 (빠른 비교용) |
| val_n — 최종 보고 | **12** (원 OPSD 논문과 동일) |

---

## 4. Results

### 4-1. 최종 성능 (full training, step 900)

§2-5의 메인 방법(ours)과 difficulty-only baseline(diff)을 full training 최종 시점에서 비교합니다(avg@n, %).

| Benchmark | diff | ours | Δ |
| --- | --- | --- | --- |
| AIME 2024 | 53.3 | 56.7 | +3.3 |
| AIME 2025 | 36.7 | 43.3 | +6.7 |
| HMMT 2025 | 27.8 | 28.9 | +1.1 |
| MATH-500 | 79.6 | 85.0 | +5.4 |
| **평균** | **49.3** | **53.5** | **+4.1** |

ours가 네 benchmark에서 diff를 앞서며, 특히 AIME 2025(+6.7)와 MATH-500(+5.4)에서 큽니다. 다만 HMMT 2025의 +1.1은 val_n=3의 노이즈 범위에 들어 단정하기 이르고, 진행 중인 pass@12 재측정(§4-3)으로 확정합니다. 어려운 문제를 더 풀게 하려는 연구 목표에 비추어, 난이도 외에 representation 축을 더한 구성이 어려운 구간에서 도움이 된다는 1차 근거입니다.

### 4-2. 학습 경과에 따른 성능 (step-curve)

50 step마다 저장한 checkpoint로 학습 경과를 봅니다. step은 stage 진행과 다음과 같이 대응합니다 — **step 100 ≈ stage 1(쉬움), step 400 ≈ stage 3(중간 난이도), step 650 ≈ stage 4, step 900 = stage 5 완료(전체)**.

| Benchmark | step 100 | step 400 | step 650 | step 900 |
| --- | --- | --- | --- | --- |
| AIME 2024 | 30.0 / 30.0 | 61.1 / 54.4 | 54.4 / 53.3 | 56.7 / 53.3 |
| AIME 2025 | 26.7 / 32.2 | 51.1 / 44.4 | 47.8 / 48.9 | 43.3 / 36.7 |
| HMMT 2025 | 15.6 / 13.3 | 31.1 / 32.2 | 33.3 / 27.8 | 28.9 / 27.8 |
| MATH-500 | 84.6 / 83.0 | 86.4 / 84.8 | 85.0 / 82.4 | 85.0 / 79.6 |

*각 칸은 ours / diff (avg@n, %).*

두 가지가 드러납니다.

**첫째, ours의 우위는 쉬운 구간이 아니라 중간 난이도 구간에서 생깁니다.** 어려운 세 benchmark(AIME·HMMT)의 평균 격차를 보면, stage 1만 학습한 step 100에서는 ours가 앞서지 않습니다(24.1 대 25.2, −1.1). 중간 난이도 stage까지 학습한 step 400에서 격차가 벌어지고(47.8 대 43.7, +4.1), 이후 끝까지 유지됩니다(step 900에서 +3.7). 같은 난이도대 안에서 과목을 representation 순서로 재배치한 효과가, 그 재배치가 일어나는 중간 stage 학습에서 나타나는 것으로 해석됩니다.

**둘째, 최종 stage에서 diff는 MATH-500 성능이 떨어지지만 ours는 유지됩니다.** 가장 어려운 stage 5를 학습하는 step 650 → 900 구간에서 diff는 82.4 → 79.6으로 내려가는 반면, ours는 85.0으로 평평합니다. 난이도 밴드를 넓게 겹쳐 stage 전이를 완만하게 만든 구성이, 어려운 구간 학습 중 쉬운 문제 성능이 깎이는 것을 줄인 것으로 보입니다.

지표의 잡음은 감안해야 합니다. AIME·HMMT는 문제 30개에 val_n = 3(시도 90회)이라 단일 step 값의 변동이 ±5%p 정도입니다. 따라서 step 650의 일시적 하락보다는, step 400과 step 900에서 일관되게 나타나는 양의 격차를 신호로 봅니다.

### 4-3. 진행 중 / 예정

| 항목 | 목적 | 상태 |
| --- | --- | --- |
| pass@12 재측정 (최종 step) | 원 OPSD 논문과 동일한 val_n = 12로 최종 수치를 견고화 | 진행 중 |
| random control 학습·평가 | ours가 난이도-정합 무작위를 이기는지 확인 → 과목 효과 격리 | 학습 대기 |

pass@12 결과가 나오면 §4-1 수치를 그것으로 교체하고, random control 결과로 "이득이 과목 정렬에서 온다"는 주장을 확정할 계획입니다.

---

## 5. Motivation 보강을 위한 추가 method 제안 
— **representation-guided curriculum이 OPSD와 결합되어야 하는 이유**

### 5-1. 교수님 피드백 - 새로 생각해본 아이디어 (아직 이 부분은 실험 없음)

<aside>
💡

저번 미팅에서 제 연구의 **"왜 representation-guided curriculum이 하필 OPSD와 결합되어야 하는가"를 설득하는 부분이 부족하고 motivation이 약하다는 피드백**을 받았습니다. 

**현재 연구의 주 method는 "representation이 비슷한 문제끼리 모아 매끄러운 순서로 학습하여 추론 능력을 축적시킨다"입니다.**

 그런데 이렇게 설명하면 SFT나 GRPO를 두고도 똑같이 말할 수 있습니다. 다시 말해 **OPSD를 다른 학습법으로 바꿔 끼워도 같은 이야기가 되어 OPSD가 꼭 필요한 이유를 대지 못합니다.** 여기에 motivation의 약점이 있습니다.

따라서 이 문제를 해결하기 위해, 현 연구의 ordering axis가 OPSD에서만 특별한 의미를 갖는다는 점을 보여야 한다고 생각했습니다. 아래 두 idea로 답하겠습니다 — idea 2는 진단, idea 1은 처방입니다.

</aside>

### 5-2. OPSD의 핵심 (전제)

> OPSD에서는 한 모델이 teacher와 student 역할을 맡습니다. student는 문제만 보고 rollout을 생성하고, teacher는 같은 모델이 reference solution을 본 상태로 그 토큰들의 분포를 다시 계산합니다. 둘 사이의 JSD를 loss로 씁니다. 여기서 teacher는 정답을 아는 상태로 풀이를 rationalize해 주는 역할을 맡습니다.
> 

> 그래서 teacher가 그 문제를 얼마나 잘 rationalize하느냐에 따라 student가 받는 training signal의 질이 달라집니다. **이런 "rationalizability"는 SFT(고정 target)나 GRPO(scalar reward)에서는 찾을 수 없고, OPSD에서만 다룰 수 있습니다.**
> 

### 5-3. **Idea 1 — Stage-wise Teacher Upgrade**

**각 stage를 마친 student가 다음 stage의 teacher로 승격(promote)**

<aside>
💡

**표준 OPSD에서는 teacher를 base 모델로 고정**합니다. 그런데 **base 모델은 어려운 문제일수록 rationalize를 잘 못해서 정작 hard 구간에서 signal이 약해집니다.** 

**현재 연구에서 Stage-wise Teacher Upgrade curriculum으로 이를 해결할 수 있습니다.** 어떤 stage를 마치고 나면 모델은 그 구간에서 이미 성능이 향상돼 있으니, 이 **향상된 student를 다음 stage의 teacher로 promote**하면 됩니다. 그러면 base teacher가 가장 약했던 hard 구간에서 오히려 가장 강해진 teacher로 signal을 줄 수 있을 것입니다.

이 방식은 OPSD와 curriculum을 둘 다 갖춰야만 성립합니다. teacher가 없는 SFT·GRPO에서는 "teacher를 갱신한다"를 말할 수 없고, stage가 없으면 "언제 갱신할지"를 정할 수 없습니다. 

</aside>

### 5-4. Idea 2 — Rationalizability Alignment

<aside>
💡

현재 연구는 문제를 unit(subject×level) 단위의 representation 유사도로 묶어 curriculum 순서를 정합니다. 여기서 저는 **모델이 어떤 문제를 internal representation에서 어떻게 인식하는지가, 그 풀이를 얼마나 자연스럽게 받아들이고 rationalize할 수 있는지와 맞닿아 있다고 생각합니다.**

이렇게 생각한 근거는 이렇습니다. **(unit 구분에 쓰는 neuron activation shift가 나타내는) representation에는 모델이 문제를 어떤 방식으로 이해하는지가 담겨 있습니다.** **representation이 가까운 문제들을 모델은 비슷한 방식으로 다루고, representation이 먼 문제들은 다르게 다룹니다.** 

> **teacher가 reference solution을 rationalize하는 일도 결국 그 문제를 어떻게 이해하느냐에 달려 있습니다. 그렇다면 representation을 기준으로 정렬한 unit 순서를, teacher가 각 문제를 얼마나 잘 rationalize하는가(rationalizability)의 순서와 자연스럽게 이어 볼 수 있습니다.**
> 

> **이 연결이 성립한다면, unit으로 정렬한다는 것은 곧 teacher가 잘 rationalize하는 문제부터 어려운 문제 순으로 학습한다는 뜻이 됩니다. 그리고 rationalizability는 reference solution을 보는 teacher가 있어야만 정의되는, OPSD 고유의 양입니다. 따라서 unit 기반 정렬은 OPSD와 결합될 때 비로소 "teacher가 잘 가르칠 수 있는 문제부터 배운다"는 의미를 갖습니다. teacher rationalization이 없는 SFT나 GRPO에서는 같은 정렬을 해도 이런 의미가 생기지 않습니다.**
> 

이것이 아이디어 2의 핵심입니다. unit(subject×level) representation은 단순한 "비슷한 문제 묶기"가 아니라, **OPSD의 training signal인 teacher rationalizability와 이어진 정렬 기준으로 해석할 수 있습니다. 바로 이 해석을 근거로, representation-guided curriculum을 다른 학습법이 아닌 OPSD와 묶을 수 있습니다.**

</aside>

### 5-5. 종합

<aside>
💡

representation-guided curriculum을 다른 학습법이 아닌 OPSD와 결합해야 하는 이유는 두 가지입니다.

1. **ordering 기준이 OPSD와 이어져 있습니다 (idea 2).** unit(representation) 순서는 teacher rationalizability 순서와 맞닿아 있고, rationalizability는 OPSD의 teacher에게만 있습니다.
2. **teacher를 단계마다 강화하는 일은 OPSD와 curriculum이 둘 다 있어야 가능합니다 (idea 1).** 강화할 teacher는 OPSD에, 강화 시점을 정하는 stage는 curriculum에 있습니다.

OPSD를 SFT·GRPO로 바꾸면 teacher rationalization이 사라져 두 이유 모두 무너집니다.

</aside>

---

## 6. Limitations & Plan

### 6-1. 한계

- **과목 효과가 아직 격리되지 않았습니다.** 현재 diff 대비 우위는 두 요인이 섞여 있습니다 — 난이도 밴드를 넓게 겹친 변화(§3-2 A)와 과목 재배치(§3-2 B). 과목만의 기여는 random control 비교로만 확정되는데, 그 학습이 아직 끝나지 않았습니다.
- **성능차가 크지 않고 일부는 잠정입니다.** 평균 +4.1%p이며 HMMT는 +1.1로 작습니다. step별 측정이 val_n=3/1이라 ±5%p 수준의 노이즈가 있어, pass@12 재측정 전까지 일부 수치는 확정 전입니다.
- **설정이 단일합니다.** Qwen3-8B 한 모델, 한 학습셋에서 얻은 결과이고, random control도 seed 하나만 예정입니다. 다른 모델·스케일·seed로의 일반화는 확인하지 않았습니다.
- **§5의 motivation은 제안 단계입니다.** idea 1·2는 논리적 제안이며, 아직 학습이나 측정으로 뒷받침하지 않았습니다.

### 6-2. 다음 단계

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| pass@12 확정 | 최종 step 수치를 원 논문 설정으로 견고화하고 §0·§4-1 갱신 | 진행 중 |
| random control | ours가 난이도-정합 무작위를 이기는지 확인 → 과목 효과 격리 (필요 시 seed 추가) | 학습 대기 |
| curriculum 보완 | $\alpha$·stage 수·밴드 겹침 폭을 조정해 성능차를 키울 여지 탐색 | 예정 |
| idea 1 구현 | stage-wise teacher upgrade를 실제 학습에 적용·평가 | 조건부 (우위 확정 후) |
| idea 2 검증 | representation 축과 rationalizability의 연결을 정량적으로 측정 | 조건부 |
