# Representation-Guided Curriculum: Difficulty Cliff × Subject Geometry

**Research Note — 2026-07-01**
Track: OPSD_Curriculum · Model: Qwen3-4B · Universe: clean MATH (N = 28,743)

---

## Abstract

We construct a reinforcement-distillation curriculum whose stage structure is derived
from the *internal representation geometry* of the base model rather than from human
labels alone. Two orthogonal axes are read off from hidden-state activations: (i) a
**difficulty axis** exhibiting a two-cluster "cliff" at level 4, which we traverse with
overlapping sliding windows (`cliff`), and (ii) a **subject axis** obtained by a 1-D
metric embedding of subject centroids, which we align to co-move with difficulty
(`cliff_subjgeo`). Generation budget is ramped per stage in proportion to difficulty
(*context scaling*). The design is paired with a difficulty-matched random-subject
control (`cliff_subjrand`) so that the subject axis is tested strictly *beyond*
difficulty. The optimization objective (OPSD on-policy distillation) is inherited
unchanged from the original implementation; the curriculum is a data-ordering and
generation-length intervention only.

---

## 1. Motivation

An earlier level×subject curriculum (`ours` / subjslack, 8B) beat a difficulty-only
baseline (`diff`) but **did not** beat a difficulty-matched random-subject control
(`cond5`) on a single seed, suggesting the gain came from *overlapping mixed-level
bands* rather than subject order. Replicating the control across seeds widened its
spread enough that the original "subject is null" verdict became **underpowered rather
than settled** (the control's seed-to-seed range on 30-problem AIME benchmarks is
comparable to the treatment–control gap). We therefore (a) keep the representation-derived
difficulty structure as the backbone, (b) strengthen the subject intervention and align
it to difficulty, and (c) pair it with a matched control replicated across seeds, so a
subject effect *beyond difficulty*, if it exists at a detectable size, can be measured.

---

## 2. Setup and notation

Let $\mathcal{U}$ be the clean training universe, $|\mathcal{U}| = N = 28{,}743$. Each
problem $p$ carries a human difficulty level $\ell(p)\in\{1,\dots,8\}$ and a subject
label $s(p)\in\mathcal{S}$, $|\mathcal{S}| = 7$
($\mathcal{S}=\{$Algebra, Intermediate Algebra, Precalculus, Geometry, Prealgebra,
Number Theory, Counting & Probability$\}$).

A curriculum is a partition of $\mathcal{U}$ into ordered stages
$\Pi = (\mathcal{P}_0,\dots,\mathcal{P}_{K-1})$, $K=6$, with
$\bigsqcup_k \mathcal{P}_k = \mathcal{U}$ (each problem appears exactly once). Training
visits stages in index order; within a stage the order is a deterministic shuffle.
All arms share $\Pi$'s **marginals over level** and differ only in the axis under study,
so that comparisons are compute- and exposure-matched.

---

## 3. Difficulty backbone: the activation cliff

### 3.1 Empirical geometry

On pooled thinking-trace activations (per-layer centered $\Delta A$, $N=3025$), level
centroids form a **monotone difficulty axis**: the ordinality between centroid cosine
and $-|\Delta\ell|$ is $\rho = +0.84$ (L1–L8). The similarity matrix reveals two
clusters — easy $\{1,2,3\}$ and hard $\{5,6,7,8\}$ — separated by a **transition band at
$\ell = 4$** (L4 correlates only weakly with both sides: L4–L3 $=+0.42$, L4–L5 $=+0.47$,
else $\approx 0$). This cliff is model-invariant (8B and 4B agree, $r = 0.996$). It
motivates *overlapping* stages that carry the learner smoothly across the L3–L4–L5
transition rather than hard difficulty cuts.

### 3.2 Sliding-window partition

Define width-3 windows over levels,

$$
W_k = \{k+1,\,k+2,\,k+3\},\qquad k = 0,\dots,5,
$$

so $W_0=\{1,2,3\},\ W_1=\{2,3,4\},\ \dots,\ W_5=\{6,7,8\}$. Each level $L$ is eligible for
the stages that contain it,

$$
E_L \;=\; \{\,k : L \in W_k\,\}.
$$

Level mass is *distributed* (not duplicated) across its eligible stages using fixed
dwell weights $\omega = (\omega_0,\dots,\omega_5) = (13,23,25,16,11,11)$. For level $L$
with $n_L = |\{p:\ell(p)=L\}|$ problems, normalize over eligible stages,

$$
\tilde\omega^{(L)}_k \;=\; \frac{\omega_k}{\sum_{j\in E_L}\omega_j},\qquad k\in E_L,
$$

and allocate integer counts by cumulative flooring,

$$
c_{L,k} \;=\; \Big\lfloor n_L\!\!\sum_{\substack{j\in E_L,\, j\le k}}\!\tilde\omega^{(L)}_j \Big\rfloor
        - \Big\lfloor n_L\!\!\sum_{\substack{j\in E_L,\, j< k}}\!\tilde\omega^{(L)}_j \Big\rfloor ,
$$

with any rounding remainder assigned to the last eligible stage $\max E_L$. The counts
$c_{L,k}$ depend only on $(n_L,\omega)$ — **not on any random seed** — hence every arm
built on this backbone shares an identical level-by-stage count table $C=(c_{L,k})$.
The resulting stage sizes are

$$
|\mathcal{P}_k| \;=\; \sum_L c_{L,k} \;=\; (4071,\ 7065,\ 7626,\ 4874,\ 3160,\ 1947).
$$

The arm that fills $C$ by drawing uniformly at random within each level is `cliff_P`
(difficulty backbone, subject-agnostic).

---

## 4. Subject axis from representation geometry

### 4.1 One-dimensional subject embedding

Let $S \in \mathbb{R}^{7\times 7}$ be the centered subject centroid-cosine matrix (pooled
thinking activations). Convert to a dissimilarity $D_{ij} = 1 - S_{ij}$ and apply
classical MDS: with the centering matrix $J = I - \tfrac{1}{7}\mathbf{1}\mathbf{1}^\top$,

$$
B \;=\; -\tfrac12\, J \, D^{\circ 2} \, J ,
\qquad
B = \sum_i \lambda_i\, v_i v_i^\top \ (\lambda_1\ge\lambda_2\ge\cdots),
$$

and take the leading axis as the subject coordinate

$$
\phi(s) \;=\; \sqrt{\lambda_1}\; v_{1,s}.
$$

The leading axis captures $\lambda_1/\sum_i|\lambda_i| = 57\%$ of the configuration and
orders subjects along a **discrete ↔ continuous** contrast.

### 4.2 Co-move orientation

The sign of $\phi$ is fixed so that the subject axis is *positively* aligned with
difficulty. Let $\bar\ell(s)$ be the mean level of subject $s$; flip if needed,

$$
\phi \leftarrow \operatorname{sign}\!\big(\operatorname{corr}(\phi,\ \bar\ell)\big)\,\phi,
\qquad \operatorname{corr}(\phi,\bar\ell) = +0.24 \ (\text{after orientation}).
$$

The oriented order (increasing $\phi$: discrete → continuous) is

$$
\text{C\&P} \;(-0.88) \prec \text{NT} \;(-0.62) \prec \text{Prealg} \;(-0.36)
\prec \text{Geom} \;(-0.03) \prec \text{Alg} \;(+0.28)
\prec \text{IntAlg} \;(+0.74) \prec \text{Precalc} \;(+0.87).
$$

The correlation with difficulty is deliberately weak ($+0.24$): the subject axis is
largely *orthogonal* to level, so "co-move" fixes the orientation without collapsing the
two axes.

### 4.3 Difficulty-matched subject assignment (`cliff_subjgeo`)

We fill the **same** count table $C$ as `cliff_P`, changing only *which* members of each
level occupy each stage. Within level $L$, sort its problems by subject coordinate,

$$
p^{(1)}_L,\dots,p^{(n_L)}_L \quad\text{s.t.}\quad
\phi\big(s(p^{(1)}_L)\big) \le \cdots \le \phi\big(s(p^{(n_L)}_L)\big),
$$

(ties broken by problem id for determinism) and deal the sorted list into the eligible
stages $E_L$ *in ascending stage order* using the counts $c_{L,k}$: the first $c_{L,\min E_L}$
go to the earliest eligible stage, and so on. Because early stages are dominated by easy
levels and now also receive low-$\phi$ (discrete) problems, while late stages receive
high-$\phi$ (continuous) problems, the subject axis **co-moves** with difficulty. This is
witnessed by the per-stage mean coordinate (rank scale, discrete $0$ → continuous $6$):

$$
\bar\phi_k \;=\; \big(1.1,\ 2.3,\ 3.1,\ 3.3,\ 3.4,\ 4.3\big)\quad(\text{monotone}\uparrow),
$$

versus a flat $\bar\phi_k \approx 2.5$ for random assignment. **Crucially, the level
marginal of every stage is byte-identical to `cliff_P`;** only the subject composition
differs, so any downstream difference is attributable to subject *beyond* difficulty.

### 4.4 Matched control (`cliff_subjrand`)

The control fills the identical $C$ by a uniform random permutation within each level,

$$
\pi_L \sim \mathrm{Unif}(\mathfrak{S}_{n_L}),\qquad
\mathcal{P}_k \supseteq \{\,p^{(\pi_L(1))}_L,\dots\,\}\ \text{by counts } c_{L,k},
$$

seeded per replicate ($\texttt{seed}\in\{0,1\}$). Across seeds `cliff_subjrand` samples the
**null distribution of random subject arrangements**; `cliff_subjgeo` is the single
geometry-structured draw to be tested against that null.

---

## 5. Context scaling

Generation budget is ramped with stage difficulty. Let

$$
m \;=\; (m_0,\dots,m_5) \;=\; (1024,\,1536,\,2048,\,2560,\,3072,\,4096).
$$

At optimization step $t$ belonging to stage $k(t)$, the on-policy sampler's maximum new
tokens is set to $m_{k(t)}$ (for boundary micro-batches spanning stages, we take the
$\max$ over stages present so no harder example is truncated). Prompt budget uses the
global cap $L_{\max}=24{,}000$, leaving $L_{\max}-m_{k(t)} \ge 19{,}904$ tokens for prompts
at all stages. The expected per-step budget is

$$
\mathbb{E}_t[m_{k(t)}] \;=\; \frac{1}{T}\sum_k |{\text{steps in }k}|\; m_k \;\approx\; 2118,
$$

roughly halving generation compute relative to a fixed $m\equiv 4096$, while allocating
long rollouts only where difficulty warrants.

---

## 6. Objective (unchanged) and controls

The learner is trained with the original **OPSD on-policy distillation** objective, a
fixed teacher ($\texttt{fixed\_teacher}=\text{true}$), LoRA adapters, and vLLM-colocated
rollouts. The trainer code (loss, generation, teacher) is inherited *verbatim*: the
curriculum layer changes only (i) the visiting order of examples (the schedule) and (ii)
the per-step generation budget (context scaling). No teacher-update mechanism is
introduced.

The comparison ladder isolates each axis:

| Axis under test | Treatment | Matched control | Reads as |
|---|---|---|---|
| Difficulty structure | `cliff_P` | `diff`, `shuffle` | overlap/cliff effect |
| Subject geometry (beyond difficulty) | `cliff_subjgeo` | `cliff_P`, `cliff_subjrand`$_{s0,s1}$ | subject-axis effect |

All subject-test arms share the level-by-stage count table $C$ and the context schedule
$m$; only subject composition varies.

---

## 7. Evaluation protocol

Checkpoints are evaluated non-thinking on AIME-2024, AIME-2025, HMMT-2025 (30 problems
each) and MATH-500. The primary statistic is $\mathrm{avg}@n$ (mean solve rate over $n$
samples), pooled over the three 30-problem competition sets (90 problems) to reduce
problem-set variance; MATH-500 is near-saturated (base $\approx 84.6$) and is reported for
completeness rather than discrimination. $\mathrm{pass}@n$ is recorded as a free
by-product but saturates and is not used to adjudicate the subject axis.

**Decision rule.** Let $\mu(\cdot)$ be pooled $\mathrm{avg}@12$. The subject axis is
supported iff

$$
\mu(\texttt{cliff\_subjgeo}) \;>\; \max\big\{\mu(\texttt{cliff\_P}),\ \mu(\texttt{cliff\_subjrand}_{s0}),\ \mu(\texttt{cliff\_subjrand}_{s1})\big\}
$$

by a margin exceeding the control's seed spread; otherwise the axis is declared null at
the achievable power and `cliff_P` remains the main curriculum.

---

## Appendix A — Arms

| Arm | Difficulty | Subject | Role |
|---|---|---|---|
| `shuffle` | none (random 6-way) | random | lower baseline |
| `diff` | tight bands (no overlap) | random | difficulty-only baseline |
| `cliff_P` | sliding window (overlap) | random | difficulty main / subject control |
| `cliff_subjgeo` | sliding window | geometry, co-move | **full main / treatment** |
| `cliff_subjrand`$_{s0,s1}$ | sliding window | random (re-seeded) | subject null draws |

Retired: `subj_V1`, `subj_shuf` (subject-primary blocked; superseded by the
difficulty-matched formulation above).

## Appendix B — Hyperparameters (`full_4b_cliff.yaml`)

Qwen3-4B · bf16 · flash-attn-2 · LoRA $r{=}64,\ \alpha{=}128$ (q,k,v,o,gate,up,down) ·
lr $5\times10^{-6}$ · max-grad-norm $0.1$ · 1 epoch · global batch $B{=}32$
($2{\times}8{\times}2$) · $\beta{=}0$ · $\lambda{=}1$ · jsd-token-clip $0.06$ ·
temperature $1.1$ · top-$p$ $0.95$ · top-$k$ $20$ · $L_{\max}{=}24{,}000$ ·
$m_{\max}{=}4096$ (ramped per §5) · $T = \lceil N/B\rceil = 899$ steps ·
within-stage order = shuffle · passes = 1.
