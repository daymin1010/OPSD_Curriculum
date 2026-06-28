# Method: A Two-Axis Curriculum from Difficulty Labels and Subject Activation Geometry

*(paper-ready draft of the curriculum-construction method; notation is self-contained.
Implementation: `build_stages_subjslack.py`. Empirical validation: `REPORT_stagebuild_subjslack_2026-06-24.md`.)*

---

## 1. Setup and notation

We are given a training pool of \(N\) reasoning problems \(\mathcal{D}=\{x_i\}_{i=1}^{N}\) for on-policy self-distillation (OPSD). Each problem carries two **label** attributes:

- a difficulty level \(\ell(x_i)\in\{1,\dots,L_{\max}\}\) (here \(L_{\max}=8\)),
- a subject \(s(x_i)\in\mathcal{S}\) (here \(|\mathcal{S}|=7\): Algebra, Intermediate Algebra, Precalculus, Geometry, Counting&Probability, Number Theory, Prealgebra).

A **unit** is a (subject, level) cell \(u=(s,\ell)\); let \(\mathcal{U}\) be the set of units with at least \(n_{\min}\) members (\(n_{\min}=30\)). A curriculum partitions \(\mathcal{D}\) into an **ordered** sequence of \(K\) stages \(\mathcal{C}=(S_1,\dots,S_K)\) (here \(K=5\)) that training visits in order.

**Goal.** Difficulty alone already induces an ordering (sort by \(\ell\)). We ask whether *subject* is a useful **second axis**, and we want the subject signal to come not from the categorical label but from **where subjects sit in the model's own internal representation**. The method below reads a subject geometry from activations, projects it to a scalar per subject, and uses it to perturb a difficulty backbone — tightly enough to preserve monotone difficulty, strongly enough to be statistically distinguishable from difficulty alone.

---

## 2. Subject geometry axis \(g\)

### 2.1 Difficulty-orthogonal subject representation
For a held-out probe set of \(M\) problems we extract, from the policy model \(\pi_\theta\) (Qwen3-8B) run in **thinking mode**, an activation-shift vector \(\Delta a_i\in\mathbb{R}^{D}\) per problem — the change in the MLP down-projection input between the first and last reasoning tokens, averaged over a subject-dominant layer window \(W\) (here \(M=3025\) pooled probe problems; faithful-mode shifts are discarded). To remove difficulty as a confound we **residualize against the per-level mean**:

\[
\tilde{a}_i \;=\; \Delta a_i \;-\; \mu_{\ell(x_i)},\qquad
\mu_{\ell}=\frac{1}{|\{j:\ell(x_j)=\ell\}|}\sum_{j:\ell(x_j)=\ell}\Delta a_j .
\]

Per-unit centroids are \(L_2\)-normalized means \(c_u = \operatorname{norm}\!\big(\operatorname{mean}_{i\in u}\,\hat{\tilde a}_i\big)\), and \(M_{uv}=\langle c_u,c_v\rangle\) is the unit–unit cosine-similarity matrix (difficulty-orthogonal subject geometry).

### 2.2 Scalar subject coordinate
We take the **leading axis** of \(M\) via classical MDS: double-center \(B=-\tfrac12 J\,(\mathbf{1}-M)\,J\) (equivalently \(B=JMJ\)) with \(J=I-\tfrac1{|\mathcal{U}|}\mathbf{1}\mathbf{1}^\top\), and let \(g^{\mathrm{raw}}_u\) be the top-eigenvector coordinate of \(B\). We pool to a **per-subject** scalar (units of the same subject are tightly clustered on this axis, justifying the pooling),

\[
g^{\mathrm{raw}}(s)=\operatorname{mean}_{u:\,\mathrm{subj}(u)=s}\,g^{\mathrm{raw}}_u,
\]

orient the sign by a fixed convention (here so that the discrete-math cluster exceeds the geometry cluster, making \(g\) reproducible up to nothing), and min–max scale across subjects to a bounded range:

\[
g(s)=\frac{g^{\mathrm{raw}}(s)-\min_{s'}g^{\mathrm{raw}}(s')}{\max_{s'}g^{\mathrm{raw}}(s')-\min_{s'}g^{\mathrm{raw}}(s')}-\tfrac12\;\in\;[-\tfrac12,\tfrac12].
\]

Intuitively \(g(s)<0\) subjects are placed *earlier* and \(g(s)>0\) *later*, **at equal difficulty**. (Empirically the algebra cluster maps to \(g<0\), the discrete/number-theory cluster to \(g>0\), geometry near \(0\).)

---

## 3. Two-axis score and stage construction

Define a per-problem **score** combining the difficulty backbone with a bounded subject perturbation:

\[
\boxed{\;\rho(x_i)\;=\;\ell(x_i)\;+\;\alpha\,g\big(s(x_i)\big)\;}\qquad \alpha\ge 0 .
\]

Because \(g\in[-\tfrac12,\tfrac12]\), the subject term lies in \([-\tfrac{\alpha}{2},\tfrac{\alpha}{2}]\): with moderate \(\alpha\) a subject can move a problem by at most \(\sim\!1\) difficulty level, so the **stage mean-levels stay monotone** while subjects are reordered *within* the overlap of adjacent levels.

**Unit-atomic partition.** Units are kept intact (all problems of a (subject,level) cell share one score and one stage). Sort units by \(\rho\); cut into \(K\) contiguous groups at the unit boundary whose cumulative mass is closest to each target \(k\,N/K\). This yields stages with naturally unequal sizes that respect cell atomicity. Within a stage, training order is a fixed shuffle (seed-controlled); the *stage* order is the curriculum.

The construction has three knobs: \(\alpha\) (subject-skew strength), \(K\) (stages), \(W\)/residualization (which representation defines \(g\)). We select \(\alpha\) by the gates in §5; here \(\alpha=2.0,\ K=5\).

---

## 4. Controls / baselines (for the ablation)

All conditions share the **same problem universe** \(\mathcal{D}\); only the stage assignment differs.

- **Difficulty-only** (\(\textsc{diff}\)): \(\rho=\ell\) (i.e. \(\alpha=0\)).
- **Ours** (\(\textsc{ours}\)): \(\rho=\ell+\alpha g\) as above.
- **Difficulty-matched random** (\(\textsc{diffmatched}\)): take \(\textsc{ours}\)'s per-(level, stage) **counts** and reassign *which* problems of each level go to each stage **uniformly at random**. This holds the level distribution per stage identical to \(\textsc{ours}\) but destroys the subject ordering — it is the critical control that isolates the subject-geometry effect from difficulty.

Beating \(\textsc{diffmatched}\) is the load-bearing claim: it shows the gain is the *geometry-driven subject ordering*, not the difficulty schedule.

---

## 5. Selection gates (why this construction, and how \(\alpha\) is chosen)

A candidate \((\alpha,K)\) is admitted only if all hold (measured on \(\mathcal{D}\)):

1. **Monotone difficulty.** Stage mean-levels strictly increasing: \(\min_k\big(\bar\ell_{k+1}-\bar\ell_k\big)\ge-\epsilon\).
2. **Tight bands.** Mean per-stage level variance below threshold (each stage spans \(\le 2\) levels).
3. **Distinct from \(\textsc{diffmatched}\)** *(load-bearing).* The per-level stage×subject deviation
   \(\;T=\sum_{\ell}\sum_{k}\sum_{s}\big|\,n_{\ell k s}-n_{\ell k}\,p_{\ell s}\big|\;\)
   is far above its permutation null (shuffle subjects within level); require \(p<0.01\).
4. **Distinct from \(\textsc{diff}\).** Same statistic vs. the difficulty-only assignment is non-trivial (difficulty agreement is *allowed* — the distinction lives on the subject axis).
5. **Representational smoothness.** Consecutive-stage distance in the **full** activation centroid space (level *not* removed) is no larger than \(\textsc{diff}\)'s — i.e. the two-axis order does not increase, and ideally decreases, the representational jump between adjacent stages.

Among admitted candidates, prefer larger subject separation (gate 3) at the smallest \(\alpha\) (tightest difficulty).

---

## 6. Algorithm

```
Input: pool D with labels (level, subject); probe activations; alpha; K; window W
Output: ordered stages (S_1, ..., S_K)

# --- subject geometry axis g (once) ---
1  for each probe problem i: a_i  <- thinking activation shift over window W
2  ã_i <- a_i - mean_level(level(i))                 # residualize out difficulty
3  for each unit u: c_u <- normalize(mean_{i in u} normalize(ã_i))
4  M    <- [ <c_u, c_v> ]_{u,v}                       # subject-geometry similarity
5  graw <- leading classical-MDS axis of M            # per-unit scalar
6  g(s) <- scale_to[-0.5,0.5]( sign * mean_{u in s} graw_u )   # per-subject

# --- two-axis stage construction ---
7  for each problem i: rho_i <- level(i) + alpha * g(subject(i))
8  order units by rho                                 # unit-atomic
9  cut ordered units into K contiguous mass-balanced stages
10 return stages in ascending-rho order

# --- controls ---
   diff         : repeat 7-10 with alpha = 0
   diffmatched  : keep ours' per-(level,stage) counts; randomize subjects within level
```

CPU-only given precomputed probe activations; \(M\) is \(|\mathcal{U}|\times|\mathcal{U}|\).

---

## 7. Properties

- **Difficulty-orthogonal subject signal.** \(g\) is built from level-residualized activations, so the second axis is not a difficulty proxy (verified: subject geometry survives a length/level confound gate).
- **Monotone + tight difficulty backbone.** Bounded \(\alpha g\) guarantees stage mean-levels increase and bands stay narrow → recovers the easy→hard mastery that pure subject-clustering destroys.
- **Identifiable subject effect.** By construction \(\textsc{ours}\) and \(\textsc{diffmatched}\) share the per-level difficulty schedule; any downstream difference is attributable to subject geometry alone.
- **Smaller representational jumps.** With \(\alpha\) chosen by gate 5, consecutive stages are closer in the model's full activation space than under difficulty-only ordering, operationalizing "smooth the representational transition between stages."

---

## 8. Instantiation used in experiments
\(K=5\), \(\alpha=2.0\), \(W\)=subject-dominant layers, \(M=3025\) thinking-mode probe problems, \(N=28{,}771\) (and stratified subsamples q4/mini150/mini100/mini50 with identical cross-arm universes). Resulting \(g\): Precalc \(-0.50\), Inter.Algebra \(-0.45\), Algebra \(-0.20\), Geometry \(-0.12\), Prealgebra \(+0.21\), Number Theory \(+0.33\), Counting&Prob \(+0.50\). Validation: stage mean-levels \(1.90\!\to\!3.04\!\to\!3.90\!\to\!4.89\!\to\!5.97\) (monotone), per-stage level var \(0.49\), \(\textsc{ours}\) vs \(\textsc{diffmatched}\) permutation \(p=0.005\), full-representation consecutive jump \(0.226<0.237\) (\(\textsc{diff}\)).

---

## 9. Relation to a path-based predecessor

The present score-based construction replaces an earlier one (\(\textsc{tiered}\); method id `tiered_difficulty_backbone_residual_within_tier`) that used the *same* residual subject geometry \(c_u\) but assigned stages by a **geometric path** rather than a scalar score. We record it because the contrast motivates the score formulation and the empirical gap is reported in §10.

**Construction.** Let \(d(u,v)=1-\langle c_u,c_v\rangle\) be the residual cosine distance.

1. **Difficulty tiers.** Sort units by \(\ell\) and split into \(T\) equal-mass tiers (\(T\) chosen by the gate below; the selected value was \(T=2\)).
2. **Within-tier nearest-neighbor path.** Inside each tier order units greedily,
   \[
   u_{(i+1)}=\arg\min_{v\,\in\,\text{tier}\,\setminus\,\{u_{(1)},\dots,u_{(i)}\}} d\big(u_{(i)},v\big),
   \]
   optionally seeded by a tier-start variant (low / high / nearest-to-previous-tail).
3. **Stitch and split.** Concatenate tiers low\(\to\)high into one unit order \(\pi\), then apply the same \(K\)-way mass split: \(\sigma_{\textsc{tiered}}(x)=\mathrm{Split}_K(\pi)\big(u(x)\big)\).
4. **Selection gate.** Over \((T,\text{variant})\), admit a candidate iff the Spearman rank correlation with the difficulty ordering lies in a target band and stage mean-levels are monotone,
   \[
   \rho_{\mathrm{sp}}\big(\sigma_{\textsc{diff}},\sigma_{\textsc{tiered}}\big)\in[0.4,0.7],\qquad \min_k\big(\bar\ell_{k+1}-\bar\ell_k\big)\ge-\epsilon,
   \]
   and pick \(\arg\min|\rho_{\mathrm{sp}}-0.55|\).

**Why it fails, formally.** Here a problem's stage is a function of its **position along the path** \(\pi\), and the path may jump to a unit that is far in difficulty but near in geometry. Difficulty is therefore *not* the primary sort key; with \(T=2\) the within-tier path oscillates \(\ell\), so the \(K\)-way mass cut on \(\pi\) slices *across* levels. The result (§10) is a non-monotone difficulty backbone and a per-stage level variance an order of magnitude above \(\textsc{diff}\) — there is no clean "level 2–3 mastery" stage, which is where \(\textsc{tiered}\) lost the most accuracy.

**The score fixes this by construction.** Replacing the path with the additive scalar
\(\rho(x)=\ell(x)+\alpha\,g(s(x))\) (§3) makes \(\ell\) the primary key and bounds the subject perturbation by \(|\alpha g|\le\alpha/2\). Two consequences: (i) \(\textsc{diff}\) is exactly the \(\alpha=0\) member of this family, so \(\textsc{diff}\) and \(\textsc{ours}\) differ **only** in the term \(\alpha g(s)\) — the subject contribution is identifiable; (ii) monotone, tight difficulty bands hold automatically rather than via a search gate.

---

## 10. Distributional comparison of the three constructions

All statistics below are computed on the **identical** universe \(N=28{,}771\) shared by the three stage assignments \(\sigma_{\textsc{diff}},\sigma_{\textsc{ours}},\sigma_{\textsc{tiered}}\) (so any difference is in the *assignment*, not the data). Permutation tests use \(500\) within-level subject shuffles (minimum attainable \(p=1/501\approx0.002\)).

| statistic | \(\textsc{diff}\) | \(\textsc{ours}\) (score) | \(\textsc{tiered}\) (path) |
|---|---:|---:|---:|
| Spearman \(\rho_{\mathrm{sp}}(\sigma_{\textsc{diff}},\cdot)\) | \(1.000\) | \(+0.889\) | \(+0.687\) |
| problems whose stage \(\neq\) \(\textsc{diff}\) | \(0\%\) | \(38.7\%\) | \(67.9\%\) |
| mean \(|\Delta\text{stage}|\) vs \(\textsc{diff}\) | \(0\) | \(0.405\) | \(0.865\) |
| stage mean-levels | \(1.76\!\to\!3.0\!\to\!4.0\!\to\!5.0\!\to\!6.22\) | \(1.90\!\to\!3.04\!\to\!3.90\!\to\!4.89\!\to\!5.97\) | \(2.90\!\to\!2.75\!\to\!3.91\!\to\!5.17\!\to\!5.31\) |
| monotone difficulty | yes | **yes** | **no** (\(\Delta=-0.15\)) |
| mean per-stage level variance | \(0.110\) | \(0.488\) | \(1.306\) |
| per-stage subject-comp. TV vs \(\textsc{diff}\) (mass-wtd) | — | \(0.162\) | \(0.574\) |
| within-level deviation \(T\) | \(1{,}893\) | \(28{,}174\) | \(36{,}490\) |
| \(T\,/\) permutation-null mean | \(25.3\times\) | \(37.8\times\) | \(34.2\times\) |
| permutation \(p\) | \(0.002\) | \(0.002\) | \(0.002\) |
| stage\(\times\)subject Cramér's \(V\) | \(0.312\) | \(0.379\) | \(0.648\) |

where the **within-level subject-deviation** statistic conditions on difficulty and is the cleanest pure-subject measure,
\[
T=\sum_{\ell}\sum_{k}\sum_{s}\Big|\,n_{\ell k s}-n_{\ell k}\,\hat p_{\ell s}\Big|,\qquad \hat p_{\ell s}=\frac{n_{\ell s}}{n_{\ell}},
\]
(\(n_{\ell k s}\) = #problems of level \(\ell\), stage \(k\), subject \(s\); \(T=0\) iff stage is independent of subject within every level). The per-stage **subject-composition TV** vs \(\textsc{diff}\) is \(\tfrac12\sum_s|p_k^{\textsc{diff}}(s)-p_k^{\bullet}(s)|\), mass-weighted over stages.

**Reading the table.**
- **\(\textsc{ours}\) stays close to the difficulty schedule but reorganizes subjects.** It keeps \(\rho_{\mathrm{sp}}=0.889\) with \(\textsc{diff}\), monotone mean-levels, and tight bands (variance \(0.49\), vs \(\textsc{tiered}\)'s \(1.31\)); yet \(38.7\%\) of problems change stage and the within-level subject signal rises to \(37.8\times\) its permutation null — the highest of the three. This is the intended regime: *minimal difficulty distortion, maximal clean subject reordering.*
- **\(\textsc{tiered}\) buys subject association by breaking difficulty.** Its high raw association (Cramér's \(V=0.648\)) and large TV (\(0.574\)) are conflated with difficulty disruption: monotonicity fails and per-stage variance is \(\sim\!12\times\) \(\textsc{diff}\). Its within-level \(T\) ratio (\(34.2\times\)) is actually *below* \(\textsc{ours}\), confirming that its extra movement is difficulty scrambling, not extra difficulty-orthogonal subject structure.
- **\(\textsc{diff}\) is not perfectly subject-neutral** (\(T=1{,}893\), \(25\times\) null): equal-mass cuts that fall inside a boundary level split that level's subjects by the deterministic alphabetical tie-break. \(\textsc{ours}\) raises this within-level structure by \(\sim\!15\times\) (to \(28{,}174\)) and replaces the arbitrary tie-break with the activation-geometry order \(g\).

In short, the score construction is the point that **maximizes difficulty-orthogonal subject reorganization per unit of difficulty distortion**: it moves a third of the problems and multiplies the within-level subject signal while leaving the easy\(\to\)hard backbone monotone and tight, which the path construction could not do.
