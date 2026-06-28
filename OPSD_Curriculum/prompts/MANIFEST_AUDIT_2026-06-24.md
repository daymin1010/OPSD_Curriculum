# MANIFEST AUDIT — Executive Summary

**Date**: 2026-06-24 03:00 KST
**Scope**: 8 manifests (4 rungs × 2 arms) + row table verification

## Verdict (TL;DR)

| Check | Result |
|---|---|
| Cross-arm universe identity (each rung) | ✅ **PERFECT** — MD5 match, Jaccard = 1.0 for all 4 rungs |
| Stratification preserved across rungs | ✅ **OK** — q4/full ≈ 0.24-0.27, mini100/q4 ≈ 0.44, mini50/q4 ≈ 0.22 per stage |
| Level distribution preserved | ✅ **OK** — mean 4.01→4.03→4.03→4.04, std 1.55-1.57 |
| Subject distribution preserved | ✅ **OK** — ratios stable within ±1pp |
| **mini50 ⊂ mini100 (strict nesting)** | ⚠️ **FAIL** — Jaccard 0.17, only 700/1603 (44%) shared |
| q4 ⊂ full, mini100 ⊂ q4, mini50 ⊂ q4 | ✅ **OK** |

## Answer to verifier's two questions

### Q1: "Same subset across 5 conditions?"
- Within wave-1 (current experiment: cond2_diff vs cond3_ours), **YES at every rung**:
  - full: identical 28,771 problems
  - q4: identical 7,193
  - mini100: identical 3,198
  - mini50: identical 1,603
- MD5 hash matches per rung (e.g., mini50 = `036306e58762` for both arms). Symmetric difference = 0. Jaccard = 1.0000.
- The five conditions in the manifest registry (`cond1_random`, `cond2_diff`, `cond3_ours_C2`, `cond4_shuffle`, `cond5_diffmatched`) are **all built from the same 28,771 universe**; only stage ordering differs. We currently run cond2 vs cond3 only.

### Q2: "Difficulty / subject ratios preserved?"
- **Yes, stratification is preserved.** See §3, §5, §7, §8 of detailed audit.
- Per-stage subsampling rate is uniform (no stage over/under-represented in mini rungs).
- Overall level histogram and subject mix nearly identical across rungs.

### Q3 (from boss's framing): "Internal consistency — Cline's chat summary had T=51 for mini50"
- **Cline's chat doc had off-by-one error in mini50/mini100 T.** Actual schedule:
  - mini50: **diff T=52, ours T=53** (chat said T≈51)
  - mini100: **diff T=102, ours T=103** (chat said T≈100)
  - q4: **diff T=227, ours T=228** (chat said T=225)
  - full: **diff T=900, ours T=901** (chat said T=900)
- Difference comes from `tail_policy=partial`: every stage uses `ceil(n/32)` steps, so 5 stages × ~1 extra each = +5 above the naive N/32 estimate.
- Per-stage boundaries are now precisely known (§4).
- **No correction needed** — training config uses `num_train_epochs=1` and follows manifest order; the ~1-2 extra steps don't change the experiment.

## The one issue worth noting: mini50 ⊄ mini100

`make_mini_manifests.py` samples mini50 and mini100 **independently** from q4 (each is its own stratified subsample), not in a nested fashion. Concretely:
- mini50: 1,603 problems from q4
- mini100: 3,198 problems from q4 (independent draw, same seed=42 but different fraction)
- Overlap: |mini50 ∩ mini100| = **700** (Jaccard 0.17)

### Is this a problem?
**For A/B fairness within each rung: NO.** mini50_diff vs mini50_ours use the SAME 1,603 problems. mini100_diff vs mini100_ours use the SAME 3,198 problems. These are the comparisons that matter for our research question (stage order effect).

**For "scale ladder" interpretation: PARTIAL.** mini50 and mini100 are independent points on the ladder, not strict scaling of the same experiment. This means:
- ✅ Each point is internally fair (its own diff vs ours)
- ✅ Per-rung peak step finding is valid
- ⚠️ Comparing mini50@step50 with mini100@step50 introduces problem-level noise (different problems)
- ⚠️ Cannot claim mini50 is a "warm-up" or "prefix" of mini100

### Should we fix it?
Two options:
1. **Accept as-is**: argue mini50 and mini100 are independent points on the ladder; each answers a question at its step budget. The shared 700-problem intersection still provides anchor.
2. **Re-roll mini50 as a nested subset of mini100**: regenerate `stages_cond2_diff_mini50.json` and `stages_cond3_ours_C2_mini50.json` to be a strict subset of mini100. Would require canceling jobs 88005/88006 and re-submitting.

**Recommendation**: ACCEPT AS-IS for wave-1.
- The 4 jobs are pending in queue; canceling and rebuilding loses signal but adds delay.
- The non-nesting actually has a small **scientific upside**: two independent subsamples give us a robustness check — if both mini50 and mini100 show the same diff vs ours pattern, the effect is robust to specific problem selection. If they disagree, we know problem-level variance is high.
- For final report, just disclose: "mini50 and mini100 are independent stratified subsamples of q4, sharing 44% of problems".

---

# MANIFEST AUDIT — universe identity, nesting, proportions, schedule

Source row table: `src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet`
B_glob = 32, tail_policy = partial

## 1. Universe identity (diff vs ours, per rung)

| rung | N(diff) | N(ours) | MD5(diff) | MD5(ours) | identical | symdiff |
|---|---|---|---|---|---|---|
| full | 28771 | 28771 | `3f54d1a51c71` | `3f54d1a51c71` | ✅ | 0 |
| q4 | 7193 | 7193 | `f0bdd2b694c0` | `f0bdd2b694c0` | ✅ | 0 |
| mini100 | 3198 | 3198 | `a25df44a01a9` | `a25df44a01a9` | ✅ | 0 |
| mini50 | 1603 | 1603 | `036306e58762` | `036306e58762` | ✅ | 0 |

## 2. Nesting check: mini50 ⊂ mini100 ⊂ q4 ⊂ full (per arm)

| arm | mini50⊂mini100 | mini100⊂q4 | q4⊂full | mini50⊂full |
|---|---|---|---|---|
| diff | ❌ | ✅ | ✅ | ✅ |
| ours | ❌ | ✅ | ✅ | ✅ |

## 3. Stage composition (N, mean level, subject mix)

### rung = full

| arm | order | stage_idx | N | level μ | level σ | top-3 subjects |
|---|---|---|---|---|---|---|
| diff | 0 | 0 | 5755 | 1.81 | 0.48 | Prealgebra(2348), Algebra(1561), Counting & Probability(667) |
| diff | 1 | 1 | 5754 | 3.03 | 0.17 | Algebra(1769), Counting & Probability(961), Number Theory(909) |
| diff | 2 | 2 | 5754 | 4.00 | 0.00 | Algebra(1554), Geometry(1317), Number Theory(1101) |
| diff | 3 | 3 | 5754 | 4.97 | 0.17 | Geometry(1389), Algebra(1365), Number Theory(1271) |
| diff | 4 | 4 | 5754 | 6.25 | 0.59 | Number Theory(1621), Geometry(1370), Counting & Probability(798) |
| ours | 0 | 0 | 5301 | 2.90 | 0.91 | Algebra(3079), Geometry(1190), Precalculus(700) |
| ours | 1 | 1 | 6144 | 2.75 | 0.67 | Counting & Probability(2341), Prealgebra(1998), Algebra(1805) |
| ours | 2 | 2 | 5759 | 3.91 | 1.81 | Intermediate Algebra(1957), Algebra(1444), Number Theory(1443) |
| ours | 3 | 3 | 5933 | 5.17 | 0.98 | Number Theory(3970), Precalculus(1468), Algebra(495) |
| ours | 4 | 4 | 5634 | 5.31 | 1.01 | Geometry(4076), Counting & Probability(1518), Number Theory(23) |

### rung = q4

| arm | order | stage_idx | N | level μ | level σ | top-3 subjects |
|---|---|---|---|---|---|---|
| diff | 0 | 0 | 1370 | 1.80 | 0.48 | Prealgebra(566), Algebra(364), Counting & Probability(167) |
| diff | 1 | 1 | 1412 | 3.03 | 0.17 | Algebra(448), Counting & Probability(229), Number Theory(207) |
| diff | 2 | 2 | 1522 | 4.00 | 0.00 | Algebra(418), Geometry(340), Number Theory(288) |
| diff | 3 | 3 | 1485 | 4.98 | 0.14 | Geometry(381), Algebra(358), Number Theory(320) |
| diff | 4 | 4 | 1404 | 6.25 | 0.60 | Number Theory(402), Geometry(350), Counting & Probability(184) |
| ours | 0 | 0 | 1326 | 2.95 | 0.92 | Algebra(778), Geometry(285), Precalculus(178) |
| ours | 1 | 1 | 1520 | 2.77 | 0.68 | Counting & Probability(585), Prealgebra(483), Algebra(452) |
| ours | 2 | 2 | 1410 | 3.96 | 1.81 | Intermediate Algebra(478), Algebra(381), Number Theory(328) |
| ours | 3 | 3 | 1479 | 5.15 | 0.98 | Number Theory(1002), Precalculus(360), Algebra(117) |
| ours | 4 | 4 | 1458 | 5.28 | 0.99 | Geometry(1071), Counting & Probability(372), Number Theory(8) |

### rung = mini100

| arm | order | stage_idx | N | level μ | level σ | top-3 subjects |
|---|---|---|---|---|---|---|
| diff | 0 | 0 | 606 | 1.80 | 0.48 | Prealgebra(252), Algebra(160), Counting & Probability(74) |
| diff | 1 | 1 | 632 | 3.03 | 0.18 | Algebra(203), Counting & Probability(102), Number Theory(92) |
| diff | 2 | 2 | 678 | 4.00 | 0.00 | Algebra(184), Geometry(151), Number Theory(128) |
| diff | 3 | 3 | 658 | 4.98 | 0.12 | Geometry(169), Algebra(159), Number Theory(142) |
| diff | 4 | 4 | 624 | 6.25 | 0.60 | Number Theory(180), Geometry(156), Counting & Probability(82) |
| ours | 0 | 0 | 589 | 2.95 | 0.92 | Algebra(346), Geometry(126), Precalculus(79) |
| ours | 1 | 1 | 676 | 2.77 | 0.68 | Counting & Probability(260), Prealgebra(215), Algebra(201) |
| ours | 2 | 2 | 626 | 3.95 | 1.81 | Intermediate Algebra(212), Algebra(169), Number Theory(145) |
| ours | 3 | 3 | 658 | 5.15 | 0.98 | Number Theory(446), Precalculus(160), Algebra(52) |
| ours | 4 | 4 | 649 | 5.28 | 1.00 | Geometry(476), Counting & Probability(166), Number Theory(4) |

### rung = mini50

| arm | order | stage_idx | N | level μ | level σ | top-3 subjects |
|---|---|---|---|---|---|---|
| diff | 0 | 0 | 312 | 1.83 | 0.51 | Prealgebra(126), Algebra(88), Counting & Probability(37) |
| diff | 1 | 1 | 299 | 3.01 | 0.08 | Algebra(85), Counting & Probability(51), Number Theory(46) |
| diff | 2 | 2 | 344 | 4.00 | 0.00 | Algebra(101), Geometry(76), Number Theory(64) |
| diff | 3 | 3 | 333 | 4.97 | 0.17 | Geometry(85), Algebra(80), Number Theory(71) |
| diff | 4 | 4 | 315 | 6.26 | 0.61 | Number Theory(90), Geometry(79), Counting & Probability(41) |
| ours | 0 | 0 | 295 | 2.95 | 0.91 | Algebra(173), Geometry(63), Precalculus(40) |
| ours | 1 | 1 | 338 | 2.77 | 0.68 | Counting & Probability(130), Prealgebra(107), Algebra(101) |
| ours | 2 | 2 | 314 | 3.96 | 1.82 | Intermediate Algebra(106), Algebra(85), Number Theory(73) |
| ours | 3 | 3 | 329 | 5.15 | 0.99 | Number Theory(223), Precalculus(80), Algebra(26) |
| ours | 4 | 4 | 327 | 5.28 | 1.00 | Geometry(240), Counting & Probability(83), Prealgebra(2) |

## 4. Schedule (B_glob=32, tail_policy=partial)

| rung | arm | T_total | per-stage (order: n → T_stage → cum_T) |
|---|---|---|---|
| full | diff | **900** | 0: 5755→180→180 ; 1: 5754→180→360 ; 2: 5754→180→540 ; 3: 5754→180→720 ; 4: 5754→180→900 |
| full | ours | **901** | 0: 5301→166→166 ; 1: 6144→192→358 ; 2: 5759→180→538 ; 3: 5933→186→724 ; 4: 5634→177→901 |
| q4 | diff | **227** | 0: 1370→43→43 ; 1: 1412→45→88 ; 2: 1522→48→136 ; 3: 1485→47→183 ; 4: 1404→44→227 |
| q4 | ours | **228** | 0: 1326→42→42 ; 1: 1520→48→90 ; 2: 1410→45→135 ; 3: 1479→47→182 ; 4: 1458→46→228 |
| mini100 | diff | **102** | 0: 606→19→19 ; 1: 632→20→39 ; 2: 678→22→61 ; 3: 658→21→82 ; 4: 624→20→102 |
| mini100 | ours | **103** | 0: 589→19→19 ; 1: 676→22→41 ; 2: 626→20→61 ; 3: 658→21→82 ; 4: 649→21→103 |
| mini50 | diff | **52** | 0: 312→10→10 ; 1: 299→10→20 ; 2: 344→11→31 ; 3: 333→11→42 ; 4: 315→10→52 |
| mini50 | ours | **53** | 0: 295→10→10 ; 1: 338→11→21 ; 2: 314→10→31 ; 3: 329→11→42 ; 4: 327→11→53 |

## 5. Per-stage N ratios across rungs (proportionality sanity)

### arm = diff

| stage(order) | full | q4 | q4/full | mini100 | m100/q4 | mini50 | m50/q4 |
|---|---|---|---|---|---|---|---|
| 0 | 5755 | 1370 | 0.238 | 606 | 0.442 | 312 | 0.228 |
| 1 | 5754 | 1412 | 0.245 | 632 | 0.448 | 299 | 0.212 |
| 2 | 5754 | 1522 | 0.265 | 678 | 0.445 | 344 | 0.226 |
| 3 | 5754 | 1485 | 0.258 | 658 | 0.443 | 333 | 0.224 |
| 4 | 5754 | 1404 | 0.244 | 624 | 0.444 | 315 | 0.224 |

### arm = ours

| stage(order) | full | q4 | q4/full | mini100 | m100/q4 | mini50 | m50/q4 |
|---|---|---|---|---|---|---|---|
| 0 | 5301 | 1326 | 0.250 | 589 | 0.444 | 295 | 0.222 |
| 1 | 6144 | 1520 | 0.247 | 676 | 0.445 | 338 | 0.222 |
| 2 | 5759 | 1410 | 0.245 | 626 | 0.444 | 314 | 0.223 |
| 3 | 5933 | 1479 | 0.249 | 658 | 0.445 | 329 | 0.222 |
| 4 | 5634 | 1458 | 0.259 | 649 | 0.445 | 327 | 0.224 |

## 6. Cross-arm universe identity (same problems used in both arms?)

| rung | |U(diff) ∩ U(ours)| | |U(diff) ∪ U(ours)| | Jaccard |
|---|---|---|---|
| full | 28771 | 28771 | 1.0000 |
| q4 | 7193 | 7193 | 1.0000 |
| mini100 | 3198 | 3198 | 1.0000 |
| mini50 | 1603 | 1603 | 1.0000 |

## 7. Overall level histogram per rung (arm=diff; ours mirrors by §1)

| rung | N | level=1 | 2 | 3 | 4 | 5 | 6 | 7 | mean | std |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 28771 | 1319 | 4230 | 5790 | 6091 | 5988 | 3590 | 1710 | 4.01 | 1.57 |
| q4 | 7193 | 320 | 1002 | 1416 | 1596 | 1556 | 866 | 421 | 4.03 | 1.55 |
| mini100 | 3198 | 142 | 444 | 630 | 710 | 692 | 385 | 187 | 4.03 | 1.55 |
| mini50 | 1603 | 71 | 223 | 315 | 356 | 346 | 193 | 94 | 4.04 | 1.56 |

## 8. Subject distribution per rung (arm=diff)

| rung | N | Algebra | Counting & Probability | Geometry | Intermediate Algebra | Number Theory | Other | Prealgebra | Precalculus |
|---|---|---|---|---|---|---|---|---|---|
| full | 28771 | 6823 | 3859 | 5266 | 2289 | 5436 | 0 | 2903 | 2195 |
| q4 | 7193 | 1728 | 957 | 1356 | 563 | 1338 | 0 | 705 | 546 |
| mini100 | 3198 | 768 | 426 | 602 | 250 | 595 | 0 | 314 | 243 |
| mini50 | 1603 | 385 | 213 | 303 | 125 | 298 | 0 | 157 | 122 |
