#!/usr/bin/env python3
"""
compare_pilot1_pilot2.py — Track A replication, RECOMPUTED (not parsed).
=======================================================================
Re-runs the EXACT similarity metrics (within / between / gap / perm-p /
level-ordinality) of similarity_analysis.py on BOTH pilots from the raw ΔA
.pt files, then lays them side-by-side for a like-for-like replication call.

WHY recompute (not parse the REPORT .md):
  - markdown parsing is brittle;
  - we must enforce identical filtering + a common LEVEL range (pilot2 has
    no L8) so the two pilots describe the SAME population.

REPLICATION PHILOSOPHY (important):
  - Each pilot is centered by its OWN global mean (μ_pilot1 for pilot1,
    μ_pilot2 for pilot2). Permutation shuffles labels WITHIN each pilot.
    We do NOT pool or cross-center — replication asks whether the SAME
    structure appears INDEPENDENTLY in each sample. (Pooled / μ_train
    centering belongs only to the later cross-universe Track C, which is the
    OPPOSITE handling — do not confuse the two.)

FILTERING (applied identically to both pilots):
  - load every *.pt (the loader applies no content filter);
  - drop rows with any non-finite ΔA (reported; expected 0);
  - restrict to LEVEL in {1..7} for ALL groupings (common range; pilot2 has
    no L8) so subject/level/unit all describe the same L1-L7 population.

METHOD NOTE (framing — group-similarity is OUR analysis, not NAIT):
  group-centroid cosine, global-mean centering, level ordinality and the
  label-permutation test are a diagnostic WE built to ask "does ΔA carry
  (subject/level) structure?". They are NOT the NAIT paper's method. NAIT's
  PCA-direction scoring is applied later, in the curriculum-direction stage.
  (Track C, supervised difficulty direction, is NAIT-inspired but supervised.)

OUTPUT: analysis/REPORT_pilot2_comparison.md   (CPU only; no GPU)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np

import similarity_analysis as sa  # reuse the validated metric functions

ANALYSIS = Path(__file__).resolve().parent
ACT = ANALYSIS.parent  # .../reasoning_pivot/activation
OUT = ANALYSIS / "REPORT_pilot2_comparison.md"

PILOT1_DIR = ACT / "outputs" / "pilot" / "shifts"
PILOT2_DIR = ACT / "outputs" / "pilot2" / "shifts"

MODES = [("THINKING", "DAT"), ("FAITHFUL", "DAF")]
COLS = ["subject", "level", "unit"]
LEVEL_MIN, LEVEL_MAX = 1, 7  # common range (pilot2 has no L8)

# Per-grouping permutation budget. subject/level are the replication-critical
# comparisons (cheap: few groups) -> 1000. unit has ~50 groups so each perm
# recomputes ~50 centroids => 5-10x costlier; it is descriptive, so 200 perms
# (p-resolution 0.005) is ample. Reported per-row in the table for honesty.
PERM_BY_COL = {"subject": 1000, "level": 1000, "unit": 200}



def finite_mask(*arrs):
    m = None
    for A in arrs:
        f = np.isfinite(A.reshape(A.shape[0], -1)).all(axis=1)
        m = f if m is None else (m & f)
    return m


def metrics_for_pilot(shifts_dir: Path, max_n=None):
    """Load a pilot, self-center, restrict to L1-L7, compute all metrics.
    Returns (info_dict, results_dict[(mode,col)] -> stats)."""
    DAF, DAT, md = sa.load_pilot(shifts_dir, max_n)
    n_loaded = len(md)

    # drop non-finite rows (identical filter both pilots)
    fm = finite_mask(DAF, DAT)
    n_nonfinite = int((~fm).sum())
    DAF, DAT, md = DAF[fm], DAT[fm], md.loc[fm].reset_index(drop=True)

    # restrict to common LEVEL range
    lvmask = (md["level"] >= LEVEL_MIN) & (md["level"] <= LEVEL_MAX)
    n_outside = int((~lvmask).sum())
    lvmask = lvmask.to_numpy()
    DAF, DAT, md = DAF[lvmask], DAT[lvmask], md.loc[lvmask].reset_index(drop=True)
    n_final = len(md)

    info = {
        "dir": str(shifts_dir), "n_loaded": n_loaded,
        "n_nonfinite_dropped": n_nonfinite, "n_outside_L1_7_dropped": n_outside,
        "n_final": n_final,
        "levels": sorted(md["level"].unique().tolist()),
        "subjects": int(md["subject"].nunique()),
        "units_ge_minN": int((md["unit"].value_counts() >= sa.MIN_N).sum()),
    }

    arrs = {"DAT": DAT, "DAF": DAF}
    results = {}
    for mode, key in MODES:
        DA = arrs[key].astype(np.float32)
        mu = DA.mean(axis=0, keepdims=True)          # per-pilot self-mean
        DA_c = DA - mu                                # CENTERED (primary)
        DAn = sa.normalize_members(DA_c)              # precompute once
        for col in COLS:
            vc = md[col].value_counts()
            if col == "unit":
                order = sorted([g for g in vc.index if vc[g] >= sa.MIN_N])
            elif col == "level":
                order = sorted(vc.index.tolist(), key=lambda x: int(x))
            else:
                order = sorted(vc.index.tolist())
            if len(order) < 2:
                results[(mode, col)] = None
                continue
            idxg = {g: md.index[md[col] == g].to_numpy() for g in order}
            cents = sa.centroids(DA_c, idxg)
            S = sa.sim_matrix(cents, order)
            wm, bm, gap, _ = sa.within_between(DA_c, idxg, cents, order, DAn=DAn)
            labels = md[col].to_numpy()
            keep = np.isin(labels, order)
            # per-grouping permutation budget (subject/level=1000, unit=200)
            sa.N_PERM = PERM_BY_COL.get(col, 1000)
            p = sa.perm_pvalue(DA_c[keep], labels[keep], order, gap,
                               DAn=DAn[keep])
            # off-diagonal mean (supporting)
            n = len(order)
            od = [S[a, b] for a in range(n) for b in range(a + 1, n)]
            offdiag_mean = float(np.mean(od)) if od else float("nan")
            entry = {"G": n, "group_sizes": {str(g): int(len(idxg[g])) for g in order},
                     "within": wm, "between": bm, "gap": gap, "p": p,
                     "n_perm": sa.N_PERM, "offdiag_mean": offdiag_mean}

            if col == "level":
                levs = [int(x) for x in order]
                pc, pnd = [], []
                for a in range(n):
                    for b in range(a + 1, n):
                        pc.append(S[a, b]); pnd.append(-abs(levs[a] - levs[b]))
                entry["level_ord_rho"] = sa.spearman(pc, pnd)
            results[(mode, col)] = entry
        del DA, DA_c, DAn
    del DAF, DAT
    return info, results


def fmt(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--max-n", type=int, default=None, help="smoke: limit .pt per pilot")
    ap.add_argument("--estimate-only", action="store_true",
                    help="run tiny n-perm to project wall-time, then exit")
    args = ap.parse_args()

    if args.estimate_only:
        sa.N_PERM = 10
        t0 = time.time()
        info1, res1 = metrics_for_pilot(PILOT1_DIR, args.max_n)
        dt = time.time() - t0
        # 6 (mode×col) blocks per pilot, 2 pilots; perm time ~ linear in N_PERM
        per_perm = dt / max(1, 10 * 6)  # rough per-perm-per-block sec (pilot1 only here)
        proj = per_perm * args.n_perm * 6 * 2 + dt  # +load overhead approx
        print(f"[ESTIMATE] pilot1 load+10perm×6blocks = {dt:.1f}s")
        print(f"[ESTIMATE] projected full run (n_perm={args.n_perm}, both pilots) "
              f"~ {proj/60:.1f} min (rough)")
        return

    sa.N_PERM = args.n_perm
    print(f"[run] N_PERM={sa.N_PERM}, max_n={args.max_n}")
    t0 = time.time()
    print("=== PILOT1 ===")
    info1, res1 = metrics_for_pilot(PILOT1_DIR, args.max_n)
    print(f"  pilot1 done ({time.time()-t0:.0f}s)")
    print("=== PILOT2 ===")
    info2, res2 = metrics_for_pilot(PILOT2_DIR, args.max_n)
    print(f"  both done ({time.time()-t0:.0f}s)")

    L = []
    L.append("# Track A — Replication Comparison: pilot1 vs pilot2 (RECOMPUTED)")
    L.append("")
    L.append("> **Framing.** group-centroid cosine, global-mean centering, level "
             "ordinality, and the label-permutation test are OUR diagnostic for "
             "\"does ΔA carry (subject/level) structure?\" — NOT the NAIT paper's "
             "method. NAIT's PCA-direction scoring is applied later in the "
             "curriculum-direction stage. Track C (supervised difficulty "
             "direction) is NAIT-inspired but supervised.")
    L.append("")
    L.append("**Replication design:** each pilot centered by its OWN mean "
             "(μ_pilot1 / μ_pilot2); permutations shuffle labels WITHIN each "
             "pilot. No pooling / no cross-centering (that is Track C only).")
    L.append("")
    L.append(f"**N_PERM (per grouping)** = subject:{PERM_BY_COL['subject']}, "
             f"level:{PERM_BY_COL['level']}, unit:{PERM_BY_COL['unit']} "
             "(unit has ~50 groups => costlier per perm; 200 gives p-resolution "
             "0.005, ample for a descriptive grouping). metric source = "
             "recomputed via similarity_analysis.py functions (identical method).")

    L.append("")
    # N-filter transparency
    L.append("## Population (filters applied identically to both)")
    L.append("")
    L.append("| | pilot1 | pilot2 |")
    L.append("|---|---|---|")
    L.append(f"| shifts dir | `{Path(info1['dir']).relative_to(ACT)}` | `{Path(info2['dir']).relative_to(ACT)}` |")
    L.append(f"| .pt loaded | {info1['n_loaded']} | {info2['n_loaded']} |")
    L.append(f"| non-finite ΔA dropped | {info1['n_nonfinite_dropped']} | {info2['n_nonfinite_dropped']} |")
    L.append(f"| outside L1–L7 dropped | {info1['n_outside_L1_7_dropped']} | {info2['n_outside_L1_7_dropped']} |")
    L.append(f"| **final N (L1–L7)** | **{info1['n_final']}** | **{info2['n_final']}** |")
    L.append(f"| levels | {info1['levels']} | {info2['levels']} |")
    L.append(f"| subjects | {info1['subjects']} | {info2['subjects']} |")
    L.append(f"| units (n≥{sa.MIN_N}) | {info1['units_ge_minN']} | {info2['units_ge_minN']} |")
    L.append("")
    L.append("> Note: the older pilot1 report quoted N=1541 because it was run on "
             "an earlier 1541-file snapshot; the loader applies no content filter, "
             "so current N = loadable .pt count. This recompute supersedes it.")
    L.append("")

    # Main comparison table (raw numbers — verdict is reference only)
    L.append("## Main comparison (within / between / gap / perm-p)")
    L.append("")
    L.append("| MODE | group | G(p1/p2) | within p1/p2 | between p1/p2 | "
             "**gap** p1/p2 (Δ) | perm-p p1/p2 | offdiag p1/p2 | levelρ p1/p2 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for mode, _ in MODES:
        for col in COLS:
            e1 = res1.get((mode, col)); e2 = res2.get((mode, col))
            if e1 is None and e2 is None:
                continue
            G = f"{e1['G'] if e1 else '-'}/{e2['G'] if e2 else '-'}"
            wi = f"{fmt(e1['within']) if e1 else '-'} / {fmt(e2['within']) if e2 else '-'}"
            be = f"{fmt(e1['between']) if e1 else '-'} / {fmt(e2['between']) if e2 else '-'}"
            if e1 and e2:
                dgap = e2['gap'] - e1['gap']
                ga = f"**{fmt(e1['gap'])} / {fmt(e2['gap'])}** (Δ{fmt(dgap)})"
            else:
                ga = f"{fmt(e1['gap']) if e1 else '-'} / {fmt(e2['gap']) if e2 else '-'}"
            pp = f"{fmt(e1['p'],4,False) if e1 else '-'} / {fmt(e2['p'],4,False) if e2 else '-'}"
            of = f"{fmt(e1['offdiag_mean']) if e1 else '-'} / {fmt(e2['offdiag_mean']) if e2 else '-'}"
            lr1 = e1.get('level_ord_rho') if e1 else None
            lr2 = e2.get('level_ord_rho') if e2 else None
            lr = (f"{fmt(lr1)} / {fmt(lr2)}"
                  if (lr1 is not None or lr2 is not None) else "—")
            L.append(f"| {mode} | {col} | {G} | {wi} | {be} | {ga} | {pp} | {of} | {lr} |")
    L.append("")

    # Reference-only auto verdict (NOT a conclusion)
    L.append("## Reference heuristic (NOT a conclusion — judge from raw numbers above)")
    L.append("")
    L.append("Screening rule (loose): gap same sign + |Δgap|/max(|gap|) < 0.5 + "
             "both perm-p < 0.05; level: ordinality ρ same sign. "
             "`|Δgap|/max<0.5` is permissive — final call is human, from the "
             "raw within/between/gap/perm-p columns.")
    L.append("")
    for mode, _ in MODES:
        for col in COLS:
            e1 = res1.get((mode, col)); e2 = res2.get((mode, col))
            if not (e1 and e2):
                continue
            same_sign = np.sign(e1['gap']) == np.sign(e2['gap'])
            denom = max(abs(e1['gap']), abs(e2['gap']), 1e-9)
            close = abs(e2['gap'] - e1['gap']) / denom < 0.5
            sig = (e1['p'] < 0.05) and (e2['p'] < 0.05)
            flags = []
            flags.append("gap同부호" if same_sign else "gap부호반전")
            flags.append("크기근접" if close else "크기차이")
            flags.append("둘다유의" if sig else "유의성부족")
            verdict = "PASS(참고)" if (same_sign and close and sig) else "재검토"
            extra = ""
            if col == "level":
                lr1 = e1.get('level_ord_rho'); lr2 = e2.get('level_ord_rho')
                if lr1 is not None and lr2 is not None and np.isfinite(lr1) and np.isfinite(lr2):
                    extra = f" | ordinalityρ {'同부호' if np.sign(lr1)==np.sign(lr2) else '부호반전'} ({lr1:+.2f}/{lr2:+.2f})"
            L.append(f"- [{mode}/{col}] {verdict}: " + ", ".join(flags) + extra)
    L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
