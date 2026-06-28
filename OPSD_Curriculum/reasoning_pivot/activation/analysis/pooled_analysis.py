#!/usr/bin/env python3
"""
pooled_analysis.py — Phase 1 POOLED (pilot1 + pilot2) similarity analysis.
==========================================================================
Track C / pooled handling (the OPPOSITE of the Track-A replication in
compare_pilot1_pilot2.py): we MERGE pilot1 and pilot2 into ONE population
(canonical N = 3025, finite) and center by a SINGLE pooled global mean
(μ_pooled), then run the group-similarity diagnostic on the merged set.

POOLED PHILOSOPHY (do NOT confuse with replication):
  - compare_pilot1_pilot2.py asks "does the SAME structure appear
    INDEPENDENTLY in each pilot?" -> per-pilot self-mean, within-pilot perms.
  - HERE we ask "what is the structure of the FULL labelled population?"
    -> concat both pilots, subtract ONE pooled mean μ_pooled, permute labels
    across the whole pooled set. This is the canonical / main analysis.

METHOD NOTE (framing — group-similarity is OUR diagnostic, not NAIT):
  group-centroid layer-averaged cosine, global-mean centering, level
  ordinality and the label-permutation test are a diagnostic WE built to ask
  "does ΔA carry (subject/level/unit) structure?". They are NOT the NAIT
  paper's PCA-direction scoring (applied later in the curriculum-direction
  stage; Track C supervised direction is NAIT-inspired but supervised).

FILTERING:
  - load every *.pt from BOTH pilots (loader applies no content filter);
  - drop rows with any non-finite ΔA (reported; expected 0);
  - subject / unit groupings use the FULL pooled set (all levels);
  - LEVEL grouping is reported TWICE: L1–L8 (full, includes the n=66 L8 cell)
    AND L1–L7 (drops L8) so the L8 effect is isolated. The L8 cell is small
    (n=66) and present in only 5/8 subjects (all from pilot1) => NO standalone
    L8 conclusions.

OUTPUT: analysis/REPORT_pooled_3025.md  (canonical; CPU only, no GPU)
        Does NOT overwrite N_AUDIT.md / REPORT_pilot2_comparison.md / etc.
"""
from __future__ import annotations
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

import similarity_analysis as sa  # reuse the validated metric functions

ANALYSIS = Path(__file__).resolve().parent
ACT = ANALYSIS.parent  # .../reasoning_pivot/activation
OUT = ANALYSIS / "REPORT_pooled_3025.md"

PILOT1_DIR = ACT / "outputs" / "pilot" / "shifts"
PILOT2_DIR = ACT / "outputs" / "pilot2" / "shifts"

MODES = [("THINKING", "DAT"), ("FAITHFUL", "DAF")]
# (col, level_min, level_max, tag) — subject/unit use full range; level twice.
GROUPINGS = [
    ("subject", 1, 8, "subject"),
    ("level", 1, 8, "level (L1-L8, incl. n=66 L8)"),
    ("level", 1, 7, "level (L1-L7, L8 dropped)"),
    ("unit", 1, 8, "unit (subject x level, n>=MIN_N)"),
]

# Per-grouping permutation budget (same convention as compare_pilot1_pilot2.py).
PERM_BY_COL = {"subject": 1000, "level": 1000, "unit": 200}

LAYERS = 36  # ΔA layer count (per-layer cosine averaging dim)


def fast_perm_pvalue(A2, B2, labels, order, observed_gap, n_perm, L=LAYERS):
    """BLAS-vectorized, EXACT-equivalent label-permutation p-value.

    Identical metric to similarity_analysis.perm_pvalue but reformulated so each
    permutation costs two GEMMs instead of a Python per-group fancy-index +
    einsum loop. Uses the identities (per layer l):
      C_g[l]   = (sum_{i in g} ΔA_i[l]) / n_g      -> normalized -> Cn_g[l]
      within_g = mean_l < mean_{i in g} ΔAn_i[l] , Cn_g[l] >
               = (1/(L*n_g)) * < sum_{i in g} ΔAn_i , Cn_g >   (full (l,d) dot)
      between  = mean over g!=h of mean_l cos(Cn_g[l], Cn_h[l])
      gap      = mean_g within_g - mean_{g<h} between   (equal group weighting)
    Group sums of raw (A2) and per-layer-normalized (B2) members are obtained via
    one-hot @ data GEMMs. Groups with <2 members in a permutation are dropped
    (matches the reference). A2,B2: (N, L*D) float32.
    """
    N, M = A2.shape
    D = M // L
    g_of = {g: i for i, g in enumerate(order)}
    base = np.fromiter((g_of[x] for x in labels), dtype=np.int64, count=N)
    G = len(order)
    eye = np.eye(G, dtype=np.float32)
    ge = 0
    log_every = max(1, n_perm // 10)
    for it in range(n_perm):
        perm = sa.rng.permutation(base)

        onehot = eye[perm]                 # (N,G)
        counts = onehot.sum(0)             # (G,)
        valid = counts >= 2
        if int(valid.sum()) < 2:
            continue
        Sraw = (onehot.T @ A2)[valid]      # (Gv, M)  raw group sums
        Snorm = (onehot.T @ B2)[valid]     # (Gv, M)  normalized-member group sums
        cnt = counts[valid]
        Gv = Sraw.shape[0]
        Sraw3 = Sraw.reshape(Gv, L, D)
        nrm = np.linalg.norm(Sraw3, axis=2, keepdims=True) + 1e-8
        Cn3 = Sraw3 / nrm                  # per-layer normalized centroids
        Snorm3 = Snorm.reshape(Gv, L, D)
        within_g = (Snorm3 * Cn3).sum(axis=(1, 2)) / (L * cnt)
        within_mean = float(within_g.mean())
        Spair = np.zeros((Gv, Gv), dtype=np.float32)
        for l in range(L):
            Cl = Cn3[:, l, :]
            Spair += Cl @ Cl.T
        Spair /= L
        iu = np.triu_indices(Gv, 1)
        between_mean = float(Spair[iu].mean())
        if (within_mean - between_mean) >= observed_gap:
            ge += 1
        if (it + 1) % log_every == 0:
            print(f"    perm {it+1}/{n_perm} (ge={ge})", flush=True)
    return (ge + 1) / (n_perm + 1)



def finite_mask(*arrs):
    m = None
    for A in arrs:
        f = np.isfinite(A.reshape(A.shape[0], -1)).all(axis=1)
        m = f if m is None else (m & f)
    return m


def load_pooled(max_n=None):
    """Load pilot1 + pilot2, drop non-finite, concat into ONE pooled set.
    Returns (DAF, DAT, md) with md carrying a `_pilot` provenance column."""
    DAF1, DAT1, md1 = sa.load_pilot(PILOT1_DIR, max_n)
    DAF2, DAT2, md2 = sa.load_pilot(PILOT2_DIR, max_n)
    md1 = md1.copy(); md1["_pilot"] = "pilot1"
    md2 = md2.copy(); md2["_pilot"] = "pilot2"

    DAF = np.concatenate([DAF1, DAF2], axis=0)
    DAT = np.concatenate([DAT1, DAT2], axis=0)
    md = pd.concat([md1, md2], ignore_index=True)
    del DAF1, DAF2, DAT1, DAT2
    gc.collect()

    n_loaded = len(md)
    fm = finite_mask(DAF, DAT)
    n_nonfinite = int((~fm).sum())
    DAF, DAT, md = DAF[fm], DAT[fm], md.loc[fm].reset_index(drop=True)
    print(f"[pooled] loaded={n_loaded} non-finite-dropped={n_nonfinite} "
          f"finite_N={len(md)}", flush=True)
    return DAF, DAT, md, {"n_loaded": n_loaded, "n_nonfinite": n_nonfinite,
                          "n_final": len(md)}


def one_grouping(DA_c, DAn, md, col, lv_min, lv_max):
    """Compute within/between/gap/perm-p (+ level ordinality) for one grouping
    on already-centered data DA_c (per-layer L2-normed DAn precomputed).
    Restricts to [lv_min, lv_max] when col != already full. Returns entry|None."""
    lvmask = ((md["level"] >= lv_min) & (md["level"] <= lv_max)).to_numpy()
    sub_md = md.loc[lvmask].reset_index(drop=True)
    sub_DA = DA_c[lvmask]
    sub_DAn = DAn[lvmask]

    vc = sub_md[col].value_counts()
    if col == "unit":
        order = sorted([g for g in vc.index if vc[g] >= sa.MIN_N])
    elif col == "level":
        order = sorted(vc.index.tolist(), key=lambda x: int(x))
    else:
        order = sorted(vc.index.tolist())
    if len(order) < 2:
        return None

    idxg = {g: sub_md.index[sub_md[col] == g].to_numpy() for g in order}
    cents = sa.centroids(sub_DA, idxg)
    S = sa.sim_matrix(cents, order)
    wm, bm, gap, _ = sa.within_between(sub_DA, idxg, cents, order, DAn=sub_DAn)

    labels = sub_md[col].to_numpy()
    keep = np.isin(labels, order)
    n_perm = PERM_BY_COL.get(col, 1000)
    nk = int(keep.sum())
    Ak = np.ascontiguousarray(sub_DA[keep], dtype=np.float32).reshape(nk, -1)
    Bk = np.ascontiguousarray(sub_DAn[keep], dtype=np.float32).reshape(nk, -1)
    p = fast_perm_pvalue(Ak, Bk, labels[keep], order, gap, n_perm)
    del Ak, Bk


    n = len(order)
    od = [S[a, b] for a in range(n) for b in range(a + 1, n)]
    offdiag_mean = float(np.mean(od)) if od else float("nan")

    entry = {"G": n, "n_rows": int(len(sub_md)),
             "group_sizes": {str(g): int(len(idxg[g])) for g in order},
             "within": wm, "between": bm, "gap": gap, "p": p,
             "n_perm": n_perm, "offdiag_mean": offdiag_mean,

             "order": [str(o) for o in order]}
    if col == "level":
        levs = [int(x) for x in order]
        pc, pnd = [], []
        for a in range(n):
            for b in range(a + 1, n):
                pc.append(S[a, b]); pnd.append(-abs(levs[a] - levs[b]))
        entry["level_ord_rho"] = sa.spearman(pc, pnd)
    return entry


def fmt(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None,
                    help="smoke: limit .pt per pilot")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="override ALL per-grouping perm budgets (smoke)")
    ap.add_argument("--estimate-only", action="store_true",
                    help="run tiny n-perm to project wall-time, then exit")
    args = ap.parse_args()

    if args.n_perm is not None:
        for k in PERM_BY_COL:
            PERM_BY_COL[k] = args.n_perm
    if args.estimate_only:
        for k in PERM_BY_COL:
            PERM_BY_COL[k] = 10

    t0 = time.time()
    DAF, DAT, md, ninfo = load_pooled(args.max_n)
    print(f"[load] done ({time.time()-t0:.0f}s)", flush=True)

    arrs = {"DAT": DAT, "DAF": DAF}
    results = {}  # (mode, grp_tag) -> entry
    for mode, key in MODES:
        DA = arrs[key].astype(np.float32)
        mu = DA.mean(axis=0, keepdims=True)   # SINGLE pooled mean (per layer)
        DA_c = DA - mu
        del DA
        DAn = sa.normalize_members(DA_c)
        print(f"[{mode}] centered by pooled mean; running groupings...", flush=True)
        for col, lv_min, lv_max, tag in GROUPINGS:
            te = time.time()
            entry = one_grouping(DA_c, DAn, md, col, lv_min, lv_max)
            results[(mode, tag, col)] = entry
            g = entry["G"] if entry else "-"
            print(f"  [{mode}/{tag}] G={g} ({time.time()-te:.0f}s)", flush=True)
        del DA_c, DAn
        gc.collect()

    if args.estimate_only:
        print(f"[ESTIMATE] full pipeline @ n_perm=10 took {time.time()-t0:.0f}s; "
              f"scale perm-bound blocks by target/10.")
        return

    # ───────────── L8 / pooled provenance facts (for the report) ─────────────
    l8 = md[md["level"] == 8]
    l8_total = int(len(l8))
    l8_by_subject = l8["subject"].value_counts()
    l8_by_pilot = l8["_pilot"].value_counts()
    lvl_counts = md["level"].value_counts().sort_index()
    pilot_counts = md["_pilot"].value_counts()

    # ───────────────────────────── write report ─────────────────────────────
    L = []
    L.append("# Phase 1 — POOLED (pilot1 + pilot2) ΔA Group-Similarity (canonical)")
    L.append("")
    L.append("> **Framing.** group-centroid layer-averaged cosine, pooled-mean "
             "centering, level ordinality, and the label-permutation test are "
             "OUR diagnostic for \"does ΔA carry (subject/level/unit) "
             "structure?\" — NOT the NAIT paper's PCA-direction scoring (applied "
             "later in the curriculum-direction stage; Track C supervised "
             "direction is NAIT-inspired but supervised).")
    L.append("")
    L.append("**Pooled design (OPPOSITE of the Track-A replication).** Both "
             "pilots are MERGED into one population and centered by a SINGLE "
             "pooled global mean (μ_pooled, per layer). Permutations shuffle "
             "labels across the WHOLE pooled set. This is the canonical / main "
             "analysis. (Per-pilot self-centering + within-pilot permutation is "
             "the replication track in `REPORT_pilot2_comparison.md` — do not "
             "confuse the two handlings.)")
    L.append("")
    L.append(f"**N_PERM (per grouping):** subject:{PERM_BY_COL['subject']}, "
             f"level:{PERM_BY_COL['level']}, unit:{PERM_BY_COL['unit']}. "
             "metric source = recomputed via similarity_analysis.py functions "
             "(identical method to the per-pilot reports).")
    L.append("")

    # Population / N transparency
    L.append("## Population (canonical N)")
    L.append("")
    L.append(f"- raw .pt loaded (pilot1+pilot2) = **{ninfo['n_loaded']}**")
    L.append(f"- non-finite ΔA dropped = **{ninfo['n_nonfinite']}** (expected 0)")
    L.append(f"- **finite pooled N = {ninfo['n_final']}** (canonical; '3000' is a "
             "nickname only — always report raw / finite / per-filter analysis-N)")
    L.append(f"- provenance: " + ", ".join(f"{k}={int(v)}" for k, v in pilot_counts.items()))
    L.append("")
    L.append("level counts (pooled finite):")
    L.append("")
    L.append("| level | " + " | ".join(str(int(k)) for k in lvl_counts.index) + " |")
    L.append("|---|" + "|".join("---" for _ in lvl_counts.index) + "|")
    L.append("| n | " + " | ".join(str(int(v)) for v in lvl_counts.values) + " |")
    L.append("")
    L.append("### L8 caveat (read before any L8 reading)")
    L.append("")
    L.append(f"- L8 total = **{l8_total}**; provenance: " +
             ", ".join(f"{k}={int(v)}" for k, v in l8_by_pilot.items()) +
             " (i.e. **L8 is entirely from pilot1**).")
    L.append(f"- L8 exists in only {l8_by_subject.shape[0]}/8 subjects: " +
             ", ".join(f"{s}:{int(c)}" for s, c in l8_by_subject.items()) + ".")
    L.append("- Algebra / Prealgebra / Precalculus have L8 = 0 (handoff quirk #4).")
    L.append("- => **NO standalone L8 conclusions** (n small + subject imbalance). "
             "We report level grouping BOTH with (L1–L8) and without (L1–L7) the "
             "L8 cell so its effect is isolated.")
    L.append("")

    # Main table
    L.append("## Main results (within / between / gap / perm-p)")
    L.append("")
    L.append("| MODE | grouping | G | analysis-N | within | between | "
             "**gap** | perm-p | offdiag | level ρ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for mode, _ in MODES:
        for col, lv_min, lv_max, tag in GROUPINGS:
            e = results.get((mode, tag, col))
            if e is None:
                L.append(f"| {mode} | {tag} | - | - | - | - | - | - | - | - |")
                continue
            lr = e.get("level_ord_rho")
            lr_s = fmt(lr) if lr is not None else "—"
            L.append(f"| {mode} | {tag} | {e['G']} | {e['n_rows']} | "
                     f"{fmt(e['within'])} | {fmt(e['between'])} | "
                     f"**{fmt(e['gap'])}** | {fmt(e['p'], 4, False)} | "
                     f"{fmt(e['offdiag_mean'])} | {lr_s} |")
    L.append("")
    L.append("Interpretation guide: gap = within − between (higher => groups "
             "internally coherent & mutually distinct); perm-p = P(gap_perm ≥ "
             "obs) under label shuffle; level ρ>0 => adjacent levels more "
             "similar (ordinality). Compare the L1–L8 vs L1–L7 level rows to see "
             "how the small L8 cell moves the level structure.")
    L.append("")

    # Per-grouping group sizes (transparency)
    L.append("## Group sizes (per grouping, THINKING; identical to FAITHFUL)")
    L.append("")
    for col, lv_min, lv_max, tag in GROUPINGS:
        e = results.get(("THINKING", tag, col))
        if e is None:
            continue
        L.append(f"- **{tag}** (G={e['G']}, N={e['n_rows']}): " +
                 ", ".join(f"{g}:{n}" for g, n in e["group_sizes"].items()))
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] wrote {OUT}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
