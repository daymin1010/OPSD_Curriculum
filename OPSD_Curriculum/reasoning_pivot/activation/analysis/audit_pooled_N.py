#!/usr/bin/env python3
"""
audit_pooled_N.py — Phase 0 data audit for the pooled (pilot1+pilot2) analysis.
================================================================================
CPU-only. No GPU. Reuses similarity_analysis.load_pilot so the audit sees the
EXACT same arrays the analysis will use.

Answers, with hard numbers, three questions before any analysis:
  Q1  N reconciliation: how many .pt load per pilot, how many have non-finite
      ΔA (NaN/Inf), and therefore the finite analysis-N. (Resolves the
      "1541 vs 1608" question: if pilot1 non-finite ≈ 67 => the slide's 1541
      was the finite subset of the SAME population (hypothesis a); if ≈ 0 =>
      the population grew after the slide (hypothesis b).)
  Q2  L8 availability: exact count of level-8 rows (overall + per pilot +
      per subject). Drives the conditional L8-extraction decision.
  Q3  pooled subject x level crosstab + finite pooled N.

Also prints pilot1 .pt mtime span (earliest/latest) as a cross-check for the
(a)/(b) question (a late cluster of files => population grew).

OUTPUT: analysis/N_AUDIT.md  (+ stdout)
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import similarity_analysis as sa  # same dir; reuse validated loader

ANALYSIS = Path(__file__).resolve().parent
ACT = ANALYSIS.parent
OUT = ANALYSIS / "N_AUDIT.md"

PILOT1_DIR = ACT / "outputs" / "pilot" / "shifts"
PILOT2_DIR = ACT / "outputs" / "pilot2" / "shifts"

# Threshold below which we consider L8 "insufficient" and trigger extraction.
L8_MIN_USABLE = 20


def finite_mask(*arrs):
    m = None
    for A in arrs:
        f = np.isfinite(A.reshape(A.shape[0], -1)).all(axis=1)
        m = f if m is None else (m & f)
    return m


def mtime_span(shifts_dir: Path):
    pts = list(shifts_dir.glob("*.pt"))
    if not pts:
        return None
    ts = sorted(os.path.getmtime(p) for p in pts)
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    # crude "late cluster" detector: how many files in last 5% of the time span
    span = ts[-1] - ts[0]
    late = sum(1 for t in ts if (ts[-1] - t) <= max(1.0, 0.0) and span > 0)
    return {"n": len(ts), "earliest": fmt(ts[0]), "latest": fmt(ts[-1]),
            "span_hours": round(span / 3600.0, 2)}


def audit_pilot(name: str, shifts_dir: Path):
    print(f"=== loading {name}: {shifts_dir} ===", flush=True)
    t0 = time.time()
    DAF, DAT, md = sa.load_pilot(shifts_dir, None)
    n_loaded = len(md)
    fmF = finite_mask(DAF)
    fmT = finite_mask(DAT)
    fm = fmF & fmT
    info = {
        "name": name,
        "dir": str(shifts_dir),
        "n_loaded": n_loaded,
        "nonfinite_DAF": int((~fmF).sum()),
        "nonfinite_DAT": int((~fmT).sum()),
        "nonfinite_either": int((~fm).sum()),
        "finite_N": int(fm.sum()),
        "mtime": mtime_span(shifts_dir),
    }
    md_f = md.loc[fm].reset_index(drop=True)
    print(f"  {name}: loaded={n_loaded} finite={info['finite_N']} "
          f"(nonfinite either={info['nonfinite_either']}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return info, md_f


def main():
    rows = []
    info1, md1 = audit_pilot("pilot1", PILOT1_DIR)
    info2, md2 = audit_pilot("pilot2", PILOT2_DIR)

    md1["_pilot"] = "pilot1"
    md2["_pilot"] = "pilot2"
    pooled = pd.concat([md1, md2], ignore_index=True)
    pooled_N = len(pooled)

    # crosstabs
    ct = pd.crosstab(pooled["subject"], pooled["level"])
    lvl_counts = pooled["level"].value_counts().sort_index()
    l8_total = int((pooled["level"] == 8).sum())
    l8_p1 = int((md1["level"] == 8).sum())
    l8_p2 = int((md2["level"] == 8).sum())
    l8_by_subj = pooled.loc[pooled["level"] == 8, "subject"].value_counts()

    trigger = l8_total < L8_MIN_USABLE

    # ---- write report ----
    L = []
    L.append("# Phase 0 — N / L8 Audit (pooled pilot1+pilot2)")
    L.append("")
    L.append("CPU-only. Reuses `similarity_analysis.load_pilot` (identical arrays "
             "to the analysis). Non-finite = any NaN/Inf in DAF or DAT.")
    L.append("")
    L.append("## Q1 — N reconciliation (raw / finite)")
    L.append("")
    L.append("| | .pt loaded | non-finite (DAF) | non-finite (DAT) | non-finite (either) | **finite N** |")
    L.append("|---|---|---|---|---|---|")
    for inf in (info1, info2):
        L.append(f"| {inf['name']} | {inf['n_loaded']} | {inf['nonfinite_DAF']} | "
                 f"{inf['nonfinite_DAT']} | {inf['nonfinite_either']} | **{inf['finite_N']}** |")
    L.append(f"| **pooled** | {info1['n_loaded']+info2['n_loaded']} | "
             f"{info1['nonfinite_DAF']+info2['nonfinite_DAF']} | "
             f"{info1['nonfinite_DAT']+info2['nonfinite_DAT']} | "
             f"{info1['nonfinite_either']+info2['nonfinite_either']} | "
             f"**{pooled_N}** |")
    L.append("")
    # (a)/(b) hypothesis call for pilot1
    p1_nf = info1["nonfinite_either"]
    diff_to_1541 = info1["n_loaded"] - 1541
    L.append(f"**pilot1 file mtime span:** {info1['mtime']}")
    L.append("")
    L.append(f"**1541 vs {info1['n_loaded']} call:** pilot1 non-finite(either) = "
             f"**{p1_nf}**, loaded−1541 = **{diff_to_1541}**.")
    if p1_nf == diff_to_1541 and diff_to_1541 > 0:
        L.append(f"→ **(a) NaN/Inf hypothesis CONFIRMED**: the {diff_to_1541} files "
                 f"beyond 1541 are exactly the non-finite ΔA rows. The slide's 1541 "
                 f"was the finite subset of the SAME {info1['n_loaded']}-file population.")
    elif p1_nf <= 2:
        L.append(f"→ **(b) population-growth hypothesis likely**: almost all "
                 f"{info1['n_loaded']} files are finite, so 1541 was an earlier "
                 f"snapshot; current finite N supersedes it. Check mtime span above "
                 f"for a late file cluster.")
    else:
        L.append(f"→ **mixed/unclear**: non-finite={p1_nf} ≠ (loaded−1541)={diff_to_1541}. "
                 f"Inspect manually; report both numbers, use finite N as canonical.")
    L.append("")
    L.append("## Q2 — L8 availability (drives conditional L8 extraction)")
    L.append("")
    L.append(f"- L8 total (pooled finite): **{l8_total}**  (pilot1={l8_p1}, pilot2={l8_p2})")
    L.append(f"- usability threshold = {L8_MIN_USABLE}")
    L.append(f"- **DECISION: {'TRIGGER L8 extraction (Phase 1.5)' if trigger else 'L8 sufficient — proceed straight to pooled analysis'}**")
    if l8_total > 0:
        L.append("")
        L.append("L8 by subject:")
        for s, c in l8_by_subj.items():
            L.append(f"  - {s}: {c}")
    L.append("")
    L.append("## Q3 — pooled level distribution + subject×level")
    L.append("")
    L.append("level counts (pooled finite):")
    L.append("")
    L.append("| level | " + " | ".join(str(int(k)) for k in lvl_counts.index) + " |")
    L.append("|---|" + "|".join("---" for _ in lvl_counts.index) + "|")
    L.append("| n | " + " | ".join(str(int(v)) for v in lvl_counts.values) + " |")
    L.append("")
    L.append("subject × level crosstab (pooled finite):")
    L.append("")
    L.append("```")
    L.append(ct.to_string())
    L.append("```")
    L.append("")
    L.append("## Canonical N convention")
    L.append("")
    L.append(f"- raw .pt = {info1['n_loaded']+info2['n_loaded']}  /  "
             f"finite pooled N = **{pooled_N}**.")
    L.append("- Always report raw / finite / analysis-N. '3000' is a nickname only.")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n[OK] wrote {OUT}")
    print(f"[DECISION] L8_total={l8_total} -> "
          f"{'TRIGGER extraction' if trigger else 'sufficient'}")


if __name__ == "__main__":
    main()
