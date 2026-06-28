#!/usr/bin/env python3
"""
make_pilot_report.py
====================
CPU post-processing for a thinking-mode extraction run (smoke2 or pilot).

1. merges pilot_meta_rank*.jsonl  -> pilot_meta.parquet
2. loads each shifts/{id}.pt, stacks dA_faithful / dA_thinking
3. joins labels, writes REPORT_*.md with:
     - extraction health (finish_reason, think_status, OOM/error counts)
     - is_correct rate overall + by level/subject
     - ΔA signal: per-level mean |dA| (L2), Spearman ρ(|dA|, level),
       ρ(|dA|, r1_cot_token_count), correct-vs-incorrect |dA| gap
     - global PC1 variance-explained of dA_faithful (sanity vs prior PC1 finding)

Usage:
  PY make_pilot_report.py --output-dir <dir> [--tag smoke2|pilot]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path("/scratch/lami2026/personal/jimin_2782")
LABELS = BASE / "src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet"


def spearman(a, b) -> float:
    a = pd.Series(a).rank()
    b = pd.Series(b).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    shifts_dir = out_dir / "shifts"

    # ── merge meta jsonl ────────────────────────────────────────────────────
    rows = []
    for jf in sorted(out_dir.glob("pilot_meta_rank*.jsonl")):
        for line in jf.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    meta = pd.DataFrame(rows)
    if "problem_id" in meta.columns:
        meta = meta.drop_duplicates("problem_id", keep="last")
    meta_pq = out_dir / "pilot_meta.parquet"
    meta.to_parquet(meta_pq, index=False)

    n = len(meta)
    lines = [f"# Extraction Report — {args.tag}", ""]
    lines.append(f"- meta rows: **{n}** (output_dir=`{out_dir}`)")
    if "status" in meta.columns:
        lines.append(f"- status counts: {meta['status'].value_counts().to_dict()}")
    ok = meta[meta.get("status", "ok") == "ok"] if "status" in meta.columns else meta

    pt_files = sorted(shifts_dir.glob("*.pt"))
    lines.append(f"- shifts/*.pt files: **{len(pt_files)}**")
    lines.append("")

    if "finish_reason" in ok.columns:
        lines.append(f"- finish_reason: {ok['finish_reason'].value_counts().to_dict()}")
    if "think_status" in ok.columns:
        lines.append(f"- think_status: {ok['think_status'].value_counts().to_dict()}")
    if "truncated" in ok.columns:
        lines.append(f"- truncated: {int(ok['truncated'].sum())}/{len(ok)}")
    lines.append("")

    # ── is_correct ──────────────────────────────────────────────────────────
    if "is_correct" in ok.columns:
        ic = ok.dropna(subset=["is_correct"])
        if len(ic):
            lines.append(f"## is_correct (n scored = {len(ic)})")
            lines.append(f"- overall correct rate: **{ic['is_correct'].mean():.3f}**")
            if "level" in ic.columns:
                by_l = ic.groupby("level")["is_correct"].agg(["mean", "count"])
                lines.append("\n### by level\n")
                lines.append(by_l.to_string())
                lines.append(f"\n- ρ(is_correct, level) = "
                             f"{spearman(ic['is_correct'].astype(float), ic['level']):.3f} "
                             f"(expect NEGATIVE: harder → lower)")
            if "subject" in ic.columns:
                by_s = ic.groupby("subject")["is_correct"].agg(["mean", "count"])
                lines.append("\n### by subject\n")
                lines.append(by_s.to_string())
            lines.append("")

    # ── load ΔA tensors ───────────────────────────────────────────────────
    daf, dat, ids, levels, cots, corrects = [], [], [], [], [], []
    for pf in pt_files:
        try:
            d = torch.load(pf, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not bool(d.get("think_valid", False)):
            # still keep faithful (always valid); thinking may be zero
            pass
        daf.append(d["dA_faithful"].norm(dim=1).numpy())   # (L,) per-layer L2
        dat.append(d["dA_thinking"].norm(dim=1).numpy())
        ids.append(d["problem_id"])
        levels.append(int(d.get("level", -1)))
        cots.append(float(d.get("r1_cot_token_count", -1) or -1))
        corrects.append(d.get("is_correct", None))

    if daf:
        DAF = np.stack(daf)           # (N, L)
        DAT = np.stack(dat)
        levels = np.array(levels)
        cots = np.array(cots)
        faithful_mag = DAF.mean(axis=1)   # mean over layers
        think_mag = DAT.mean(axis=1)

        lines.append(f"## ΔA signal (N={len(ids)}, layers={DAF.shape[1]})")
        lines.append(f"- mean |dA_faithful| (layer-avg): {faithful_mag.mean():.3f}")
        lines.append(f"- mean |dA_thinking| (layer-avg): {think_mag.mean():.3f}")
        lines.append(f"- ρ(|dA_faithful|, level)          = {spearman(faithful_mag, levels):.3f}")
        lines.append(f"- ρ(|dA_faithful|, r1_cot_tokens)  = {spearman(faithful_mag, cots):.3f}")
        lines.append(f"- ρ(|dA_thinking|, level)          = {spearman(think_mag, levels):.3f}")
        lines.append(f"- ρ(|dA_thinking|, r1_cot_tokens)  = {spearman(think_mag, cots):.3f}")

        # per-level magnitude
        dfm = pd.DataFrame({"level": levels, "faithful": faithful_mag, "thinking": think_mag})
        lines.append("\n### |dA| by level (layer-avg mean)\n")
        lines.append(dfm.groupby("level").mean().to_string())

        # correct vs incorrect gap
        cor = np.array([c if isinstance(c, bool) else None for c in corrects], dtype=object)
        mask = np.array([c is not None for c in cor])
        if mask.sum() > 4:
            cb = np.array([bool(x) for x in cor[mask]])
            fm = faithful_mag[mask]
            if cb.any() and (~cb).any():
                lines.append("\n### correct vs incorrect |dA_faithful|")
                lines.append(f"- correct  : mean={fm[cb].mean():.3f} (n={cb.sum()})")
                lines.append(f"- incorrect: mean={fm[~cb].mean():.3f} (n={(~cb).sum()})")

        # global PC1 variance explained on dA_faithful (concat layers? use layer-avg vec)
        # use full (N, L) matrix centered
        X = DAF - DAF.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(X, compute_uv=False)
            ev = (s ** 2)
            pc1 = ev[0] / ev.sum()
            lines.append(f"\n- global PC1 var-explained (|dA_faithful| layer profile): {pc1:.3f}")
        except Exception:
            pass
        lines.append("")

    report = out_dir / f"REPORT_{args.tag}.md"
    report.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print(f"[OK] wrote {report}")
    print("\n".join(str(x) for x in lines[:40]))


if __name__ == "__main__":
    main()
