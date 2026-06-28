#!/usr/bin/env python3
"""
compute_summary.py
==================
Verification checklist & stats from pass_rate parquet.
Run after pass_rate_measurement.py completes (CPU-only, no GPU).

Usage:
    python compute_summary.py \
        --input_path outputs/pass_rate_pilot_2666.parquet \
        --output_txt outputs/pass_rate_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Pass rate summary stats")
    p.add_argument("--input_path", type=str,
                   default="outputs/pass_rate_pilot_2666.parquet")
    p.add_argument("--output_txt", type=str,
                   default="outputs/pass_rate_summary.txt")
    return p.parse_args()


def fmt(val, pct_of=None, fmt_str=".4f"):
    s = f"{val:{fmt_str}}"
    if pct_of is not None and pct_of > 0:
        s += f"  ({val/pct_of*100:.1f}%)"
    return s


def compute_and_print(df: pd.DataFrame, out_lines: list):
    def line(s=""):
        print(s)
        out_lines.append(s)

    N = len(df)

    line("=" * 70)
    line("PASS RATE MEASUREMENT — VERIFICATION CHECKLIST")
    line("=" * 70)

    # ── 1. Basic counts
    line(f"\n[1] BASIC COUNTS")
    line(f"  Total samples         : {N}  (expected: 2666)")
    line(f"  N rollouts per sample : {df['pass_count'].max()} (from max pass_count)")

    # ── 2. Pass rate distribution (bucket counts)
    line(f"\n[2] PASS RATE DISTRIBUTION (k/8 buckets)")
    for k in range(9):
        rate = k / 8
        cnt  = ((df["pass_count"] == k)).sum()
        line(f"  pass={k}/8  (rate={rate:.4f}) : {cnt:5d}  ({cnt/N*100:.1f}%)")

    # ── 3. Summary stats
    pr = df["pass_rate"]
    line(f"\n[3] SUMMARY STATISTICS")
    line(f"  Mean pass rate        : {pr.mean():.4f}")
    line(f"  Median pass rate      : {pr.median():.4f}")
    line(f"  Std pass rate         : {pr.std():.4f}")
    line(f"  Min pass rate         : {pr.min():.4f}")
    line(f"  Max pass rate         : {pr.max():.4f}")

    n_zero = (df["pass_rate"] == 0.0).sum()
    n_full = (df["pass_rate"] == 1.0).sum()
    n_part = N - n_zero - n_full
    line(f"\n[4] EXTREME BUCKETS")
    line(f"  Pass=0   (all wrong)  : {n_zero:5d}  ({n_zero/N*100:.1f}%)")
    line(f"  Pass=8/8 (all right)  : {n_full:5d}  ({n_full/N*100:.1f}%)")
    line(f"  Partial  (1-7/8)      : {n_part:5d}  ({n_part/N*100:.1f}%)")

    # ── 5. Truncation stats
    tc = df["truncation_count"]
    line(f"\n[5] TRUNCATION (max_tokens=4096 hit)")
    line(f"  Total truncated rollouts      : {tc.sum()}")
    line(f"  Mean truncated per sample     : {tc.mean():.3f}")
    line(f"  Samples with any truncation   : {(tc > 0).sum()} ({(tc > 0).sum()/N*100:.1f}%)")
    line(f"  Samples all 8 truncated       : {(tc == 8).sum()}")

    # ── 6. Response length stats
    line(f"\n[6] RESPONSE LENGTH (tokens)")
    mrl = df["mean_response_length"]
    mrl_ok  = df["mean_response_length_correct"].dropna()
    mrl_bad = df["mean_response_length_incorrect"].dropna()
    line(f"  Overall mean length (all)     : {mrl.mean():.1f} tokens")
    line(f"  Overall mean length (correct) : {mrl_ok.mean():.1f} tokens")
    line(f"  Overall mean length (wrong)   : {mrl_bad.mean():.1f} tokens")

    # ── 7. GPT-label level breakdown
    line(f"\n[7] GPT-LABEL LEVEL × PASS RATE (difficulty sanity check)")
    for lv in sorted(df["level"].unique()):
        sub = df[df["level"] == lv]
        lv_mean = sub["pass_rate"].mean()
        lv_zero = (sub["pass_rate"] == 0.0).sum()
        lv_full = (sub["pass_rate"] == 1.0).sum()
        line(f"  Level {lv:2d}: n={len(sub):4d}  "
             f"mean_pass={lv_mean:.4f}  "
             f"pass=0: {lv_zero:3d} ({lv_zero/len(sub)*100:.0f}%)  "
             f"pass=8: {lv_full:3d} ({lv_full/len(sub)*100:.0f}%)")

    # ── 8. Subject breakdown
    line(f"\n[8] SUBJECT × PASS RATE")
    subj_stats = df.groupby("subject")["pass_rate"].agg(["mean", "count"]).sort_values("mean")
    for subj, row in subj_stats.iterrows():
        line(f"  {subj:40s} n={int(row['count']):4d}  mean={row['mean']:.4f}")

    # ── 9. Pass=0 analysis (hard tail)
    line(f"\n[9] PASS=0 HARD-TAIL ANALYSIS")
    df_zero = df[df["pass_rate"] == 0.0]
    line(f"  Total pass=0 samples: {len(df_zero)}")

    if "data_source" in df.columns:
        line(f"\n  data_source distribution of pass=0 samples:")
        ds_counts = df_zero["data_source"].value_counts()
        for ds, cnt in ds_counts.items():
            line(f"    {str(ds)[:60]:60s}: {cnt:4d} ({cnt/len(df_zero)*100:.1f}%)")

    line(f"\n  subject distribution of pass=0 samples:")
    sub_counts = df_zero["subject"].value_counts()
    for subj, cnt in sub_counts.items():
        line(f"    {subj:40s}: {cnt:4d} ({cnt/len(df_zero)*100:.1f}%)")

    line(f"\n  level distribution of pass=0 samples:")
    for lv in sorted(df["level"].unique()):
        all_at_lv  = (df["level"] == lv).sum()
        zero_at_lv = (df_zero["level"] == lv).sum()
        pct_of_lv  = zero_at_lv / all_at_lv * 100 if all_at_lv > 0 else 0
        line(f"    Level {lv:2d}: {zero_at_lv:3d} pass=0 out of {all_at_lv:4d} total ({pct_of_lv:.0f}%)")

    # ── 10. GPT level × pass=0 pct
    line(f"\n[10] GPT level × pass=0 rate (key diagnostic)")
    for lv in sorted(df["level"].unique()):
        sub      = df[df["level"] == lv]
        zero_pct = (sub["pass_rate"] == 0.0).sum() / len(sub) * 100
        full_pct = (sub["pass_rate"] == 1.0).sum() / len(sub) * 100
        line(f"  Level {lv:2d}: pass=0 rate={zero_pct:5.1f}%  pass=8 rate={full_pct:5.1f}%")

    line("")
    line("=" * 70)
    line("END OF SUMMARY")
    line("=" * 70)


def main():
    args = parse_args()
    in_path  = Path(args.input_path)
    out_txt  = Path(args.output_txt)

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    print(f"[INFO] Loaded {len(df)} rows from {in_path}")

    out_lines = []
    compute_and_print(df, out_lines)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\n[INFO] Summary saved → {out_txt}")


if __name__ == "__main__":
    main()
