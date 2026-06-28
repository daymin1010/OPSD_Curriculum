"""Post-process labeling output:
  1) Normalize 66 out-of-vocabulary subjects → "Other"
  2) Add problem_id = sha1(problem_text)[:16]
  3) Save → outputs/openthoughts_30k_labels_final.parquet
  4) Build pilot universe candidate (stratified by subject×level, target ~3000)

Strict:
  - Keep original `subject` raw in `subject_raw`
  - Final canonical column `subject` ∈ ALLOWED_SUBJ
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV  = HERE / "outputs" / "openthoughts_30k_labels.csv"
PARQ = HERE / "outputs" / "openthoughts_30k_labels_final.parquet"
PILOT = HERE / "outputs" / "pilot_universe_candidate.parquet"
PILOT_REPORT = HERE / "outputs" / "REPORT_pilot.md"

ALLOWED_SUBJ = {"Algebra","Counting & Probability","Geometry",
                "Intermediate Algebra","Number Theory","Prealgebra",
                "Precalculus","Other"}

# Reproducibility
RNG_SEED = 42
PILOT_TARGET = 3000     # roughly match prior 2666 pilot size
MIN_PER_CELL = 5        # cells smaller than this: take ALL rows
MAX_PER_CELL = 80       # cap per (subject,level) cell so head doesn't dominate


def main():
    df = pd.read_csv(CSV)
    print(f"[load] {len(df):,} rows")

    # --- 1. subject normalization ---
    df["subject_raw"] = df["subject"]
    bad_mask = ~df["subject"].isin(ALLOWED_SUBJ)
    n_bad = int(bad_mask.sum())
    print(f"[normalize] {n_bad} out-of-vocab subjects → 'Other'")
    print(df.loc[bad_mask, "subject_raw"].value_counts().to_string())
    df.loc[bad_mask, "subject"] = "Other"

    # --- 2. problem_id ---
    def sha1_id(s: str) -> str:
        return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]
    df["problem_id"] = df["problem_text"].map(sha1_id)
    n_unique = df["problem_id"].nunique()
    print(f"[id] problem_id unique: {n_unique:,}/{len(df):,}  (dup={len(df)-n_unique})")

    # column order
    cols = (["problem_id","row_index","source","subject","level",
             "subject_raw","problem_text","problem_char_len","problem_qwen_tok_len",
             "r1_cot_token_count","solution_char_len","correct","answer",
             "raw_response","error","finish_reason","prompt_tokens",
             "completion_tokens","latency_s","attempts","model","prompt_sha"])
    df = df[cols]
    df.to_parquet(PARQ, index=False)
    print(f"[save] {PARQ}  ({PARQ.stat().st_size/1024/1024:.1f} MB)")

    # --- 3. cross-tab AFTER normalization ---
    print("\n[xtab] subject × level (after normalization)")
    ct = pd.crosstab(df["subject"], df["level"])
    print(ct.to_string())

    # --- 4. Build pilot universe (stratified) ---
    rng = np.random.default_rng(RNG_SEED)
    picked_ids = []
    diagnostic = []
    for (sub, lv), g in df.groupby(["subject","level"]):
        n = len(g)
        if n <= MIN_PER_CELL:
            take = n
        else:
            # proportional to sqrt(n) capped at MAX
            take = int(min(MAX_PER_CELL, max(MIN_PER_CELL, round(np.sqrt(n)*5))))
            take = min(take, n)
        idx = rng.choice(g.index.values, size=take, replace=False)
        picked_ids.extend(idx.tolist())
        diagnostic.append((sub, lv, n, take))
    pilot = df.loc[picked_ids].copy()

    # rebalance toward PILOT_TARGET: shuffle and slice if oversize, else keep
    pilot = pilot.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    if len(pilot) > PILOT_TARGET:
        pilot = pilot.iloc[:PILOT_TARGET].copy()

    pilot.to_parquet(PILOT, index=False)
    print(f"\n[pilot] saved {len(pilot):,} rows → {PILOT}")
    print("[pilot] subject × level distribution:")
    print(pd.crosstab(pilot["subject"], pilot["level"]).to_string())

    # report
    md = ["# Pilot Universe Candidate", "",
          f"- source CSV : `outputs/openthoughts_30k_labels.csv`",
          f"- size       : **{len(pilot):,}** problems (target {PILOT_TARGET})",
          f"- stratifier : (subject, level), 8×8 = 64 cells (subject ∈ 8 canonical)",
          f"- sampler    : sqrt(n)*5, capped {MIN_PER_CELL}–{MAX_PER_CELL}, seed={RNG_SEED}",
          "",
          "## Subject × Level after normalization (full 29,434)",
          "```", ct.to_string(), "```",
          "",
          "## Pilot subject × level",
          "```", pd.crosstab(pilot["subject"], pilot["level"]).to_string(), "```",
          "",
          "## Per-cell pilot sampling diagnostics",
          "| subject | level | full_n | pilot_n |",
          "|---|---|---:|---:|",
          ]
    for sub,lv,n,take in diagnostic:
        md.append(f"| {sub} | {int(lv)} | {n} | {take} |")
    md.append("")
    md.append("## Length / difficulty sanity in pilot")
    md.append(f"- ρ(level, r1_cot_token_count) in pilot: **{pilot[['level','r1_cot_token_count']].corr(method='spearman').iloc[0,1]:.3f}**")
    md.append(f"- ρ(level, problem_qwen_tok_len)   in pilot: **{pilot[['level','problem_qwen_tok_len']].corr(method='spearman').iloc[0,1]:.3f}**")
    PILOT_REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] {PILOT_REPORT}")


if __name__ == "__main__":
    main()
