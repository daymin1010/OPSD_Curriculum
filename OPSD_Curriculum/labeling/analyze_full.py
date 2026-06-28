"""Full-run sanity check + distribution analysis + REPORT_full.md.

Inputs:  outputs/openthoughts_30k_labels.csv
Outputs: outputs/REPORT_full.md
         (and prints summary to stdout)
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV  = HERE / "outputs" / "openthoughts_30k_labels.csv"
REPORT = HERE / "outputs" / "REPORT_full.md"

ALLOWED_SUBJ = {"Algebra","Counting & Probability","Geometry",
                "Intermediate Algebra","Number Theory","Prealgebra",
                "Precalculus","Other"}

def main():
    print(f"[load] {CSV}")
    df = pd.read_csv(CSV)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    issues = []
    md = ["# OpenThoughts 30K — Full Labeling Report", ""]
    md.append(f"- file: `{CSV.relative_to(HERE.parent.parent.parent)}`")
    md.append(f"- rows: **{len(df):,}** (expected 29,434)")
    md.append(f"- cols: {len(df.columns)}")
    md.append("")

    # ---------- 1. counts / errors ----------
    md.append("## 1. Basic integrity")
    err_n   = (df["error"].fillna("") != "").sum()
    null_s  = df["subject"].isna().sum()
    null_l  = df["level"].isna().sum()
    empty_r = (df["raw_response"].fillna("") == "").sum()
    md.append(f"- error non-empty: **{err_n}**")
    md.append(f"- subject null: **{null_s}**")
    md.append(f"- level null: **{null_l}**")
    md.append(f"- raw_response empty: **{empty_r}**")
    if err_n or null_s or null_l or empty_r:
        issues.append(f"errors/nulls present: err={err_n} sub_null={null_s} lvl_null={null_l}")

    # row_index integrity
    ri_unique = df["row_index"].is_unique
    ri_set = set(df["row_index"])
    expected = set(range(29434))
    missing  = expected - ri_set
    extra    = ri_set - expected
    md.append(f"- row_index unique: {ri_unique}")
    md.append(f"- row_index range matches 0..29433: missing={len(missing)} extra={len(extra)}")
    if missing or extra or not ri_unique:
        issues.append(f"row_index mismatch: missing={len(missing)} extra={len(extra)} unique={ri_unique}")

    # finish_reason
    md.append(f"- finish_reason `length` (truncation): **{(df['finish_reason']=='length').sum()}**")

    # ---------- 2. category validity ----------
    md.append("\n## 2. Category validity")
    bad_s = (~df["subject"].isin(ALLOWED_SUBJ)).sum()
    md.append(f"- subjects outside 8 allowed: **{bad_s}**")
    md.append(f"- level dtype: `{df['level'].dtype}`")
    bad_l = (~df["level"].between(1,8)).sum()
    md.append(f"- level outside [1,8]: **{bad_l}**")
    if bad_s or bad_l:
        issues.append(f"category invalid: bad_subject={bad_s} bad_level={bad_l}")

    # ---------- 3. raw_response strict JSON ----------
    def ok_json(s):
        try:
            j = json.loads(s)
            return ("subject" in j and "level" in j and len(j)==2)
        except Exception:
            return False
    df["_raw_ok"] = df["raw_response"].fillna("").map(ok_json)
    n_ok = int(df["_raw_ok"].sum())
    md.append(f"\n## 3. raw_response valid JSON: **{n_ok}/{len(df)}**")
    if n_ok != len(df):
        issues.append(f"raw_response invalid JSON: {len(df)-n_ok}")

    # ---------- 4. tokens / latency / attempts ----------
    md.append("\n## 4. Tokens / latency / retries")
    md.append(f"- prompt_tokens median/max: **{df['prompt_tokens'].median():.0f}** / {df['prompt_tokens'].max():.0f}")
    md.append(f"- completion_tokens median/max: **{df['completion_tokens'].median():.0f}** / {df['completion_tokens'].max():.0f}")
    md.append(f"- total prompt_tokens: {df['prompt_tokens'].sum():,}")
    md.append(f"- total completion_tokens: {df['completion_tokens'].sum():,}")
    md.append(f"- attempts >1: **{(df['attempts']>1).sum()}**, max attempts: {df['attempts'].max()}")
    md.append(f"- latency mean/p50/p95/p99 (s): {df['latency_s'].mean():.2f} / {df['latency_s'].median():.2f} / {df['latency_s'].quantile(0.95):.2f} / {df['latency_s'].quantile(0.99):.2f}")

    # cost estimate (gpt-4.1-mini: $0.40/1M in, $1.60/1M out; ignore caching)
    cost_no_cache = df["prompt_tokens"].sum()*0.40/1e6 + df["completion_tokens"].sum()*1.60/1e6
    md.append(f"- cost estimate (no caching): **${cost_no_cache:.2f}**  (실제 ~$9 → caching 효과)")

    # ---------- 5. meta consistency ----------
    md.append("\n## 5. Meta consistency")
    md.append(f"- models: {sorted(df['model'].unique().tolist())}")
    md.append(f"- prompt_sha: {sorted(df['prompt_sha'].unique().tolist())}")

    # ---------- 6. distributions ----------
    md.append("\n## 6. Distributions")
    md.append("\n### 6.1 subject")
    md.append("```\n" + df["subject"].value_counts().to_string() + "\n```")

    md.append("\n### 6.2 level")
    md.append("```\n" + df["level"].value_counts().sort_index().to_string() + "\n```")

    md.append("\n### 6.3 source")
    md.append("```\n" + df["source"].value_counts().to_string() + "\n```")

    md.append("\n### 6.4 subject × level cross-tab")
    ct_sl = pd.crosstab(df["subject"], df["level"], margins=True)
    md.append("```\n" + ct_sl.to_string() + "\n```")

    md.append("\n### 6.5 source × level cross-tab")
    md.append("```\n" + pd.crosstab(df["source"], df["level"], margins=True).to_string() + "\n```")

    md.append("\n### 6.6 source × subject cross-tab")
    md.append("```\n" + pd.crosstab(df["source"], df["subject"], margins=True).to_string() + "\n```")

    # ---------- 7. sparse cell diagnosis ----------
    md.append("\n## 7. Sparse cells (subject × level, count < 30)")
    ct = pd.crosstab(df["subject"], df["level"])
    sparse = ct.stack()[ct.stack() < 30].sort_values()
    md.append(f"- # sparse cells: **{len(sparse)}** of {ct.size}")
    if len(sparse):
        md.append("```")
        md.append(sparse.to_string())
        md.append("```")
    # also list which levels have any subjects with n<30
    md.append(f"- cells with n=0: **{(ct==0).sum().sum()}**")

    # ---------- 8. signal quality: level vs r1_cot_token_count ----------
    md.append("\n## 8. Difficulty signal — Spearman ρ with level")
    corr = df[["level","r1_cot_token_count","problem_qwen_tok_len","problem_char_len","solution_char_len"]].corr(method="spearman")["level"].round(3)
    md.append("```\n" + corr.to_string() + "\n```")
    md.append(f"- level vs r1_cot_token_count overall: ρ = **{corr['r1_cot_token_count']}**")

    # per-source if multi-source
    if df["source"].nunique() > 1:
        md.append("\nPer-source level vs r1_cot ρ:")
        rows = []
        for src, g in df.groupby("source"):
            if len(g) >= 50:
                r = g[["level","r1_cot_token_count"]].corr(method="spearman").iloc[0,1]
                rows.append(f"  {src:<15} n={len(g):>6}  ρ={r:.3f}")
        md.append("```\n" + "\n".join(rows) + "\n```")

    # ---------- 9. comparison with smoke 200 ----------
    smoke_csv = HERE/"outputs"/"smoke200_labels.csv"
    if smoke_csv.exists():
        md.append("\n## 9. Smoke (200) vs Full (29,434)")
        sm = pd.read_csv(smoke_csv)
        s_dist = sm["level"].value_counts(normalize=True).sort_index()
        f_dist = df["level"].value_counts(normalize=True).sort_index()
        tab = pd.DataFrame({"smoke_pct": (s_dist*100).round(1),
                            "full_pct":  (f_dist*100).round(1)}).fillna(0)
        md.append("```\nlevel distribution (%)\n" + tab.to_string() + "\n```")

    # ---------- 10. sample inspection: per-level first row ----------
    md.append("\n## 10. Qualitative samples — one per level")
    for lv in sorted(df["level"].dropna().unique()):
        r = df[df["level"]==lv].iloc[0]
        txt = str(r["problem_text"]).replace("\n"," ")[:140]
        md.append(f"- **L{int(lv)} {r['subject']}**: `{txt}`")

    # ---------- 11. Issues summary ----------
    md.append("\n## 11. Issues")
    if issues:
        for x in issues:
            md.append(f"- ⚠️ {x}")
    else:
        md.append("- ✅ no integrity issues detected.")

    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"[done] report → {REPORT}")

    # console summary
    print()
    print("=== Summary ===")
    print(f"rows         : {len(df):,}")
    print(f"errors       : {err_n}")
    print(f"truncations  : {(df['finish_reason']=='length').sum()}")
    print(f"raw_ok       : {n_ok}/{len(df)}")
    print(f"row_index    : missing={len(missing)} extra={len(extra)} unique={ri_unique}")
    print(f"subject bad  : {bad_s}")
    print(f"level bad    : {bad_l}")
    print(f"retries>1    : {(df['attempts']>1).sum()}")
    print(f"cost est     : ${cost_no_cache:.2f}  (no caching)")
    print(f"ρ(level,r1)  : {corr['r1_cot_token_count']}")
    print()
    print("subject distribution:")
    print(df["subject"].value_counts().to_string())
    print()
    print("level distribution:")
    print(df["level"].value_counts().sort_index().to_string())
    print()
    print("sparse cells (n<30):", len(sparse))

if __name__ == "__main__":
    main()
