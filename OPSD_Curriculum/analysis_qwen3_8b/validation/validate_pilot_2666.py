#!/usr/bin/env python3
"""
validate_pilot_2666.py  —  Pilot 2,666 Validation: Track A + Track B + Cross-Track
CPU only, no SLURM, memory <= 4 GB.
Run: python validate_pilot_2666.py
"""

from __future__ import annotations
import sys, os, json, time, random, subprocess, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# scipy / sklearn optional — fallback if unavailable
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not available; Spearman / ANOVA skipped")

try:
    from sklearn.decomposition import PCA as sk_PCA
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not available; PCA / silhouette skipped")

import torch

# ─────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────
BASE = Path("/scratch/lami2026/personal/jimin_2782")
TASK2_DIR = BASE / "src/4.6_Task2"
QWEN_DIR  = BASE / "src/OPSD_Curriculum/analysis_qwen3_8b"

TRACK_A_PARQUET = QWEN_DIR / "outputs/pass_rate_pilot_2666.parquet"
TRACK_B_DIR     = QWEN_DIR / "activation/outputs/shifts"
C5_JSON         = TASK2_DIR / "activation/analysis/full_final/C5_outlier_samples.json"

OUT_DIR        = QWEN_DIR / "validation/outputs"
PLOTS_DIR      = OUT_DIR / "plots"
SPOT_DIR       = OUT_DIR / "spot_check"
REPORT_PATH    = OUT_DIR / "report.md"

EXPECTED_N    = 2666
EXPECTED_L    = 36
N_ROLLOUTS    = 8
MID_LAYERS    = [15, 20, 25]

for d in [OUT_DIR, PLOTS_DIR, SPOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────
_report_lines: list[str] = []

def _rpt(line: str = ""):
    _report_lines.append(line)
    print(line)

def flush_report():
    REPORT_PATH.write_text("\n".join(_report_lines) + "\n", encoding="utf-8")

PASS_BUDGET  = []   # (section, result)  result in {PASS, WARN, FAIL}
DECISION: dict[str,str] = {}

def record(section: str, result: str):
    PASS_BUDGET.append((section, result))
    emoji = {"PASS":"✅","WARN":"⚠️","FAIL":"❌"}.get(result, "❓")
    _rpt(f"**Status**: {emoji} {result}")

# ─────────────────────────────────────────────────────
# Load reference ID set from _nait_common
# ─────────────────────────────────────────────────────
def load_pilot_id_set() -> set[str]:
    sys.path.insert(0, str(TASK2_DIR / "activation/analysis"))
    from _nait_common import BASE_DIR, load_metadata, resolve_shift_dirs  # type: ignore
    df = load_metadata(resolve_shift_dirs(None) + [TASK2_DIR / "activation/full_shifts_l7l8"])
    # status can be 'completed', 'ok', 'ok (skipped)' depending on pipeline version
    ok_statuses = {"completed", "ok", "ok (skipped)"}
    df = df[df["status"].isin(ok_statuses)].drop_duplicates(subset="id")
    assert len(df) == EXPECTED_N, f"Expected {EXPECTED_N}, got {len(df)} (status vals: {df['status'].value_counts().to_dict() if len(df) else 'empty'})"
    return set(df["id"].astype(str).tolist())

# ─────────────────────────────────────────────────────
# Section A
# ─────────────────────────────────────────────────────
def section_A(df_a: pd.DataFrame, pilot_ids: set[str]) -> dict:
    """Returns dict with keys for cross-track use."""
    results = {}

    # ── A1. Integrity ────────────────────────────────
    _rpt("\n### A1. Integrity")
    # NaN in mean_response_length_correct/incorrect is structurally expected:
    # pass=0 samples have no correct responses (NaN correct), pass=1 have no incorrect (NaN incorrect)
    EXPECTED_NAN_COLS = {"mean_response_length_correct", "mean_response_length_incorrect"}
    non_nan_check_cols = [c for c in df_a.columns if c not in EXPECTED_NAN_COLS]
    null_count = df_a[non_nan_check_cols].isnull().any(axis=1).sum()
    nan_correct  = df_a["mean_response_length_correct"].isna().sum()  if "mean_response_length_correct"  in df_a.columns else 0
    nan_incorrect = df_a["mean_response_length_incorrect"].isna().sum() if "mean_response_length_incorrect" in df_a.columns else 0
    rows = [
        ("Row count", EXPECTED_N, len(df_a), len(df_a) == EXPECTED_N),
        ("ID set match (diff=0)", 0, len(set(df_a["sample_id"].astype(str)) ^ pilot_ids),
            set(df_a["sample_id"].astype(str)) == pilot_ids),
        ("pass_count range 0-8", "0-8",
            f"{df_a['pass_count'].min()}-{df_a['pass_count'].max()}",
            df_a["pass_count"].between(0,8).all()),
        ("pass_rate == pass_count/8", "all equal",
            "check", ((df_a["pass_count"]/8 - df_a["pass_rate"]).abs() < 1e-9).all()),
        ("raw_responses len==8", "all 8",
            "check", all(len(r)==N_ROLLOUTS for r in df_a["raw_responses"])),
        (f"Null rows (excl. len_correct/incorrect NaN)",
            0, null_count, null_count == 0),
        (f"NaN len_correct (pass=0 expected)", f"~{(df_a['pass_count']==0).sum()}",
            nan_correct, nan_correct <= (df_a["pass_count"]==0).sum()),
    ]
    _rpt("\n| Check | Expected | Actual | Status |")
    _rpt("|---|---|---|---|")
    a1_ok = True
    for name, exp, act, ok in rows:
        mark = "✓" if ok else "✗"
        _rpt(f"| {name} | {exp} | {act} | {mark} |")
        if not ok: a1_ok = False
    record("A1", "PASS" if a1_ok else "FAIL")
    _rpt(f"\n**Findings**: {'All integrity checks passed.' if a1_ok else 'Integrity failure — see table.'}")

    # ── A2. Pass rate distribution ────────────────────
    _rpt("\n### A2. Pass Rate Distribution (§5.7 Hybrid input)")
    pr = df_a["pass_rate"].values
    pass0_n    = (pr == 0.0).sum();  pass0_r = pass0_n / len(pr)
    pass1_n    = (pr == 1.0).sum();  pass1_r = pass1_n / len(pr)
    buckets = {
        "0":           (pr == 0.0).sum(),
        "(0, 0.125]":  ((pr > 0) & (pr <= 0.125)).sum(),
        "(0.125, 0.25]": ((pr > 0.125) & (pr <= 0.25)).sum(),
        "(0.25, 0.5]": ((pr > 0.25) & (pr <= 0.5)).sum(),
        "(0.5, 0.875)": ((pr > 0.5) & (pr < 0.875)).sum(),
        "[0.875, 1.0)": ((pr >= 0.875) & (pr < 1.0)).sum(),
        "1.0":          (pr == 1.0).sum(),
    }
    _rpt(f"\n- pass=0 ratio: **{pass0_r:.3f}** ({pass0_n}/{len(pr)})")
    _rpt(f"- pass=1.0 ratio: **{pass1_r:.3f}** ({pass1_n}/{len(pr)})")
    _rpt(f"- mean pass_rate: {pr.mean():.3f}, median: {np.median(pr):.3f}")
    _rpt("\n| Bucket | Count | % |")
    _rpt("|---|---|---|")
    for b, cnt in buckets.items():
        _rpt(f"| {b} | {cnt} | {cnt/len(pr)*100:.1f}% |")

    # §5.7 decision
    if pass0_r < 0.10:
        hybrid = "HYBRID=NO"
    elif pass0_r <= 0.30:
        hybrid = "HYBRID=LAYERED"
    else:
        hybrid = "HYBRID=REQUIRED"
    DECISION["§5.7 Hybrid"] = hybrid
    _rpt(f"\n**§5.7 Auto-Decision**: `{hybrid}` (pass=0 = {pass0_r:.1%})")

    # Plot
    bins = [0, 0.001, 0.125+0.001, 0.25+0.001, 0.5+0.001, 0.875, 1.0+0.001]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pr, bins=np.linspace(0,1,17), edgecolor="black", color="steelblue")
    for bnd in [0.125, 0.25, 0.5, 0.875]:
        ax.axvline(bnd, linestyle="--", color="red", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Pass Rate"); ax.set_ylabel("Count")
    ax.set_title("A2: Pass Rate Histogram (N=2,666)")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "A2_pass_rate_histogram.png", dpi=120)
    plt.close(fig)
    _rpt("\n**Plots**: `plots/A2_pass_rate_histogram.png`")
    record("A2", "PASS")

    results["pass0_n"] = pass0_n
    results["pass0_r"] = pass0_r

    # ── A3. Signal alignment ──────────────────────────
    _rpt("\n### A3. Signal Alignment (pass_rate ↔ level)")
    level_int = df_a["level"].values.astype(float)
    if HAS_SCIPY:
        rho, pval = sp_stats.spearmanr(pr, level_int)
    else:
        rho, pval = float("nan"), float("nan")
    _rpt(f"\n- Spearman ρ(pass_rate, level) = **{rho:.4f}** (p={pval:.3e})")

    # per subject mean
    subj_means = df_a.groupby("subject")["pass_rate"].agg(["mean","std","count"])
    _rpt("\n**Pass rate by subject** (top 5 by mean):")
    _rpt("\n| Subject | Mean | Std | N |")
    _rpt("|---|---|---|---|")
    for s, row2 in subj_means.sort_values("mean", ascending=False).head(5).iterrows():
        _rpt(f"| {s} | {row2['mean']:.3f} | {row2['std']:.3f} | {int(row2['count'])} |")

    # level × subject heatmap
    pivot = df_a.pivot_table(values="pass_rate", index="level", columns="subject", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.8), max(4, len(pivot)*0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Subject"); ax.set_ylabel("Level")
    ax.set_title(f"A3: Mean Pass Rate by Level × Subject  (ρ={rho:.3f})")
    plt.colorbar(im, ax=ax, label="Pass Rate")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "A3_pass_rate_vs_level_heatmap.png", dpi=120)
    plt.close(fig)
    _rpt("\n**Plots**: `plots/A3_pass_rate_vs_level_heatmap.png`")

    # NOTE: MATH levels 1-8 are difficulty levels (higher = harder),
    # so we expect NEGATIVE rho (higher level → lower pass rate).
    # Use |rho| for thresholding; sign just indicates direction.
    if not np.isnan(rho):
        absrho = abs(rho)
        sign_ok = rho < 0  # expected direction
        if absrho > 0.30 and sign_ok:    a3 = f"ACCEPTABLE_A3=YES (ρ={rho:.3f}, expected negative)"
        elif absrho > 0.30:               a3 = f"ACCEPTABLE_A3=YES_INVERTED (ρ={rho:.3f}, unexpected positive)"
        elif absrho > 0.15:               a3 = f"ACCEPTABLE_A3=MARGINAL (|ρ|={absrho:.3f})"
        else:                             a3 = f"ACCEPTABLE_A3=FAIL (|ρ|={absrho:.3f})"
    else:
        a3 = "ACCEPTABLE_A3=UNKNOWN"
    DECISION["Pilot A3"] = a3
    _rpt(f"\n**Auto-Decision**: `{a3}`")
    record("A3", "PASS" if "YES" in a3 else ("WARN" if "MARGINAL" in a3 else "FAIL"))
    results["rho_level"] = rho; results["a3"] = a3

    # ── A4. Response quality ──────────────────────────
    _rpt("\n### A4. Response Quality")
    trunc_total = df_a["truncation_count"].sum()
    trunc_denom = EXPECTED_N * N_ROLLOUTS
    trunc_rate  = trunc_total / trunc_denom
    _rpt(f"\n- Truncation rate: {trunc_rate:.3f} ({trunc_total}/{trunc_denom} rollouts)")
    _rpt(f"- Mean resp len (all): {df_a['mean_response_length'].mean():.1f} tokens")
    _rpt(f"- Mean resp len (correct): {df_a['mean_response_length_correct'].dropna().mean():.1f} tokens")
    _rpt(f"- Mean resp len (incorrect): {df_a['mean_response_length_incorrect'].dropna().mean():.1f} tokens")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_a["mean_response_length_correct"].dropna(), bins=50, alpha=0.6, label="Correct", color="green")
    ax.hist(df_a["mean_response_length_incorrect"].dropna(), bins=50, alpha=0.6, label="Incorrect", color="red")
    ax.set_xlabel("Mean Response Length (tokens)"); ax.set_ylabel("Count")
    ax.set_title("A4: Response Length — Correct vs Incorrect")
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "A4_response_length_correct_vs_incorrect.png", dpi=120)
    plt.close(fig)
    _rpt("\n**Plots**: `plots/A4_response_length_correct_vs_incorrect.png`")

    # Spot check: pass=0 samples
    pass0_df = df_a[df_a["pass_rate"] == 0.0].head(10)
    lines = [f"# Spot Check: pass=0 samples (first 10)\n"]
    for _, row in pass0_df.iterrows():
        lines.append(f"## Sample {row['sample_id']} | level={row['level']} | subject={row['subject']}")
        lines.append(f"**ground_truth**: `{str(row['ground_truth'])[:200]}`\n")
        for i, resp in enumerate(row["raw_responses"][:2]):
            lines.append(f"**response[{i}]** (first 500 chars):\n```\n{str(resp)[:500]}\n```\n")
        lines.append("---")
    (SPOT_DIR / "A4_pass_0_samples.md").write_text("\n".join(lines), encoding="utf-8")

    # Spot check: pass=1 samples
    pass1_df = df_a[df_a["pass_rate"] == 1.0].head(3)
    lines = [f"# Spot Check: pass=1.0 samples (first 3)\n"]
    for _, row in pass1_df.iterrows():
        lines.append(f"## Sample {row['sample_id']} | level={row['level']} | subject={row['subject']}")
        lines.append(f"**ground_truth**: `{str(row['ground_truth'])[:200]}`\n")
        for i, resp in enumerate(row["raw_responses"][:2]):
            lines.append(f"**response[{i}]** (first 500 chars):\n```\n{str(resp)[:500]}\n```\n")
        lines.append("---")
    (SPOT_DIR / "A4_pass_full_samples.md").write_text("\n".join(lines), encoding="utf-8")
    _rpt("\n**Spot check**: `spot_check/A4_pass_0_samples.md`, `A4_pass_full_samples.md`")
    record("A4", "PASS")

    # ── A5. Stratification ────────────────────────────
    _rpt("\n### A5. Stratification")
    unit_counts = df_a.groupby(["subject","level"]).size()
    _rpt(f"\n- N units: {len(unit_counts)}")
    _rpt(f"- Mean per unit: {unit_counts.mean():.1f} (expected ~{EXPECTED_N}/{len(unit_counts):.0f})")
    _rpt(f"- Min/Max per unit: {unit_counts.min()} / {unit_counts.max()}")
    _rpt(f"- Actual total: {EXPECTED_N} (expected ≈2600 at 50/unit × 52 units)")
    record("A5", "PASS")

    # ── A6. Resource ──────────────────────────────────
    _rpt("\n### A6. Resource (Track A wallclock)")
    slurm_logs = sorted((BASE / "runs").glob("slurm-nait_pilot_qwen3.*.iREMB-C-07.out"))
    if slurm_logs:
        try:
            txt = slurm_logs[-1].read_text()
            # look for Date lines
            dates = [l for l in txt.splitlines() if "Date:" in l]
            if len(dates) >= 2:
                _rpt(f"\n- Track B sbatch log: `{slurm_logs[-1].name}`")
                _rpt(f"- Start: {dates[0].strip()}")
                _rpt(f"- End:   {dates[-1].strip()}")
        except Exception:
            pass
    _rpt("\n(Track A SLURM log not found in standard location; wallclock not extracted)")
    record("A6", "PASS")

    return results

# ─────────────────────────────────────────────────────
# Section B  (streaming)
# ─────────────────────────────────────────────────────
def section_B(pilot_ids: set[str]) -> dict:
    results = {}
    pt_files = sorted(TRACK_B_DIR.glob("*.pt"))
    meta_path = TRACK_B_DIR / "shifts_metadata.jsonl"

    # ── B1. Integrity ─────────────────────────────────
    _rpt("\n### B1. Integrity")
    n_pt = len(pt_files)
    meta_rows = meta_valid = meta_ids = 0
    meta_id_set: set[str] = set()
    meta_err_status = 0
    if meta_path.exists():
        for line in meta_path.open():
            try:
                obj = json.loads(line)
                meta_rows += 1
                meta_valid += 1
                sid = str(obj.get("id",""))
                meta_id_set.add(sid)
                if obj.get("status","") == "error":
                    meta_err_status += 1
            except Exception:
                meta_rows += 1

    # checkpoints
    cp_ids: set[str] = set()
    for c in range(4):
        cp = TRACK_B_DIR / f"shifts_checkpoint_chunk{c}.json"
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                for sid in d.get("processed_ids", []):
                    cp_ids.add(str(sid))
            except Exception:
                pass

    rows = [
        (".pt file count", EXPECTED_N, n_pt, n_pt == EXPECTED_N),
        ("metadata row count", EXPECTED_N, meta_valid, meta_valid == EXPECTED_N),
        ("metadata ID == pilot_ids", "∅ diff", len(meta_id_set ^ pilot_ids), meta_id_set == pilot_ids),
        ("status==error rate", "<1%", f"{meta_err_status}/{ meta_valid}", meta_err_status/max(meta_valid,1) < 0.01),
        ("checkpoint union card", EXPECTED_N, len(cp_ids), len(cp_ids) == EXPECTED_N),
    ]
    _rpt("\n| Check | Expected | Actual | Status |")
    _rpt("|---|---|---|---|")
    b1_ok = True
    for name, exp, act, ok in rows:
        mark = "✓" if ok else "✗"
        _rpt(f"| {name} | {exp} | {act} | {mark} |")
        if not ok: b1_ok = False
    record("B1", "PASS" if b1_ok else "FAIL")
    _rpt(f"\n**Findings**: {'All B1 checks passed.' if b1_ok else 'B1 integrity failure.'}")

    # ── B2. Tensor shape ──────────────────────────────
    _rpt("\n### B2. Tensor Shape (random 10 .pt)")
    rng = random.Random(42)
    sample_pts = rng.sample(pt_files, min(10, len(pt_files)))
    b2_ok = True
    b2_details = []
    for fp in sample_pts:
        try:
            s = torch.load(str(fp), map_location="cpu", weights_only=False)
            shifts = s.get("shifts", {})
            n_layers = len(shifts)
            shapes = [v.shape for v in shifts.values()]
            dtypes = [str(v.dtype) for v in shifts.values()]
            nans   = any(torch.isnan(v.float()).any() for v in shifts.values())
            infs   = any(torch.isinf(v.float()).any() for v in shifts.values())
            t1     = s.get("t1_idx", -1)
            tK     = s.get("tK_idx", -1)
            ngen   = s.get("num_generated_tokens", -1)
            ok = (n_layers == EXPECTED_L
                  and all(sh == torch.Size([12288]) for sh in shapes)
                  and all(d == "torch.bfloat16" for d in dtypes)
                  and not nans and not infs and t1 >= 0 and tK >= t1)
            b2_details.append((fp.stem, n_layers, shapes[0] if shapes else "?", dtypes[0] if dtypes else "?", ok))
            if not ok: b2_ok = False
            del s
        except Exception as e:
            b2_details.append((fp.stem, "ERROR", str(e), "", False))
            b2_ok = False

    _rpt(f"\n- n_layers expected: {EXPECTED_L}")
    _rpt(f"- intermediate_size: 12288 (Qwen3-8B)")
    _rpt(f"- dtype: bfloat16")
    _rpt(f"- 10 random spot checks: {'all OK' if b2_ok else 'SOME FAILED'}")
    record("B2", "PASS" if b2_ok else "FAIL")
    _rpt(f"\n**Findings**: Tensor shape/dtype/sanity {'OK' if b2_ok else 'FAILED'}.")

    # ── B3. Activation signal sanity (streaming) ──────
    _rpt("\n### B3. Activation Signal Sanity (streaming all 2,666)")
    layer_norms: dict[int, list] = defaultdict(list)
    sample_mean_norms: list[float] = []
    sample_ids_b: list[str] = []
    b_trunc_count = 0
    b_gen_tokens: list[int] = []
    think_count = 0
    spot_texts: list[tuple] = []   # (id, text)
    spot_trunc: list[tuple] = []   # (id, text_tail)
    # mid-layer activations for X4 PCA
    mid_vecs: list[np.ndarray] = []

    print(f"\n[B3] Streaming {len(pt_files)} .pt files...")
    t0 = time.time()
    for i, fp in enumerate(pt_files):
        try:
            s = torch.load(str(fp), map_location="cpu", weights_only=False)
            shifts = s["shifts"]
            sid    = str(s["id"])
            is_trunc = bool(s.get("is_trunc", False))
            ngen     = int(s.get("num_generated_tokens", 0))
            gen_text = str(s.get("generated_text", ""))

            # layer norms
            layer_ns = []
            for l in range(EXPECTED_L):
                n = shifts[l].float().norm().item()
                layer_norms[l].append(n)
                layer_ns.append(n)
            sample_mean_norms.append(float(np.mean(layer_ns)))
            sample_ids_b.append(sid)

            if is_trunc:
                b_trunc_count += 1
                if len(spot_trunc) < 5:
                    spot_trunc.append((sid, gen_text[-200:]))
            b_gen_tokens.append(ngen)

            # think tag
            if "<think>" in gen_text or "</think>" in gen_text:
                think_count += 1

            # mid-layer mean
            mid = np.mean(np.stack([shifts[l].float().numpy() for l in MID_LAYERS]), axis=0)
            mid_vecs.append(mid.astype(np.float32))

            if len(spot_texts) < 10:
                spot_texts.append((sid, gen_text[:500]))

            del s
        except Exception as e:
            print(f"  [WARN] failed {fp.name}: {e}")

        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(pt_files)}] elapsed={time.time()-t0:.0f}s")

    # Stats
    layer_means = np.array([np.mean(layer_norms[l]) for l in range(EXPECTED_L)])
    layer_stds  = np.array([np.std(layer_norms[l])  for l in range(EXPECTED_L)])
    layer_p5    = np.array([np.percentile(layer_norms[l], 5)  for l in range(EXPECTED_L)])
    layer_p95   = np.array([np.percentile(layer_norms[l], 95) for l in range(EXPECTED_L)])

    b3_ok_min = layer_means.min() > 0.01
    b3_ok_max = layer_means.max() < 100.0
    b_trunc_rate = b_trunc_count / len(pt_files)

    _rpt(f"\n- Samples processed: {len(sample_mean_norms)}")
    _rpt(f"- Layer norm range: min_mean={layer_means.min():.3f}, max_mean={layer_means.max():.3f}")
    _rpt(f"- Sample mean norm: mean={np.mean(sample_mean_norms):.3f}, p5={np.percentile(sample_mean_norms,5):.3f}, p95={np.percentile(sample_mean_norms,95):.3f}")
    _rpt(f"- Truncation rate (is_trunc): {b_trunc_rate:.3f} ({b_trunc_count}/{len(pt_files)})")
    _rpt(f"- Tokens generated: median={int(np.median(b_gen_tokens))}, p95={int(np.percentile(b_gen_tokens,95))}, max={max(b_gen_tokens)}")
    _rpt(f"- <think> tags found: **{think_count}** (should be 0)")

    # Per-layer boxplot
    fig, ax = plt.subplots(figsize=(14, 4))
    data_for_box = [layer_norms[l] for l in range(EXPECTED_L)]
    bp = ax.boxplot(data_for_box, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5))
    ax.set_xlabel("Layer"); ax.set_ylabel("||ΔA||_2")
    ax.set_title("B3: Per-Layer Activation Shift Norm (2,666 samples)")
    ax.set_xticks(range(1, EXPECTED_L+1, 4))
    ax.set_xticklabels(range(0, EXPECTED_L, 4))
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "B3_layer_norm_boxplot.png", dpi=120)
    plt.close(fig)

    # Sample norm histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sample_mean_norms, bins=60, color="steelblue", edgecolor="black")
    ax.set_xlabel("Mean ||ΔA||_2 across 36 layers"); ax.set_ylabel("Count")
    ax.set_title("B3: Per-Sample Mean Activation Norm")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "B3_sample_norm_histogram.png", dpi=120)
    plt.close(fig)
    _rpt("\n**Plots**: `plots/B3_layer_norm_boxplot.png`, `B3_sample_norm_histogram.png`")

    b3_status = "PASS" if (b3_ok_min and b3_ok_max and think_count == 0) else ("WARN" if think_count == 0 else "FAIL")
    record("B3", b3_status)

    # ── B4. Response quality ──────────────────────────
    _rpt("\n### B4. Response Quality (Track B)")
    _rpt(f"\n- is_trunc rate: {b_trunc_rate:.3f}")
    _rpt(f"- think-tag count: {'**0 ✓**' if think_count == 0 else f'**{think_count} ❌ (chat template bug)**'}")

    lines = ["# Spot Check: B4 — random 10 generated_text (first 500 chars)\n"]
    for sid, txt in spot_texts:
        lines.append(f"## Sample {sid}\n```\n{txt}\n```\n---")
    (SPOT_DIR / "B4_generated_text_samples.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Spot Check: B4 — truncated samples (last 200 chars)\n"]
    for sid, tail in spot_trunc:
        lines.append(f"## Sample {sid}\n```\n{tail}\n```\n---")
    (SPOT_DIR / "B4_truncated_samples.md").write_text("\n".join(lines), encoding="utf-8")
    _rpt("\n**Spot check**: `spot_check/B4_generated_text_samples.md`, `B4_truncated_samples.md`")

    b4_ok = (think_count == 0)
    record("B4", "PASS" if b4_ok else "FAIL")

    # ── B5. Cross-chunk consistency ───────────────────
    _rpt("\n### B5. Cross-Chunk Consistency")
    chunk_ids: list[set[str]] = []
    for c in range(4):
        cp = TRACK_B_DIR / f"shifts_checkpoint_chunk{c}.json"
        cids: set[str] = set()
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                cids = set(str(x) for x in d.get("processed_ids", []))
            except Exception:
                pass
        chunk_ids.append(cids)
    chunk_sizes = [len(s) for s in chunk_ids]
    union_size  = len(set.union(*chunk_ids))
    pairwise_intersects = []
    for i in range(4):
        for j in range(i+1, 4):
            pairwise_intersects.append(len(chunk_ids[i] & chunk_ids[j]))

    b5_ok = (max(pairwise_intersects) == 0 and union_size == EXPECTED_N)
    _rpt(f"\n- Chunk sizes: {chunk_sizes}")
    _rpt(f"- Union cardinality: {union_size} (expected {EXPECTED_N})")
    _rpt(f"- Pairwise intersections: {pairwise_intersects} (should all be 0)")
    record("B5", "PASS" if b5_ok else "FAIL")

    # ── B6. Storage ───────────────────────────────────
    _rpt("\n### B6. Storage")
    try:
        r = subprocess.run(["du", "-sh", str(TRACK_B_DIR)], capture_output=True, text=True)
        _rpt(f"\n- Directory size: {r.stdout.strip()}")
        total_bytes = sum(fp.stat().st_size for fp in pt_files)
        _rpt(f"- Avg .pt size: {total_bytes/len(pt_files)/1024:.1f} KB")
    except Exception as e:
        _rpt(f"\n- Storage check skipped: {e}")
    record("B6", "PASS")

    return {
        "sample_ids_b": sample_ids_b,
        "sample_mean_norms": sample_mean_norms,
        "layer_norms": layer_norms,
        "b_trunc_count": b_trunc_count,
        "mid_vecs": mid_vecs,
        "think_count": think_count,
    }

# ─────────────────────────────────────────────────────
# Section X  (cross-track)
# ─────────────────────────────────────────────────────
def section_X(df_a: pd.DataFrame, b_res: dict, pilot_ids: set[str]):
    sample_ids_b  = b_res["sample_ids_b"]
    mean_norms    = b_res["sample_mean_norms"]
    mid_vecs      = b_res["mid_vecs"]

    df_b_summary = pd.DataFrame({
        "sample_id": [str(x) for x in sample_ids_b],
        "mean_norm": mean_norms,
    })

    # ── X1. ID set match ──────────────────────────────
    _rpt("\n### X1. ID Set Match (3-way)")
    ids_a = set(df_a["sample_id"].astype(str))
    ids_b = set(df_b_summary["sample_id"])
    r1 = ids_a ^ pilot_ids; r2 = ids_b ^ pilot_ids; r3 = ids_a ^ ids_b
    x1_ok = (len(r1) == 0 and len(r2) == 0 and len(r3) == 0)
    _rpt(f"\n- A △ ref = {len(r1)}, B △ ref = {len(r2)}, A △ B = {len(r3)}")
    record("X1", "PASS" if x1_ok else "FAIL")

    # ── X2. Cross-track signal ────────────────────────
    _rpt("\n### X2. Pass Rate ↔ Activation Norm (Pilot Diagnostic #1)")
    df_x = pd.merge(
        df_a[["sample_id","pass_rate","level","subject","truncation_count"]].assign(sample_id=df_a["sample_id"].astype(str)),
        df_b_summary,
        on="sample_id"
    )

    # bucket assignment
    def bucket(pr):
        if pr == 0.0: return "0"
        if pr <= 0.125: return "(0,0.125]"
        if pr <= 0.25:  return "(0.125,0.25]"
        if pr <= 0.5:   return "(0.25,0.5]"
        if pr < 0.875:  return "(0.5,0.875)"
        if pr < 1.0:    return "[0.875,1)"
        return "1.0"
    df_x["bucket"] = df_x["pass_rate"].apply(bucket)
    bucket_order = ["0","(0,0.125]","(0.125,0.25]","(0.25,0.5]","(0.5,0.875)","[0.875,1)","1.0"]

    cross = df_x.groupby("bucket")["mean_norm"].agg(["mean","std","count"]).reindex(bucket_order)
    _rpt("\n| Pass Bucket | Mean Norm | Std | N |")
    _rpt("|---|---|---|---|")
    for b, row2 in cross.iterrows():
        if not np.isnan(row2["mean"]):
            _rpt(f"| {b} | {row2['mean']:.3f} | {row2['std']:.3f} | {int(row2['count'])} |")

    if HAS_SCIPY:
        rho_x, pval_x = sp_stats.spearmanr(df_x["pass_rate"], df_x["mean_norm"])
        _rpt(f"\n- Spearman ρ(pass_rate, mean_norm) = **{rho_x:.4f}** (p={pval_x:.3e})")
        # ANOVA across buckets
        groups = [df_x[df_x["bucket"] == b]["mean_norm"].values for b in bucket_order
                  if b in df_x["bucket"].values and len(df_x[df_x["bucket"]==b]) > 1]
        if len(groups) >= 2:
            try:
                f_stat, anova_p = sp_stats.f_oneway(*groups)
                _rpt(f"- ANOVA F={f_stat:.2f}, p={anova_p:.3e}")
            except Exception:
                anova_p = float("nan")
                f_stat  = float("nan")
        else:
            anova_p = float("nan"); f_stat = float("nan")
        x2_signal_strong = (not np.isnan(anova_p) and anova_p < 0.05)
    else:
        rho_x = pval_x = anova_p = f_stat = float("nan")
        x2_signal_strong = False

    # C5 outlier check
    c5_in_pilot: list[str] = []
    if C5_JSON.exists():
        try:
            c5_data = json.loads(C5_JSON.read_text())
            c5_ids = set(str(x) for x in c5_data) if isinstance(c5_data, list) else set(str(x) for x in c5_data.keys())
            c5_in_pilot = list(c5_ids & set(df_x["sample_id"]))
            if c5_in_pilot:
                c5_df = df_x[df_x["sample_id"].isin(c5_in_pilot)]
                c5_pass0 = (c5_df["pass_rate"] == 0.0).sum()
                norms_all = df_x["mean_norm"].values
                p95_thresh = np.percentile(norms_all, 95)
                c5_highnorm = (c5_df["mean_norm"] >= p95_thresh).sum()
                _rpt(f"\n**C5 outlier re-validation**:")
                _rpt(f"- C5 IDs ∩ pilot: {len(c5_in_pilot)} samples")
                _rpt(f"- pass=0: {c5_pass0} ({c5_pass0/len(c5_df)*100:.0f}%)")
                _rpt(f"- norm ≥ p95: {c5_highnorm} ({c5_highnorm/len(c5_df)*100:.0f}%)")
        except Exception as e:
            _rpt(f"\n- C5 check skipped: {e}")
    else:
        _rpt(f"\n- C5_outlier_samples.json not found: {C5_JSON}")

    # Plot boxplot
    fig, ax = plt.subplots(figsize=(9, 4))
    bdata = [df_x[df_x["bucket"]==b]["mean_norm"].values
             for b in bucket_order if b in df_x["bucket"].values]
    blabels = [b for b in bucket_order if b in df_x["bucket"].values]
    ax.boxplot(bdata, labels=blabels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.5))
    ax.set_xlabel("Pass Rate Bucket"); ax.set_ylabel("Mean ||ΔA|| (all layers)")
    ax.set_title(f"X2: Pass Rate vs Activation Norm  (ρ={rho_x:.3f})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "X2_pass_rate_vs_activation_norm.png", dpi=120)
    plt.close(fig)
    _rpt("\n**Plots**: `plots/X2_pass_rate_vs_activation_norm.png`")

    if not np.isnan(rho_x):
        if x2_signal_strong:           x2 = "X2_SIGNAL=STRONG"
        elif abs(rho_x) > 0.05:        x2 = "X2_SIGNAL=WEAK"
        else:                          x2 = "X2_SIGNAL=NONE"
    else:
        x2 = "X2_SIGNAL=UNKNOWN"
    DECISION["X2 signal"] = x2
    _rpt(f"\n**Auto-Decision**: `{x2}`")
    record("X2", "PASS" if "STRONG" in x2 else ("WARN" if "WEAK" in x2 else "WARN"))

    # ── X3. Truncation by pass bucket ────────────────
    _rpt("\n### X3. Truncation by Pass Bucket")
    # Track A truncation count per bucket
    df_x["trunc_any"] = df_x["truncation_count"] > 0
    trunc_by_bucket = df_x.groupby("bucket")["trunc_any"].mean().reindex(bucket_order)
    _rpt("\n| Bucket | Trunc Rate (Track A) |")
    _rpt("|---|---|")
    for b, rate in trunc_by_bucket.items():
        if not np.isnan(rate):
            _rpt(f"| {b} | {rate:.3f} |")

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [trunc_by_bucket.get(b, np.nan) for b in bucket_order]
    ax.bar(range(len(bucket_order)), vals, tick_label=bucket_order, color="steelblue", edgecolor="black")
    ax.set_xlabel("Pass Rate Bucket"); ax.set_ylabel("Truncation Rate")
    ax.set_title("X3: Truncation Rate by Pass Bucket")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "X3_truncation_by_pass_bucket.png", dpi=120)
    plt.close(fig)
    _rpt("\n**Plots**: `plots/X3_truncation_by_pass_bucket.png`")
    record("X3", "PASS")

    # ── X4. Subject PCA ───────────────────────────────
    _rpt("\n### X4. Subject PCA (§8.A decision)")
    # Build subject labels aligned with mid_vecs (already in pt_files order)
    # We need to join mid_vecs with subject labels from Track A
    id2subject = dict(zip(df_a["sample_id"].astype(str), df_a["subject"]))
    subjects_b = [id2subject.get(str(sid), "Unknown") for sid in sample_ids_b]

    silhouette = float("nan")
    if HAS_SKLEARN and len(mid_vecs) == len(subjects_b):
        try:
            X = np.stack(mid_vecs, axis=0).astype(np.float32)
            pca = sk_PCA(n_components=2)
            X2d = pca.fit_transform(X)

            # encode labels
            unique_subj = sorted(set(subjects_b))
            subj2int    = {s: i for i, s in enumerate(unique_subj)}
            labels      = np.array([subj2int[s] for s in subjects_b])

            silhouette = silhouette_score(X, labels, metric="euclidean", sample_size=min(len(X), 2000))
            _rpt(f"\n- PCA explained variance (PC1+PC2): {pca.explained_variance_ratio_.sum()*100:.1f}%")
            _rpt(f"- Silhouette (subject, euclidean): **{silhouette:.4f}**")

            # plot scatter
            fig, ax = plt.subplots(figsize=(9, 7))
            cmap = plt.cm.get_cmap("tab20", len(unique_subj))
            for i, subj in enumerate(unique_subj):
                mask = np.array(subjects_b) == subj
                ax.scatter(X2d[mask, 0], X2d[mask, 1], s=4, alpha=0.4, color=cmap(i), label=subj)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"X4: Mid-layer Activation PCA by Subject  (sil={silhouette:.3f})")
            ax.legend(fontsize=5, ncol=2, markerscale=3, loc="upper right")
            fig.tight_layout(); fig.savefig(PLOTS_DIR / "X4_subject_pca_scatter.png", dpi=120)
            plt.close(fig)
            _rpt("\n**Plots**: `plots/X4_subject_pca_scatter.png`")
        except Exception as e:
            _rpt(f"\n- PCA failed: {e}")
    else:
        _rpt(f"\n- sklearn not available or shapes mismatch ({len(mid_vecs)} vecs, {len(subjects_b)} labels)")

    if not np.isnan(silhouette):
        if silhouette > 0.15:          subj = "SUBJECT=ACTIVATION"
        elif silhouette >= 0.05:       subj = "SUBJECT=MARGINAL"
        else:                          subj = "SUBJECT=GPT_LABEL"
    else:
        subj = "SUBJECT=UNKNOWN"
    DECISION["§8.A Subject"] = subj
    _rpt(f"\n**Auto-Decision**: `{subj}`")
    record("X4", "PASS")

# ─────────────────────────────────────────────────────
# Decision Sheet
# ─────────────────────────────────────────────────────
def write_decision_sheet():
    _rpt("\n---\n\n## Decision Sheet\n")

    hybrid = DECISION.get("§5.7 Hybrid", "?")
    a3     = DECISION.get("Pilot A3", "?")
    x2     = DECISION.get("X2 signal", "?")
    subj   = DECISION.get("§8.A Subject", "?")

    # Track C
    b3_ok  = all(r == "PASS" for s, r in PASS_BUDGET if s in ("B3","B2"))
    x2_ok  = "NONE" not in x2
    if b3_ok and x2_ok:
        track_c = "GO"
    elif b3_ok or x2_ok:
        track_c = "WAIT (사람 검토)"
    else:
        track_c = "STOP (method 재검토)"

    # 40K
    p0_pass = all(r == "PASS" for s, r in PASS_BUDGET if s in ("A1","B1","B5","X1"))
    a3_pass = "YES" in a3
    x2_strong = "STRONG" in x2
    if p0_pass and a3_pass and x2_strong:
        exp40k = "GO"
    elif p0_pass:
        exp40k = "MORE_ANALYSIS"
    else:
        exp40k = "STOP"

    rows = [
        ("§5.7 Hybrid difficulty", f"pass=0 ratio", hybrid, hybrid.replace("HYBRID=","")),
        ("Pilot Acceptable (A3)",  f"ρ(pass, level)", a3, a3.replace("ACCEPTABLE_A3=","")),
        ("Cross-track signal (X2)","ρ(pass, norm) + ANOVA", x2, x2.replace("X2_SIGNAL=","")),
        ("§8.A Subject axis",      "silhouette(subject)", subj, subj.replace("SUBJECT=","")),
        ("Track C 진입",           "B3 sanity + X2 signal", f"B3={'OK' if b3_ok else 'FAIL'}, X2={x2}", track_c),
        ("40K 확장 trigger",       "P0 PASS + A3 + X2 STRONG", f"P0={'OK' if p0_pass else 'FAIL'}", exp40k),
    ]

    _rpt("| Decision | Trigger | Result | Recommendation |")
    _rpt("|---|---|---|---|")
    for dec, trigger, result, rec in rows:
        _rpt(f"| {dec} | {trigger} | {result} | **{rec}** |")

    # Summary
    fail_secs = [s for s, r in PASS_BUDGET if r == "FAIL"]
    warn_secs = [s for s, r in PASS_BUDGET if r == "WARN"]
    overall   = "GO" if not fail_secs else "STOP"
    _rpt(f"\n**Overall**: `{overall}` | FAILs: {fail_secs if fail_secs else 'none'} | WARNs: {warn_secs if warn_secs else 'none'}")

    return {"track_c": track_c, "exp40k": exp40k, "overall": overall}

# ─────────────────────────────────────────────────────
# TL;DR header (written after all sections)
# ─────────────────────────────────────────────────────
def write_header(dec: dict):
    header = [
        "# Pilot 2,666 Validation Report",
        f"Generated: {datetime.now().isoformat()}",
        f"Track A source: {TRACK_A_PARQUET}",
        f"Track B source: {TRACK_B_DIR}",
        "",
        "## TL;DR",
    ]
    fail_secs = [s for s, r in PASS_BUDGET if r == "FAIL"]
    warn_secs = [s for s, r in PASS_BUDGET if r == "WARN"]
    a_status  = "✅" if not any(s.startswith("A") for s in fail_secs) else "❌"
    b_status  = "✅" if not any(s.startswith("B") for s in fail_secs) else "❌"
    x_status  = "✅" if not any(s.startswith("X") for s in fail_secs) else "❌"
    ov_emoji  = "✅" if dec["overall"] == "GO" else "❌"
    header += [
        f"- Track A: {a_status} {'Passed' if a_status=='✅' else 'Failed'}",
        f"- Track B: {b_status} {'Passed' if b_status=='✅' else 'Failed'}",
        f"- Cross-Track: {x_status} {'Passed' if x_status=='✅' else 'Failed'}",
        f"- **Overall**: `{dec['overall']}` {ov_emoji} | Track C={dec['track_c']} | 40K={dec['exp40k']}",
        "",
        "---",
        "",
        "## Detailed Sections",
        "",
    ]
    return header

# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 60)
    print("  Pilot 2,666 Validation")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 60)

    # Load reference IDs
    print("\n[LOAD] Reference pilot ID set...")
    pilot_ids = load_pilot_id_set()
    print(f"  Pilot IDs loaded: {len(pilot_ids)}")

    # Load Track A
    print("\n[LOAD] Track A parquet...")
    df_a = pd.read_parquet(TRACK_A_PARQUET)
    df_a["sample_id"] = df_a["sample_id"].astype(str)

    # Run sections
    a_res = section_A(df_a, pilot_ids)
    b_res = section_B(pilot_ids)
    section_X(df_a, b_res, pilot_ids)
    dec   = write_decision_sheet()

    # Build final report with header prepended
    header_lines = write_header(dec)
    final = header_lines + _report_lines
    REPORT_PATH.write_text("\n".join(final) + "\n", encoding="utf-8")

    elapsed = time.time() - t_start
    print(f"\n[DONE] Elapsed: {elapsed/60:.1f} min")
    print(f"Report: {REPORT_PATH}")
    print(f"Plots:  {PLOTS_DIR}")
    print(f"Spots:  {SPOT_DIR}")

    # Print TL;DR to console
    print("\n" + "="*60)
    print("DECISION SHEET")
    print("="*60)
    for dec_name, val in DECISION.items():
        print(f"  {dec_name:30s}: {val}")
    fail_secs = [s for s, r in PASS_BUDGET if r == "FAIL"]
    print(f"\n  Overall: {dec['overall']} | Track C: {dec['track_c']} | 40K: {dec['exp40k']}")
    if fail_secs: print(f"  ❌ FAILED sections: {fail_secs}")
    print("="*60)

if __name__ == "__main__":
    main()
