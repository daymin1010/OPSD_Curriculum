#!/usr/bin/env python3
"""
make_smoke_report.py
====================
Merge per-rank smoke jsonl, sanity-check, and write smoke_report.md.

CPU only — run after the GPU job finishes.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics as st

import pandas as pd

BASE = Path("/scratch/lami2026/personal/jimin_2782")
MAX_NEW_TOKENS = 8192  # keep in sync with run_smoke.sh / extract_thinking_smoke.py
OUT_DIR = BASE / "src/OPSD_Curriculum/reasoning_pivot/activation/outputs"
PILOT = BASE / "src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet"
SAMPLES = OUT_DIR / "smoke_samples.parquet"
MERGED = OUT_DIR / "smoke_meta.jsonl"
REPORT = OUT_DIR / "smoke_report.md"
TEMPLATE_PREVIEW = OUT_DIR / "chat_template_preview.txt"


def load_rank_metas() -> list[dict]:
    rows = []
    for p in sorted(OUT_DIR.glob("smoke_meta_rank*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    metas = load_rank_metas()
    print(f"[INFO] loaded {len(metas)} meta rows from rank files")

    # merge → single jsonl (sorted by level)
    metas_sorted = sorted(metas, key=lambda m: (m.get("level", 0), m.get("problem_id", "")))
    with open(MERGED, "w") as f:
        for m in metas_sorted:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # sanity: 7 unique problem_id
    pids = [m["problem_id"] for m in metas]
    n_unique = len(set(pids))
    expected = pd.read_parquet(SAMPLES)["problem_id"].tolist()
    missing = sorted(set(expected) - set(pids))
    dup = len(pids) != n_unique

    ok_rows = [m for m in metas_sorted if m.get("status") == "ok"]

    # ── gates ──
    gate_count = (n_unique == len(expected) and not missing and not dup)
    gate_hook = all(m.get("dA_faithful_stats", {}).get("l2_per_layer_mean", 0) > 0
                    for m in ok_rows) if ok_rows else False
    gate_shape = all(m.get("dA_faithful_stats", {}).get("shape", [None])[0] == 36
                     for m in ok_rows) if ok_rows else False
    gate_nan = all((not m["dA_faithful_stats"]["has_nan"] and not m["dA_faithful_stats"]["has_inf"]
                    and not m["dA_thinking_stats"]["has_nan"] and not m["dA_thinking_stats"]["has_inf"])
                   for m in ok_rows) if ok_rows else False
    # think indexing is only *expected* to be valid for NON-truncated rows;
    # truncated (length-capped) generations legitimately lack </think>.
    nontrunc_rows = [m for m in ok_rows if not m.get("truncated")]
    gate_think = (all(m.get("think_valid") for m in nontrunc_rows)
                  and len(nontrunc_rows) > 0)

    # template handling
    template_injects = [m.get("template_open_in_prompt") for m in ok_rows]
    template_consistent = (len(set(template_injects)) == 1)
    gate_template = template_consistent and gate_think

    # lengths
    gen_lens = [m["gen_len"] for m in ok_rows]
    think_spans = [m["think_span_len"] for m in ok_rows if m.get("think_valid")]
    truncs = [m for m in ok_rows if m.get("truncated")]
    hard = [m for m in ok_rows if m.get("level", 0) >= 6]
    hard_trunc = [m for m in hard if m.get("truncated")]

    # pilot stats
    pdf = pd.read_parquet(PILOT)
    correct_counts = pdf["correct"].value_counts(dropna=False).to_dict()
    r1 = pdf["r1_cot_token_count"].dropna()
    r1_med = float(r1.median())
    r1_p95 = float(r1.quantile(0.95))
    r1_max = float(r1.max())

    # template preview snippet
    tprev = ""
    if TEMPLATE_PREVIEW.exists():
        txt = TEMPLATE_PREVIEW.read_text(encoding="utf-8")
        tprev = txt[-1500:] if len(txt) > 1500 else txt

    # ── build markdown ──
    L = []
    L.append("# Phase Z-1 Smoke Report — Qwen3-8B Thinking-mode Activation Extraction\n")
    L.append(f"- Script: `src/OPSD_Curriculum/reasoning_pivot/activation/extract_thinking_smoke.py`")
    L.append(f"- Samples: `{SAMPLES}`")
    L.append(f"- Shifts (.pt): `{OUT_DIR}/smoke_shifts/{{problem_id}}.pt`")
    L.append(f"- Merged meta: `{MERGED}`")
    L.append(f"- max_new_tokens: **{MAX_NEW_TOKENS}**, sampling: do_sample=True temp=0.6 top_p=0.95 top_k=20")
    L.append(f"- GPU: L40s ×2 (iREMB-C-07), per-sample seed = int(problem_id[:8],16) % 2**31\n")

    # gates
    L.append("## Pass/Fail Gates\n")
    def mark(b): return "✅ PASS" if b else "❌ FAIL"
    L.append(f"| Gate | Result |")
    L.append(f"|---|---|")
    L.append(f"| count: 7 unique problem_id, no dup/missing | {mark(gate_count)} |")
    L.append(f"| hooks fire (dA_faithful L2 > 0 all) | {mark(gate_hook)} |")
    L.append(f"| ΔA shape [36, inter] all | {mark(gate_shape)} |")
    L.append(f"| no NaN/Inf (both ΔA) | {mark(gate_nan)} |")
    L.append(f"| think indexing valid (0<t1<tK<total) for non-truncated | {mark(gate_think)} |")
    L.append(f"| template `<think>` handling clear + consistent | {mark(gate_template)} |")
    overall = all([gate_count, gate_hook, gate_shape, gate_nan, gate_think, gate_template])
    L.append(f"\n**OVERALL: {mark(overall)}**\n")
    if missing:
        L.append(f"> ⚠️ missing problem_ids: {missing}")
    if dup:
        L.append(f"> ⚠️ duplicate problem_ids present")
    # report any non-ok rows
    bad = [m for m in metas if m.get("status") != "ok"]
    if bad:
        L.append(f"> ⚠️ {len(bad)} non-ok rows:")
        for m in bad:
            L.append(f">   - {m.get('problem_id')} (L{m.get('level')}): "
                     f"{m.get('status')} {m.get('error','')[:120]}")

    # template handling
    L.append("\n## Chat Template `<think>` Handling\n")
    if ok_rows:
        inj = ok_rows[0].get("template_open_in_prompt")
        L.append(f"- `apply_chat_template(enable_thinking=True)` injects `<think>` into prompt: "
                 f"**{'YES' if inj else 'NO'}** (consistent across samples: {template_consistent})")
        o = ok_rows[0]
        L.append(f"- `<think>` token id = {o.get('open_id')} (single_token={o.get('open_single')}, ids={o.get('open_ids')})")
        L.append(f"- `</think>` token id = {o.get('close_id')} (single_token={o.get('close_single')}, ids={o.get('close_ids')})")
    if tprev:
        L.append("\n<details><summary>chat_template_preview.txt (tail)</summary>\n")
        L.append("```")
        L.append(tprev)
        L.append("```\n</details>")

    # sample table
    L.append("\n## Sample Table\n")
    L.append("| problem_id | level | subject | prompt_len | gen_len | total_len | open_idx | close_idx | think_span | finish | truncated | think_valid |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in metas_sorted:
        if m.get("status") != "ok":
            L.append(f"| {m.get('problem_id')} | {m.get('level')} | - | - | - | - | - | - | - | {m.get('status')} | - | - |")
            continue
        L.append(f"| {m['problem_id']} | {m['level']} | {m.get('subject','')} | "
                 f"{m['prompt_len']} | {m['gen_len']} | {m['total_len']} | "
                 f"{m['open_idx']} | {m['close_idx']} | {m['think_span_len']} | "
                 f"{m['finish_reason']} | {m['truncated']} | {m['think_valid']} |")

    # length summary
    L.append("\n## Length Summary\n")
    if gen_lens:
        L.append(f"- gen_len: median={st.median(gen_lens):.0f}, p95={pct(gen_lens,0.95):.0f}, max={max(gen_lens)}")
    if think_spans:
        L.append(f"- think_span_len: median={st.median(think_spans):.0f}, p95={pct(think_spans,0.95):.0f}, max={max(think_spans)}")
    L.append(f"- truncation: **{len(truncs)}/{len(ok_rows)}** hit {MAX_NEW_TOKENS} cap "
             f"(hard L6-L8: {len(hard_trunc)}/{len(hard)})")

    # ΔA norms
    L.append("\n## ΔA Norm Comparison (faithful vs thinking)\n")
    L.append("Per-sample mean per-layer L2 norm:\n")
    L.append("| problem_id | level | ‖dA_faithful‖ (mean/layer) | ‖dA_thinking‖ (mean/layer) | think_valid |")
    L.append("|---|---|---|---|---|")
    for m in metas_sorted:
        if m.get("status") != "ok":
            continue
        L.append(f"| {m['problem_id']} | {m['level']} | "
                 f"{m['dA_faithful_stats']['l2_per_layer_mean']:.3f} | "
                 f"{m['dA_thinking_stats']['l2_per_layer_mean']:.3f} | {m['think_valid']} |")
    if ok_rows:
        f_means = [m['dA_faithful_stats']['l2_per_layer_mean'] for m in ok_rows]
        t_means = [m['dA_thinking_stats']['l2_per_layer_mean'] for m in ok_rows if m['think_valid']]
        L.append(f"\n- mean ‖dA_faithful‖ across samples = {st.mean(f_means):.3f}")
        if t_means:
            L.append(f"- mean ‖dA_thinking‖ across samples = {st.mean(t_means):.3f}")
    L.append(f"\n- dtype: float32, shape: [36, intermediate_size] (per sample, both ΔA)")

    # pilot universe stats
    L.append("\n## Pilot Universe Context\n")
    L.append(f"- `correct` value_counts: {correct_counts} "
             f"→ **JSON-parsing-success flag only, NOT a correctness signal**; "
             f"not usable as difficulty-bias indicator.")
    L.append(f"- `r1_cot_token_count` (DeepSeek-R1 CoT len): median={r1_med:.0f}, "
             f"p95={r1_p95:.0f}, max={r1_max:.0f}")

    # max_token opinion
    L.append(f"\n## max_new_tokens = {MAX_NEW_TOKENS} Adequacy Opinion\n")
    if ok_rows:
        if len(hard_trunc) == 0:
            L.append(f"- No hard sample (L6-L8) hit the {MAX_NEW_TOKENS} cap. "
                     f"→ **{MAX_NEW_TOKENS} appears adequate for smoke**; confirm on full pilot length distribution in Z-2.")
        elif len(hard_trunc) >= max(1, len(hard) // 2):
            L.append(f"- {len(hard_trunc)}/{len(hard)} hard samples hit the {MAX_NEW_TOKENS} cap (truncated). "
                     f"→ even {MAX_NEW_TOKENS} is insufficient for hard MATH thinking. "
                     f"**Adopt a 'truncated-excluded' rule for think-span ΔA** (faithful ΔA still defined), "
                     f"and/or raise cap further only if hard-level think-span ΔA is required.")
        else:
            L.append(f"- {len(hard_trunc)}/{len(hard)} hard samples truncated. "
                     f"→ borderline; flag truncated and exclude from think-span ΔA on Z-2.")
    L.append(f"\n> Note: Qwen3 thinking generations can exceed DeepSeek-R1 CoT lengths; r1_cot is only a rough prior.")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[OK] wrote {REPORT}")
    print(f"[OK] merged meta → {MERGED}")
    print(f"\nOVERALL GATE: {'PASS' if overall else 'FAIL'}")
    print(f"  count={gate_count} hook={gate_hook} shape={gate_shape} "
          f"nan={gate_nan} think={gate_think} template={gate_template}")


if __name__ == "__main__":
    main()
