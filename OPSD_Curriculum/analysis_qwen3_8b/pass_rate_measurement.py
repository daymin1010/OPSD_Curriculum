#!/usr/bin/env python3
"""
pass_rate_measurement.py
========================
Qwen3-8B (non-reasoning mode) pass rate measurement on pilot 2,666 samples.

Usage:
    python pass_rate_measurement.py \
        --output_path outputs/pass_rate_pilot_2666.parquet \
        --n_rollouts 8 \
        --max_tokens 4096 \
        --tp_size 2

Smoke test (10 samples):
    python pass_rate_measurement.py \
        --n_samples 10 \
        --output_path outputs/smoke_test.parquet \
        --tp_size 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR     = Path("/scratch/lami2026/personal/jimin_2782")
TASK2_DIR    = BASE_DIR / "src/4.6_Task2"
TRAIN_L2     = TASK2_DIR / "data/fastcurl_orig/train/train_L2.parquet"
NAIT_COMMON  = TASK2_DIR / "activation/analysis"
VERL_REWARD  = BASE_DIR / "src/verl-new"

sys.path.insert(0, str(NAIT_COMMON))
sys.path.insert(0, str(VERL_REWARD))

from _nait_common import BASE_DIR as NAIT_BASE, load_metadata, resolve_shift_dirs


# ──────────────────────────────────────────────────────────────────────────────
# Reward function
# ──────────────────────────────────────────────────────────────────────────────

def load_reward_fn():
    """Load math_verify compute_score from verl or fall back to FastCuRL."""
    try:
        from verl.utils.reward_score.math_verify import compute_score as _cs
        print("[INFO] Using verl math_verify reward function")
        return _cs
    except ImportError:
        pass
    # Manual import from the installed verl in site-packages
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "math_verify",
            str(VERL_REWARD / "verl/utils/reward_score/math_verify.py")
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        print("[INFO] Using verl math_verify (file import)")
        return mod.compute_score
    except Exception as e:
        print(f"[WARN] math_verify file import failed: {e}, trying math_verify package")
        from math_verify.grader import verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

        def compute_score(model_output: str, ground_truth: str, **kw) -> float:
            gt_boxed = "\\boxed{" + ground_truth + "}"
            gold_targets = (LatexExtractionConfig(),)
            pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())
            extracted_gold = parse(gt_boxed, gold_targets)
            extracted_pred = parse(model_output, pred_targets)
            if extracted_gold and extracted_pred:
                return max(
                    1.0 if any(verify(g, p) for g in extracted_gold) else 0.0
                    for p in extracted_pred
                )
            return 0.0

        print("[INFO] Using math_verify package (direct)")
        return compute_score


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pilot_df(n_samples: int | None = None) -> pd.DataFrame:
    """Load 2,666-row pilot metadata from NAIT shift dirs, deduplicated by id."""
    dirs = resolve_shift_dirs(None) + [NAIT_BASE / "activation/full_shifts_l7l8"]
    df = load_metadata(dirs)
    # All rows with status 'ok' or 'ok (skipped)' — no 'completed' in this dataset
    print(f"[INFO] NAIT metadata rows: {len(df)} (expected 2666)")
    if n_samples is not None and n_samples < len(df):
        df = df.head(n_samples).copy()
        print(f"[INFO] Truncated to {n_samples} samples (smoke test)")
    return df


def load_train_lookup() -> dict[str, dict]:
    """Returns {index_str: {prompt_text, ground_truth, data_source}} from train_L2."""
    df = pd.read_parquet(TRAIN_L2)
    lookup = {}
    for _, row in df.iterrows():
        ei = row["extra_info"]
        idx = str(ei["index"])
        # prompt is list of dicts [{'role': 'user', 'content': ...}]
        prompt_content = row["prompt"][0]["content"]
        ground_truth   = row["reward_model"]["ground_truth"]
        data_source    = str(row.get("data_source", "unknown"))
        lookup[idx] = {
            "prompt_text":  prompt_content,
            "ground_truth": ground_truth,
            "data_source":  data_source,
        }
    print(f"[INFO] Loaded train_L2 lookup: {len(lookup)} entries")
    return lookup


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building — Qwen3 non-reasoning chat template
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, user_text: str) -> str:
    """
    Apply Qwen3 chat template with enable_thinking=False (non-reasoning mode).
    This is critical — default is thinking mode which adds <think>...</think>.
    """
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # ⚠️ Qwen3-specific: suppress CoT thinking
    )
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-8B pass rate measurement")
    p.add_argument("--n_samples",  type=int, default=None,
                   help="Limit to N samples (None = all 2,666)")
    p.add_argument("--output_path", type=str,
                   default="outputs/pass_rate_pilot_2666.parquet")
    p.add_argument("--n_rollouts",  type=int, default=8)
    p.add_argument("--max_tokens",  type=int, default=4096)
    p.add_argument("--tp_size",     type=int, default=2,
                   help="tensor_parallel_size (match GPU count: H200 2x→2, L40s 4x→4)")
    p.add_argument("--batch_size",  type=int, default=256,
                   help="Prompts per vLLM generate() call (None=all at once)")
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p",       type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # ── 1. Load reward function early (CPU-only, before GPU init)
    compute_score = load_reward_fn()

    # ── 2. Load pilot metadata
    pilot_df = load_pilot_df(args.n_samples)
    n_pilot  = len(pilot_df)
    print(f"[INFO] Pilot samples: {n_pilot}")

    # ── 3. Load train_L2 lookup
    lookup = load_train_lookup()

    # ── 4. Build sample list (filter out missing index → should be 0)
    samples = []
    missing = []
    for _, row in pilot_df.iterrows():
        sid = str(row["id"])
        if sid not in lookup:
            missing.append(sid)
            continue
        entry = lookup[sid]
        samples.append({
            "sample_id":    sid,
            "ground_truth": entry["ground_truth"],
            "prompt_text":  entry["prompt_text"],
            "data_source":  entry["data_source"],
            "subject":      str(row.get("subject", "")),
            "level":        int(row.get("level_int", 0)),
        })
    if missing:
        print(f"[WARN] {len(missing)} pilot IDs not found in train_L2: {missing[:5]}")
    print(f"[INFO] Matched samples: {len(samples)}")

    # ── 5. Initialize vLLM
    print(f"[INFO] Loading Qwen/Qwen3-8B with tp={args.tp_size} ...")
    t_model = time.time()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-8B"
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,  # bypass torch.compile / torchinductor (avoid corrupted .so cache)
        # Disable thinking-related special tokens via vLLM
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print(f"[INFO] Model loaded in {time.time() - t_model:.1f}s")

    # Verify chat template — print sample prompt (first 300 chars)
    sample_prompt = build_prompt(tokenizer, "What is 2+2?")
    print(f"[INFO] Sample chat prompt (first 300 chars):\n{sample_prompt[:300]}")
    if "<think>" in sample_prompt:
        print("[WARN] ⚠️  <think> found in prompt — enable_thinking=False may not be working!")
    else:
        print("[INFO] ✓ No <think> in prompt (enable_thinking=False confirmed)")

    # ── 6. Build formatted prompts
    formatted_prompts = [build_prompt(tokenizer, s["prompt_text"]) for s in samples]

    sampling_params = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # ── 7. Generate — vLLM handles batching internally
    print(f"[INFO] Generating {len(formatted_prompts)} × {args.n_rollouts} rollouts ...")
    t_gen = time.time()

    # Process in batches if requested (defaults to all at once)
    all_outputs = []
    bs = args.batch_size if args.batch_size > 0 else len(formatted_prompts)
    for batch_start in range(0, len(formatted_prompts), bs):
        batch_end = min(batch_start + bs, len(formatted_prompts))
        batch_prompts = formatted_prompts[batch_start:batch_end]
        print(f"  [batch] {batch_start}–{batch_end-1} / {len(formatted_prompts)}")
        batch_out = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(batch_out)

    gen_time = time.time() - t_gen
    print(f"[INFO] Generation done in {gen_time:.1f}s ({gen_time/len(samples):.2f}s/sample)")

    # ── 8. Score results
    print("[INFO] Scoring rollouts ...")
    t_score = time.time()

    rows = []
    for i, (sample, output) in enumerate(zip(samples, all_outputs)):
        rollouts = output.outputs          # list of n_rollouts CompletionOutput
        n_roll   = len(rollouts)
        gt       = sample["ground_truth"]

        # Per-rollout results
        is_correct: list[bool]  = []
        resp_lengths: list[int] = []
        truncated: list[bool]   = []
        raw_responses: list[str] = []

        for ro in rollouts:
            resp_text = ro.text
            tok_len   = len(ro.token_ids)
            is_trunc  = (ro.finish_reason == "length")

            score      = compute_score(resp_text, gt)
            correct    = (score >= 0.5)   # 1.0 = fully correct, 0.0 = wrong

            is_correct.append(correct)
            resp_lengths.append(tok_len)
            truncated.append(is_trunc)
            raw_responses.append(resp_text)

        correct_idx   = [j for j, c in enumerate(is_correct) if c]
        wrong_idx     = [j for j, c in enumerate(is_correct) if not c]
        pass_count    = sum(is_correct)
        pass_rate     = pass_count / n_roll

        mean_len      = sum(resp_lengths) / len(resp_lengths)
        mean_len_ok   = (sum(resp_lengths[j] for j in correct_idx) / len(correct_idx)
                         if correct_idx else float("nan"))
        mean_len_bad  = (sum(resp_lengths[j] for j in wrong_idx) / len(wrong_idx)
                         if wrong_idx else float("nan"))
        trunc_count   = sum(truncated)

        rows.append({
            "sample_id":                    sample["sample_id"],
            "ground_truth":                 gt,
            "pass_rate":                    pass_rate,
            "pass_count":                   pass_count,
            "correct_indices":              correct_idx,
            "mean_response_length":         mean_len,
            "mean_response_length_correct": mean_len_ok,
            "mean_response_length_incorrect": mean_len_bad,
            "truncation_count":             trunc_count,
            "raw_responses":                raw_responses,
            "subject":                      sample["subject"],
            "level":                        sample["level"],
            "data_source":                  sample["data_source"],
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(samples):
            print(f"  scored {i+1}/{len(samples)}"
                  f" | last pass_rate={pass_rate:.3f}")

    score_time = time.time() - t_score
    print(f"[INFO] Scoring done in {score_time:.1f}s")

    # ── 9. Save parquet
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_parquet(out_path, index=False)
    print(f"[INFO] Saved {len(result_df)} rows → {out_path}")

    # ── 10. Quick verification print
    total_time = time.time() - t_start
    n_total    = len(result_df)
    mean_pr    = result_df["pass_rate"].mean()
    med_pr     = result_df["pass_rate"].median()
    n_zero     = (result_df["pass_rate"] == 0.0).sum()
    n_full     = (result_df["pass_rate"] == 1.0).sum()

    print("\n" + "="*60)
    print("QUICK VERIFICATION STATS")
    print("="*60)
    print(f"Total samples     : {n_total}")
    print(f"Mean pass rate    : {mean_pr:.4f}")
    print(f"Median pass rate  : {med_pr:.4f}")
    print(f"Pass=0 samples    : {n_zero} ({n_zero/n_total*100:.1f}%)")
    print(f"Pass=8/8 samples  : {n_full} ({n_full/n_total*100:.1f}%)")
    print(f"Total wall time   : {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Time per sample   : {total_time/n_total:.1f}s")
    print(f"Estimated 2666sam : {total_time/n_total*2666/3600:.2f}h "
          f"(if smoke test)")
    print(f"\nFirst row:\n{result_df.iloc[0][['sample_id','ground_truth','pass_rate','pass_count','subject','level']].to_dict()}")
    print(f"\nResponse preview (rollout 0, first 200 chars):\n"
          f"{result_df.iloc[0]['raw_responses'][0][:200]}")

    # Check for <think> in responses (should not appear in non-reasoning mode)
    has_think = result_df["raw_responses"].apply(
        lambda rs: any("<think>" in r for r in rs)
    ).sum()
    if has_think > 0:
        print(f"\n[WARN] ⚠️  {has_think} samples have <think> in responses! Check enable_thinking=False")
    else:
        print(f"\n[INFO] ✓ No <think> found in any response (non-reasoning mode confirmed)")
    print("="*60)


if __name__ == "__main__":
    main()
