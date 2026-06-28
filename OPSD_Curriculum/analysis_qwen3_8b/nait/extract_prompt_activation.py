#!/usr/bin/env python3
"""
extract_prompt_activation.py
============================
Qwen3-8B prompt-only activation extraction on the 2,666 pilot universe.

Purpose
-------
NAIT critical-analysis baseline. Capture activation a(t_prompt_end) — the
MLP down_proj input at the LAST PROMPT TOKEN, i.e. the position immediately
BEFORE the model would start generating. No generation is performed.

This lets us compare:
    F1_subject(prompt-only)  vs  F1_subject(ΔA)
to verify whether subject signal in ΔA reflects MODEL-INTERNAL reasoning
representation or just PROMPT-KEYWORD encoding.

Hook protocol (mirrors extract_activation_shifts_qwen3_8b.py)
-------------------------------------------------------------
    register_forward_pre_hook on layer.mlp.down_proj
    capture input[0][0, prompt_len - 1, :]
    All 36 layers, bfloat16, CPU.

Output structure
----------------
    nait/outputs/prompt_act/{sample_id}.pt
        {
            "id":              str,
            "subject":         str,
            "level":           int,
            "prompt_act":      Tensor(36, intermediate_size) bfloat16  # cat over layers
            "prompt_len":      int,
            "last_token_idx":  int  (= prompt_len - 1)
        }
    nait/outputs/prompt_activation_metadata.jsonl   (per-sample status)
    nait/outputs/prompt_activation_checkpoint_chunk{0..3}.json

Usage
-----
    python nait/extract_prompt_activation.py [OPTIONS]

Options
-------
    --model-id        HF path (default: Qwen/Qwen3-8B)
    --output-dir      (default: nait/outputs/prompt_act)
    --num-samples     limit for smoke (default: None — all 2,666)
    --resume          skip already-processed
    --device          cuda or cpu (default: cuda)
    --chunk-id        0-indexed (default: 0)
    --num-chunks      total chunks (default: 1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
except ImportError as e:
    print(f"[ERROR] Missing library: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Paths (READ ONLY for 4.6_Task2 utilities)
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path("/scratch/lami2026/personal/jimin_2782")
TASK2_DIR   = BASE_DIR / "src/4.6_Task2"
NAIT_COMMON = TASK2_DIR / "activation/analysis"
TRAIN_L2    = TASK2_DIR / "data/fastcurl_orig/train/train_L2.parquet"

sys.path.insert(0, str(NAIT_COMMON))
from _nait_common import BASE_DIR as NAIT_BASE, load_metadata, resolve_shift_dirs  # noqa

CHECKPOINT_INTERVAL = 20
DEFAULT_MODEL_ID    = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT_DIR  = str(
    BASE_DIR / "src/OPSD_Curriculum/analysis_qwen3_8b/nait/outputs/prompt_act"
)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-8B prompt-only activation extraction"
    )
    p.add_argument("--model-id",     default=DEFAULT_MODEL_ID)
    p.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-samples",  type=int, default=None)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--chunk-id",     type=int, default=0)
    p.add_argument("--num-chunks",   type=int, default=1)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (mirror Track-B style)
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def format_time(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    if h: return f"{h}h {m}m {sec}s"
    if m: return f"{m}m {sec}s"
    return f"{sec}s"


def split_chunk(samples: list, cid: int, n: int) -> list:
    N = len(samples)
    csz = N // n
    rem = N % n
    start = cid * csz + min(cid, rem)
    end   = start + csz + (1 if cid < rem else 0)
    return samples[start:end]


# ══════════════════════════════════════════════════════════════════════════════
# Pilot sample loading (identical to Track A/B)
# ══════════════════════════════════════════════════════════════════════════════

def load_train_lookup() -> dict[str, str]:
    df = pd.read_parquet(TRAIN_L2)
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        idx = str(row["extra_info"]["index"])
        lookup[idx] = row["prompt"][0]["content"]
    print(f"[INFO] train_L2 lookup: {len(lookup)} entries")
    return lookup


def load_pilot_samples(num_samples: int | None = None) -> list[dict]:
    dirs  = resolve_shift_dirs(None) + [NAIT_BASE / "activation/full_shifts_l7l8"]
    df    = load_metadata(dirs)
    ok    = {"completed", "ok", "ok (skipped)"}
    df    = df[df["status"].isin(ok)].drop_duplicates(subset="id")
    print(f"[INFO] NAIT pilot metadata rows: {len(df)} (expected 2666)")

    lookup = load_train_lookup()

    samples: list[dict] = []
    missing = 0
    for _, row in df.iterrows():
        sid = str(row["id"])
        if sid not in lookup:
            missing += 1
            continue
        samples.append({
            "id":           sid,
            "subject":      str(row.get("subject", "")),
            "level":        int(row.get("level_int", row.get("level", -1))),
            "problem_text": lookup[sid],
        })
    if missing:
        warnings.warn(f"[WARN] {missing} pilot IDs missing in train_L2")
    print(f"[INFO] Loadable pilot samples: {len(samples)}")
    if num_samples is not None:
        samples = samples[:num_samples]
        print(f"[INFO] --num-samples → {len(samples)}")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint / metadata
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(d: str, cid: int) -> str:
    return os.path.join(d, f"prompt_activation_checkpoint_chunk{cid}.json")


def load_checkpoint(d: str, cid: int) -> dict:
    p = checkpoint_path(d, cid)
    if os.path.exists(p):
        with open(p) as f:
            cp = json.load(f)
        print(f"[CHECKPOINT] chunk {cid}: done={len(cp.get('processed_ids', []))}")
        return cp
    return {"last_processed_idx": -1, "processed_ids": []}


def save_checkpoint(d: str, cid: int, last_idx: int, ids: list) -> None:
    p = checkpoint_path(d, cid)
    with open(p, "w") as f:
        json.dump({"last_processed_idx": last_idx, "processed_ids": ids}, f, indent=2)


def metadata_path(d: str) -> str:
    return os.path.join(d, "prompt_activation_metadata.jsonl")


def append_metadata(d: str, rec: dict) -> None:
    with open(metadata_path(d), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Model / tokenizer
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_id: str, device: str):
    print(f"\n[MODEL] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True,
        )
    model.eval()

    num_layers = len(model.model.layers)
    inter_dim  = model.model.layers[0].mlp.down_proj.in_features
    print(f"[MODEL] num_layers={num_layers}  intermediate_size={inter_dim}")
    assert hasattr(model.model.layers[0].mlp, "down_proj"), "no down_proj!"
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Prompt build (mirror Track-B exactly)
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    import re as _re
    m = _re.search(r"<think>(.*?)</think>", prompt, flags=_re.DOTALL)
    if m is not None and m.group(1).strip() != "":
        raise RuntimeError(
            "[ASSERT] non-empty <think>...</think> in prompt — enable_thinking=False failed!"
        )
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# Prompt-only activation capture
# ══════════════════════════════════════════════════════════════════════════════

def capture_prompt_activation(
    model,
    input_ids: torch.Tensor,
    last_idx: int,
    device: str,
) -> tuple[torch.Tensor, float]:
    """
    Forward pass on prompt only. Capture mlp.down_proj input at position last_idx
    (= prompt_len - 1) for every layer.

    Returns:
        act_tensor (L, D)  bfloat16, CPU
        forward_time_sec
    """
    num_layers = len(model.model.layers)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_pre_hook(idx: int):
        def hook(module, inputs):
            x = inputs[0]  # (1, seq_len, intermediate_size)
            captured[idx] = x[0, last_idx, :].detach().to(torch.bfloat16).cpu().clone()
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(i)))

    t_fwd = time.time()
    try:
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            model.model(input_ids=input_ids, attention_mask=attn)
    finally:
        for h in hooks:
            h.remove()
    fwd_time = time.time() - t_fwd

    # Assemble (L, D)
    if len(captured) != num_layers:
        raise RuntimeError(
            f"[ERR] captured {len(captured)} layers, expected {num_layers}"
        )
    inter_dim = captured[0].shape[0]
    out = torch.empty((num_layers, inter_dim), dtype=torch.bfloat16)
    for i in range(num_layers):
        out[i] = captured[i]

    if device == "cuda":
        torch.cuda.empty_cache()
    return out, fwd_time


# ══════════════════════════════════════════════════════════════════════════════
# Single sample processing
# ══════════════════════════════════════════════════════════════════════════════

def process_sample(
    sample: dict,
    model,
    tokenizer,
    output_dir: str,
    device: str,
    chunk_id: int,
    num_chunks: int,
) -> dict:
    sid     = sample["id"]
    pt_path = os.path.join(output_dir, f"{sid}.pt")

    if os.path.exists(pt_path):
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        return {
            "id":             sid,
            "subject":        sample.get("subject", ""),
            "level":          sample.get("level", -1),
            "prompt_len":     d.get("prompt_len", -1),
            "last_token_idx": d.get("last_token_idx", -1),
            "shift_file":     pt_path,
            "chunk_id":       chunk_id,
            "num_chunks":     num_chunks,
            "forward_time_sec": d.get("forward_time_sec", -1.0),
            "status":         "ok (skipped)",
        }

    prompt = build_prompt(tokenizer, sample["problem_text"])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    last_idx   = prompt_len - 1

    act_tensor, fwd_time = capture_prompt_activation(model, input_ids, last_idx, device)

    save_data = {
        "id":               sid,
        "subject":          sample.get("subject", ""),
        "level":            sample.get("level", -1),
        "prompt_act":       act_tensor,            # (L, D) bfloat16
        "prompt_len":       prompt_len,
        "last_token_idx":   last_idx,
        "forward_time_sec": fwd_time,
    }
    torch.save(save_data, pt_path)

    return {
        "id":             sid,
        "subject":        sample.get("subject", ""),
        "level":          sample.get("level", -1),
        "prompt_len":     prompt_len,
        "last_token_idx": last_idx,
        "shift_file":     pt_path,
        "chunk_id":       chunk_id,
        "num_chunks":     num_chunks,
        "forward_time_sec": fwd_time,
        "status":         "ok",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    print("=" * 70)
    print("  Qwen3-8B Prompt-only Activation Extraction (NAIT critical baseline)")
    print(f"  model={args.model_id} output_dir={args.output_dir}")
    print(f"  device={args.device} chunk={args.chunk_id}/{args.num_chunks}")
    print(f"  resume={args.resume} num_samples={args.num_samples}")
    print("=" * 70)

    ensure_dir(args.output_dir)

    all_samples = load_pilot_samples(num_samples=args.num_samples)
    print(f"[ASSERT] pilot count = {len(all_samples)} "
          f"(expected 2666 unless --num-samples)")

    if args.num_chunks > 1:
        samples = split_chunk(all_samples, args.chunk_id, args.num_chunks)
        print(f"[CHUNK] {args.chunk_id}/{args.num_chunks} → {len(samples)} samples")
    else:
        samples = all_samples
    total = len(samples)

    cp = load_checkpoint(args.output_dir, args.chunk_id)
    processed_ids = cp.get("processed_ids", [])
    pid_set = set(processed_ids)
    if args.resume:
        remaining = [s for s in samples if s["id"] not in pid_set]
        print(f"[RESUME] done={len(processed_ids)}  remaining={len(remaining)}")
    else:
        remaining = samples

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)
    inter_dim  = model.model.layers[0].mlp.down_proj.in_features
    num_layers = len(model.model.layers)
    print(f"\n[VERIFY] num_hidden_layers={num_layers}  "
          f"intermediate_size={inter_dim}")
    if args.device == "cuda":
        m_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[GPU] after model load: alloc={m_alloc:.2f}GB")

    start = time.time()
    done = 0
    per_times = []

    for loop_idx, sample in enumerate(remaining):
        sid = sample["id"]
        elapsed = time.time() - start
        if done > 0:
            avg = elapsed / done
            eta = avg * (len(remaining) - loop_idx)
            tinfo = f"elapsed={format_time(elapsed)} ETA={format_time(eta)}"
        else:
            tinfo = f"elapsed={format_time(elapsed)}"

        print(f"\n[{loop_idx+1}/{len(remaining)}] (chunk {args.chunk_id}) "
              f"id={sid} subj={sample.get('subject','')} "
              f"lvl={sample.get('level','')}")
        print(f"  {tinfo}")

        ts = time.time()
        try:
            meta = process_sample(
                sample=sample, model=model, tokenizer=tokenizer,
                output_dir=args.output_dir, device=args.device,
                chunk_id=args.chunk_id, num_chunks=args.num_chunks,
            )
            if "skipped" not in meta.get("status", ""):
                print(f"  prompt_len={meta['prompt_len']} "
                      f"last_idx={meta['last_token_idx']} "
                      f"fwd={meta['forward_time_sec']:.2f}s")
                processed_ids.append(sid)
                pid_set.add(sid)
        except torch.cuda.OutOfMemoryError as e:
            msg = f"CUDA OOM: {str(e)[:120]}"
            warnings.warn(f"[WARN] {sid} — {msg}")
            meta = {
                "id": sid, "subject": sample.get("subject", ""),
                "level": sample.get("level", -1), "shift_file": "",
                "chunk_id": args.chunk_id, "num_chunks": args.num_chunks,
                "status": "error", "error_message": msg,
            }
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            msg = str(e)[:200]
            warnings.warn(f"[WARN] {sid} — {msg}")
            meta = {
                "id": sid, "subject": sample.get("subject", ""),
                "level": sample.get("level", -1), "shift_file": "",
                "chunk_id": args.chunk_id, "num_chunks": args.num_chunks,
                "status": "error", "error_message": msg,
            }

        per_times.append(time.time() - ts)
        append_metadata(args.output_dir, meta)
        done += 1

        if done % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(args.output_dir, args.chunk_id, loop_idx, processed_ids)
            if args.device == "cuda":
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [CKPT] saved | GPU peak={peak:.2f}GB")

    save_checkpoint(args.output_dir, args.chunk_id, total - 1, processed_ids)

    total_elapsed = time.time() - start
    import numpy as np
    times_arr = np.array(per_times) if per_times else np.array([0.0])

    print("\n" + "=" * 70)
    print(f"  Completion Summary (chunk {args.chunk_id}/{args.num_chunks})")
    print("=" * 70)
    print(f"  Processed this run:   {done:>6,}")
    print(f"  Total done:           {len(processed_ids):>6,} / {total:,}")
    print(f"  Elapsed:              {format_time(total_elapsed)}")
    print(f"  Output dir:           {os.path.abspath(args.output_dir)}")
    print(f"  Metadata:             {metadata_path(args.output_dir)}")
    print()
    print(f"  ── Per-sample time ──")
    print(f"  Mean   : {times_arr.mean():.2f}s")
    print(f"  Median : {np.median(times_arr):.2f}s")
    print(f"  P95    : {np.percentile(times_arr, 95):.2f}s")
    print(f"  Max    : {times_arr.max():.2f}s")

    if len(per_times) > 0:
        est_chunk = times_arr.mean() * 667
        est_total = times_arr.mean() * 2666
        print()
        print(f"  ── Extrapolation ──")
        print(f"  Per-chunk (667 samples):  {format_time(est_chunk)}")
        print(f"  Total (2,666 samples):    {format_time(est_total)}")

    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  GPU peak: {peak:.2f} GB (L40S=48GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
