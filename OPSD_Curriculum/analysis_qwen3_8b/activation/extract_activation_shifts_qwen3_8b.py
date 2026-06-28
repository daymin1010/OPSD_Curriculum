#!/usr/bin/env python3
"""
extract_activation_shifts_qwen3_8b.py
======================================
Qwen3-8B (non-reasoning mode) NAIT activation shift extraction
on the 2,666 pilot sample universe.

NAIT Shift Definition (mirrors 4.6_Task2 protocol exactly):
    Δ𝒜 = 𝒜(t_K) − 𝒜(t_1)
    - t_1  : first generated token position (= prompt_len)
    - t_K  : last token position           (= seq_len - 1)
    - 𝒜    : input activation to MLP down_proj of each layer
              (register_forward_pre_hook on layer.mlp.down_proj)

Input sequence: prompt + y_hat  (model.generate() output, NOT ground truth y*)
Generation:     Greedy (do_sample=False) — deterministic single trajectory
Chat template:  enable_thinking=False  (non-reasoning mode, no <think> tags)

Usage:
    python activation/extract_activation_shifts_qwen3_8b.py [OPTIONS]

Options:
    --model-id          HF model path (default: Qwen/Qwen3-8B)
    --output-dir        output directory (default: activation/outputs/shifts)
    --max-new-tokens    max generation tokens (default: 4096)
    --resume            skip already-processed samples
    --num-samples       limit samples (smoke test)
    --device            cuda or cpu (default: cuda)
    --chunk-id          0-indexed chunk (default: 0)
    --num-chunks        total chunks for data-parallel (default: 1)

Requirements:
    pip install torch transformers accelerate pandas pyarrow
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
    print("        pip install torch transformers accelerate pandas pyarrow")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Paths  (READ ONLY — do not modify 4.6_Task2)
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR     = Path("/scratch/lami2026/personal/jimin_2782")
TASK2_DIR    = BASE_DIR / "src/4.6_Task2"
NAIT_COMMON  = TASK2_DIR / "activation/analysis"
TRAIN_L2     = TASK2_DIR / "data/fastcurl_orig/train/train_L2.parquet"

# Inject _nait_common into path (READ ONLY)
sys.path.insert(0, str(NAIT_COMMON))
from _nait_common import BASE_DIR as NAIT_BASE, load_metadata, resolve_shift_dirs  # noqa

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_INTERVAL = 5
DEFAULT_MODEL_ID    = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT_DIR  = str(
    BASE_DIR / "src/OPSD_Curriculum/analysis_qwen3_8b/activation/outputs/shifts"
)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-8B NAIT activation shift extraction (non-reasoning mode)"
    )
    p.add_argument("--model-id",       default=DEFAULT_MODEL_ID)
    p.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--resume",         action="store_true")
    p.add_argument("--num-samples",    type=int, default=None,
                   help="Limit #samples (smoke test). Applied BEFORE chunk split.")
    p.add_argument("--device",         default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--chunk-id",       type=int, default=0)
    p.add_argument("--num-chunks",     type=int, default=1)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def split_into_chunk(samples: list, chunk_id: int, num_chunks: int) -> list:
    n   = len(samples)
    csz = n // num_chunks
    rem = n % num_chunks
    start = chunk_id * csz + min(chunk_id, rem)
    end   = start + csz + (1 if chunk_id < rem else 0)
    return samples[start:end]


# ══════════════════════════════════════════════════════════════════════════════
# Pilot sample loading  (same logic as Track A pass_rate_measurement.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_train_lookup() -> dict[str, str]:
    """Returns {index_str: problem_text} from train_L2.parquet."""
    df = pd.read_parquet(TRAIN_L2)
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        ei  = row["extra_info"]
        idx = str(ei["index"])
        # prompt is list of dicts [{'role': 'user', 'content': ...}]
        lookup[idx] = row["prompt"][0]["content"]
    print(f"[INFO] train_L2 lookup: {len(lookup)} entries")
    return lookup


def load_pilot_samples(num_samples: int | None = None) -> list[dict]:
    """
    Load 2,666-row pilot universe via _nait_common.load_metadata()
    (identical to Track A pass_rate_measurement.py).

    Returns list of dicts: {id, subject, level, problem_text}
    """
    # ── NAIT shift dirs (same as Track A) ─────────────────────────────────
    dirs  = resolve_shift_dirs(None) + [NAIT_BASE / "activation/full_shifts_l7l8"]
    df    = load_metadata(dirs)
    print(f"[INFO] NAIT metadata rows: {len(df)} (expected 2666)")

    # ── Join with train_L2 to get problem_text ─────────────────────────────
    lookup = load_train_lookup()

    samples = []
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
        warnings.warn(f"[WARN] {missing} pilot IDs not found in train_L2 (check parquet)")
    print(f"[INFO] Loadable pilot samples: {len(samples)}")

    if num_samples is not None:
        samples = samples[:num_samples]
        print(f"[INFO] --num-samples limit: {len(samples)} (smoke test)")

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing  (per-chunk, mirrors 4.6 protocol)
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(output_dir: str, chunk_id: int) -> str:
    return os.path.join(output_dir, f"shifts_checkpoint_chunk{chunk_id}.json")


def load_checkpoint(output_dir: str, chunk_id: int) -> dict:
    path = checkpoint_path(output_dir, chunk_id)
    if os.path.exists(path):
        with open(path) as f:
            cp = json.load(f)
        print(f"[CHECKPOINT] Loaded chunk {chunk_id}: "
              f"last_idx={cp.get('last_processed_idx', -1)}, "
              f"done={len(cp.get('processed_ids', []))}")
        return cp
    return {"last_processed_idx": -1, "processed_ids": []}


def save_checkpoint(output_dir: str, chunk_id: int, last_idx: int,
                    processed_ids: list) -> None:
    path = checkpoint_path(output_dir, chunk_id)
    with open(path, "w") as f:
        json.dump({"last_processed_idx": last_idx,
                   "processed_ids": processed_ids}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Metadata log  (shared file, chunks all append)
# ══════════════════════════════════════════════════════════════════════════════

def metadata_path(output_dir: str) -> str:
    return os.path.join(output_dir, "shifts_metadata.jsonl")


def append_metadata(output_dir: str, record: dict) -> None:
    with open(metadata_path(output_dir), "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_id: str, device: str):
    print(f"\n[MODEL] Loading {model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id
        print("[MODEL] pad_token set to eos_token")

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,   # bfloat16 (Track A와 동일)
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model.eval()

    num_layers       = len(model.model.layers)
    intermediate_dim = model.model.layers[0].mlp.down_proj.in_features

    print(f"[MODEL] Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    print(f"[MODEL] num_hidden_layers:  {num_layers}   (expected 36 for Qwen3-8B)")
    print(f"[MODEL] intermediate_size:  {intermediate_dim}  (down_proj.in_features)")
    print(f"[MODEL] hidden_size:         {model.config.hidden_size}")

    # Verify down_proj exists on layer 0
    assert hasattr(model.model.layers[0].mlp, "down_proj"), \
        "Layer 0 has no mlp.down_proj — model structure mismatch!"

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Prompt construction  (non-reasoning mode, 4.6 protocol — no system message)
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, problem: str) -> str:
    """
    Qwen3 non-reasoning chat template.
    - No system message (mirrors 4.6 Deepseek/NAIT protocol)
    - enable_thinking=False  → no <think>...</think> wrapper
    - add_generation_prompt=True  → appends <|im_start|>assistant\n

    Asserts that <think> / </think> tags are NOT present in the resulting prompt.
    """
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Safety assert — Qwen3 with enable_thinking=False still prefills an EMPTY
    # <think>\n\n</think> block (this is the canonical non-thinking format and
    # is *correct* behavior). We only fail if there is non-whitespace content
    # inside the <think>...</think> wrapper, which would indicate the template
    # actually emitted reasoning content.
    import re as _re
    m = _re.search(r"<think>(.*?)</think>", prompt, flags=_re.DOTALL)
    if m is not None and m.group(1).strip() != "":
        raise RuntimeError(
            "[ASSERT] non-empty <think>...</think> content found in prompt — "
            "enable_thinking=False did not take effect!"
        )
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# Response generation  (Greedy — mirrors 4.6 do_sample=False protocol)
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple:
    """
    Greedy generation (do_sample=False), mirrors 4.6 protocol exactly.

    Returns:
        full_input_ids  : (1, seq_len) — [prompt tokens + generated tokens]
        prompt_len      : int          — number of prompt tokens (= t1_idx)
        generated_text  : str
        gen_time_sec    : float
        num_generated   : int
        is_trunc        : bool         — explicit truncation flag (new in this script)
    """
    inputs         = tokenizer(prompt, return_tensors="pt")
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len     = input_ids.shape[1]

    t_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # Greedy — 4.6 protocol
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time_sec = time.time() - t_start

    generated_ids  = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_generated  = generated_ids.shape[0]
    full_input_ids = output_ids  # (1, prompt_len + num_generated)

    # Explicit is_trunc:  max tokens hit AND last token is not EOS
    last_token_id = full_input_ids[0, -1].item()
    is_trunc = (num_generated >= max_new_tokens) and (last_token_id != tokenizer.eos_token_id)

    if device == "cuda":
        torch.cuda.empty_cache()

    return full_input_ids, prompt_len, generated_text, gen_time_sec, num_generated, is_trunc


# ══════════════════════════════════════════════════════════════════════════════
# Activation capture  (NAIT Δ𝒜 — mirrors 4.6 protocol exactly)
# ══════════════════════════════════════════════════════════════════════════════

def capture_activation_shifts(
    model,
    full_input_ids: torch.Tensor,
    t1_idx: int,
    tK_idx: int,
    device: str,
) -> tuple[dict, float]:
    """
    Forward pass on [prompt + y_hat], hooks on layer.mlp.down_proj (pre-hook).
    Captures activation at t1 and tK, returns shifts = act(tK) - act(t1).

    Hook location: register_forward_pre_hook on each layer.mlp.down_proj
    Matches 4.6_Task2/activation/extract_activation_shifts.py exactly.

    Returns:
        shifts           : {layer_idx (int): Tensor(intermediate_size,)} — bfloat16, CPU
        forward_time_sec : float
    """
    num_layers  = len(model.model.layers)
    activations: dict = {}
    hooks       = []

    def make_pre_hook(layer_idx: int):
        def hook(module, inputs):
            x = inputs[0]  # (1, seq_len, intermediate_size)
            activations[layer_idx] = {
                "t1": x[0, t1_idx, :].detach().cpu().clone(),
                "tK": x[0, tK_idx, :].detach().cpu().clone(),
            }
        return hook

    # Register hooks on ALL layers (0 .. num_layers-1)
    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(i))
        hooks.append(h)

    t_fwd = time.time()
    try:
        attn_mask = torch.ones_like(full_input_ids)
        with torch.no_grad():
            model.model(
                input_ids=full_input_ids,
                attention_mask=attn_mask,
            )
    finally:
        for h in hooks:
            h.remove()
    forward_time_sec = time.time() - t_fwd

    # Δ𝒜 = 𝒜(tK) − 𝒜(t1)
    shifts: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        if layer_idx in activations:
            shifts[layer_idx] = activations[layer_idx]["tK"] - activations[layer_idx]["t1"]
        else:
            warnings.warn(f"[WARN] layer {layer_idx} activation not captured!")

    del activations
    if device == "cuda":
        torch.cuda.empty_cache()

    return shifts, forward_time_sec


# ══════════════════════════════════════════════════════════════════════════════
# Single sample processing
# ══════════════════════════════════════════════════════════════════════════════

def process_sample(
    sample: dict,
    model,
    tokenizer,
    output_dir: str,
    max_new_tokens: int,
    device: str,
    chunk_id: int,
    num_chunks: int,
) -> dict:
    """
    Process one sample: generate → forward → capture shifts → save .pt.
    Returns metadata dict.
    """
    sample_id = sample["id"]
    pt_path   = os.path.join(output_dir, f"{sample_id}.pt")

    # ── Skip if already done ───────────────────────────────────────────────
    if os.path.exists(pt_path):
        print(f"  → already exists: {pt_path} (skip)")
        saved = torch.load(pt_path, map_location="cpu", weights_only=False)
        return {
            "id":                   sample_id,
            "subject":              sample.get("subject", ""),
            "level":                sample.get("level", -1),
            "t1_idx":               saved.get("t1_idx", -1),
            "tK_idx":               saved.get("tK_idx", -1),
            "num_generated_tokens": saved.get("num_generated_tokens", -1),
            "is_trunc":             saved.get("is_trunc", False),
            "shift_file":           pt_path,
            "chunk_id":             chunk_id,
            "num_chunks":           num_chunks,
            "gen_time_sec":         saved.get("gen_time_sec", -1.0),
            "forward_time_sec":     saved.get("forward_time_sec", -1.0),
            "total_time_sec":       saved.get("total_time_sec", -1.0),
            "status":               "ok (skipped)",
        }

    # ── Build prompt (non-reasoning mode) ──────────────────────────────────
    prompt = build_prompt(tokenizer, sample["problem_text"])

    # ── Generate (Greedy) ──────────────────────────────────────────────────
    (full_input_ids, prompt_len, generated_text,
     gen_time_sec, num_generated, is_trunc) = generate_response(
        model, tokenizer, prompt, max_new_tokens, device
    )

    seq_len = full_input_ids.shape[1]
    t1_idx  = prompt_len       # first generated token (NAIT t_1)
    tK_idx  = seq_len - 1      # last token          (NAIT t_K)

    # Edge case: only 1 generated token → shift = 0
    if t1_idx >= tK_idx:
        tK_idx = t1_idx

    # ── Capture activations & compute Δ𝒜 ───────────────────────────────────
    shifts, forward_time_sec = capture_activation_shifts(
        model, full_input_ids, t1_idx, tK_idx, device
    )
    total_time_sec = gen_time_sec + forward_time_sec

    # ── Save per-sample .pt ────────────────────────────────────────────────
    save_data = {
        "id":                   sample_id,
        "subject":              sample.get("subject", ""),
        "level":                sample.get("level", -1),
        "shifts":               shifts,          # {int: Tensor(intermediate_size,)}
        "t1_idx":               t1_idx,
        "tK_idx":               tK_idx,
        "num_generated_tokens": num_generated,
        "is_trunc":             is_trunc,
        "generated_text":       generated_text,
        "gen_time_sec":         gen_time_sec,
        "forward_time_sec":     forward_time_sec,
        "total_time_sec":       total_time_sec,
    }
    torch.save(save_data, pt_path)

    return {
        "id":                   sample_id,
        "subject":              sample.get("subject", ""),
        "level":                sample.get("level", -1),
        "t1_idx":               t1_idx,
        "tK_idx":               tK_idx,
        "num_generated_tokens": num_generated,
        "is_trunc":             is_trunc,
        "shift_file":           pt_path,
        "chunk_id":             chunk_id,
        "num_chunks":           num_chunks,
        "gen_time_sec":         gen_time_sec,
        "forward_time_sec":     forward_time_sec,
        "total_time_sec":       total_time_sec,
        "status":               "ok",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 70)
    print("  Qwen3-8B NAIT Activation Shift Extraction (non-reasoning mode)")
    print(f"  model:         {args.model_id}")
    print(f"  output_dir:    {args.output_dir}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  device:        {args.device}")
    print(f"  chunk:         {args.chunk_id} / {args.num_chunks}")
    print(f"  resume:        {args.resume}")
    print("=" * 70)

    ensure_dir(args.output_dir)

    # ── Load all 2,666 pilot samples ────────────────────────────────────────
    all_samples = load_pilot_samples(num_samples=args.num_samples)

    # ── Assert Track A ID consistency ──────────────────────────────────────
    # (Track A uses same NAIT metadata dirs → same ID set)
    # We just re-use the same source, so the IDs ARE identical by construction.
    # Emit count for log.
    print(f"[ASSERT] Pilot sample count: {len(all_samples)} "
          f"(expected 2666 unless --num-samples set)")

    # ── Chunk split ─────────────────────────────────────────────────────────
    if args.num_chunks > 1:
        samples = split_into_chunk(all_samples, args.chunk_id, args.num_chunks)
        print(f"[CHUNK] chunk {args.chunk_id}/{args.num_chunks}: "
              f"{len(samples)} samples")
    else:
        samples = all_samples

    total = len(samples)

    # ── Checkpoint ──────────────────────────────────────────────────────────
    cp              = load_checkpoint(args.output_dir, args.chunk_id)
    processed_ids   = cp.get("processed_ids", [])
    processed_id_set = set(processed_ids)

    if args.resume:
        remaining = [s for s in samples if s["id"] not in processed_id_set]
        print(f"[RESUME] already done: {len(processed_ids)}, remaining: {len(remaining)}")
    else:
        remaining = samples

    # ── Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)

    # Print intermediate_size for smoke-test checklist
    intermediate_size = model.model.layers[0].mlp.down_proj.in_features
    num_layers        = len(model.model.layers)
    print(f"\n[VERIFY] intermediate_size  = {intermediate_size} "
          f"(Qwen3-8B expected ≈ 22528)")
    print(f"[VERIFY] num_hidden_layers  = {num_layers} "
          f"(Qwen3-8B expected = 36)")

    # GPU memory baseline
    if args.device == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserv = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU] Memory after model load: "
              f"allocated={mem_alloc:.2f}GB  reserved={mem_reserv:.2f}GB")

    # ── Processing loop ─────────────────────────────────────────────────────
    start_time = time.time()
    done_count = 0
    per_sample_times = []

    for loop_idx, sample in enumerate(remaining):
        sample_id = sample["id"]
        elapsed   = time.time() - start_time

        if done_count > 0:
            avg_sec    = elapsed / done_count
            eta        = avg_sec * (len(remaining) - loop_idx)
            time_info  = f"elapsed={format_time(elapsed)} ETA={format_time(eta)}"
        else:
            time_info = f"elapsed={format_time(elapsed)}"

        print(f"\n[{loop_idx+1}/{len(remaining)}] (chunk {args.chunk_id}: {loop_idx+1}/{total}) "
              f"id={sample_id} subject={sample.get('subject','')} level={sample.get('level','')}")
        print(f"  {time_info}")

        t_sample_start = time.time()
        try:
            meta = process_sample(
                sample=sample,
                model=model,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                chunk_id=args.chunk_id,
                num_chunks=args.num_chunks,
            )
            if "skipped" not in meta.get("status", ""):
                trunc_mark = " [TRUNC]" if meta.get("is_trunc") else ""
                print(f"  gen={meta['num_generated_tokens']:,} tok{trunc_mark} | "
                      f"t1={meta['t1_idx']} tK={meta['tK_idx']} | "
                      f"gen={meta['gen_time_sec']:.1f}s fwd={meta['forward_time_sec']:.1f}s")
                processed_ids.append(sample_id)
                processed_id_set.add(sample_id)

        except torch.cuda.OutOfMemoryError as e:
            msg = f"CUDA OOM: {str(e)[:120]}"
            warnings.warn(f"[WARN] {sample_id} — {msg}")
            meta = {
                "id": sample_id,
                "subject": sample.get("subject", ""),
                "level": sample.get("level", -1),
                "shift_file": "",
                "chunk_id": args.chunk_id,
                "num_chunks": args.num_chunks,
                "status": "error",
                "error_message": msg,
            }
            if args.device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            msg = str(e)[:200]
            warnings.warn(f"[WARN] {sample_id} — {msg}")
            meta = {
                "id": sample_id,
                "subject": sample.get("subject", ""),
                "level": sample.get("level", -1),
                "shift_file": "",
                "chunk_id": args.chunk_id,
                "num_chunks": args.num_chunks,
                "status": "error",
                "error_message": msg,
            }

        t_sample_elapsed = time.time() - t_sample_start
        per_sample_times.append(t_sample_elapsed)

        append_metadata(args.output_dir, meta)
        done_count += 1

        if done_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(args.output_dir, args.chunk_id, loop_idx, processed_ids)
            if args.device == "cuda":
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [CHECKPOINT] saved | GPU peak={mem_peak:.2f}GB")
            else:
                print(f"  [CHECKPOINT] saved")

    # Final checkpoint
    save_checkpoint(args.output_dir, args.chunk_id, total - 1, processed_ids)

    # ── Summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    import numpy as np
    times_arr = np.array(per_sample_times) if per_sample_times else np.array([0.0])

    print("\n" + "=" * 70)
    print(f"  Completion Summary (chunk {args.chunk_id}/{args.num_chunks})")
    print("=" * 70)
    print(f"  Processed this run:   {done_count:>6,}")
    print(f"  Total done:           {len(processed_ids):>6,} / {total:,}")
    print(f"  Elapsed:              {format_time(total_elapsed)}")
    print(f"  Output dir:           {os.path.abspath(args.output_dir)}")
    print(f"  Metadata:             {metadata_path(args.output_dir)}")
    print()
    print(f"  ── Per-sample time (analysis) ──")
    print(f"  Mean   : {times_arr.mean():.1f}s")
    print(f"  Median : {np.median(times_arr):.1f}s")
    print(f"  P95    : {np.percentile(times_arr, 95):.1f}s")
    print(f"  Max    : {times_arr.max():.1f}s")

    # Extrapolate wallclock for 2,666 / 4 chunks
    if len(per_sample_times) > 0:
        est_chunk = times_arr.mean() * 666
        est_total = times_arr.mean() * 2666
        print()
        print(f"  ── Pilot extrapolation ──")
        print(f"  Estimated per-chunk (666 samples): {format_time(est_chunk)}")
        print(f"  Estimated total (2,666 samples):   {format_time(est_total)}")
        print(f"  Recommended sbatch walltime per chunk: "
              f"{format_time(est_chunk * 1.5)} (×1.5 safety margin)")

    if args.device == "cuda":
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print()
        print(f"  ── GPU ──")
        print(f"  Peak memory: {mem_peak:.2f} GB (L40S has 48 GB)")
        print(f"  Margin:      {48 - mem_peak:.1f} GB")

    print("=" * 70)


if __name__ == "__main__":
    main()
