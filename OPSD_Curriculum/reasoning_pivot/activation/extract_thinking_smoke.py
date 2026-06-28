#!/usr/bin/env python3
"""
extract_thinking_smoke.py
=========================
Phase Z-1 SMOKE — Qwen3-8B thinking-mode activation extraction sanity.

Goal (NOT statistical conclusion, just pipeline sanity):
  (1) thinking length distribution  → inform max_new_tokens
  (2) extraction pipeline sanity     → hooks fire, ΔA shape/dtype, no NaN/Inf
  (3) <think>/</think> indexing       → marker token ids + positions + two-span ΔA

This script processes ONE shard (--rank of --world-size) of the smoke sample
parquet on a single GPU (CUDA_VISIBLE_DEVICES set by the launcher).

ΔA definitions (computed BOTH, per layer → shape [num_layers, intermediate]):
  - NAIT-faithful:        t1 = position 0 (first prompt token),
                          tK = last generated token (total_len - 1)
  - NAIT-inspired(think): t1 = <think>+1 token,
                          tK = </think>-1 token

Also stores 5 raw per-position activations (for Phase Critical / Z-3 reuse):
  A_pos0, A_prompt_last (prompt_len-1), A_t1_think, A_tK_think, A_last

Activation hook: register_forward_pre_hook on each layer.mlp.down_proj,
input[0] = (1, seq_len, intermediate_size). All layers, all positions captured,
indexed immediately to the positions of interest.

Reference (read-only, not modified):
  src/OPSD_Curriculum/analysis_qwen3_8b/activation/extract_activation_shifts_qwen3_8b.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
# Qwen3 thinking-mode recommended sampling (official docs):
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.95
GEN_TOP_K = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-8B thinking-mode smoke extraction (1 shard)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--samples-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    return p.parse_args()


def per_sample_seed(problem_id: str) -> int:
    """Deterministic per-sample seed (sharding-invariant)."""
    return int(problem_id[:8], 16) % (2 ** 31)


def find_first(seq: list[int], token_id: int) -> int:
    """Return index of first occurrence of token_id in seq, else -1."""
    try:
        return seq.index(token_id)
    except ValueError:
        return -1


def count_occurrences(seq: list[int], token_id: int) -> int:
    return sum(1 for t in seq if t == token_id)


def encode_marker(tokenizer, text: str) -> tuple[list[int], bool, int]:
    """Encode a marker string without special tokens.
    Returns (ids, is_single_token, primary_id)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    is_single = (len(ids) == 1)
    primary = ids[0] if len(ids) >= 1 else -1
    return ids, is_single, primary


def load_model(model_id: str, max_retries: int = 4):
    print(f"[MODEL] loading {model_id} (bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    # Retry wrapper: concurrent CUDA-context init across ranks can raise
    # cudaErrorDevicesUnavailable ("device(s) busy or unavailable") during
    # caching_allocator_warmup. Retry with exponential backoff.
    model = None
    last_err = None
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
            )
            break
        except Exception as e:  # noqa: BLE001 - want broad CUDA/Runtime catch
            last_err = e
            wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
            print(f"[MODEL] load attempt {attempt+1}/{max_retries} failed: "
                  f"{str(e)[:160]} → retry in {wait}s", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(wait)
    if model is None:
        raise RuntimeError(f"model load failed after {max_retries} attempts: {last_err}")
    model.eval()
    num_layers = len(model.model.layers)
    inter = model.model.layers[0].mlp.down_proj.in_features
    print(f"[MODEL] num_layers={num_layers} intermediate_size={inter} "
          f"hidden={model.config.hidden_size}", flush=True)
    return model, tok, num_layers, inter


def capture_positions(model, full_ids: torch.Tensor, positions: dict[str, int],
                      num_layers: int) -> dict[str, torch.Tensor]:
    """Forward pass with pre-hooks on every layer.mlp.down_proj.
    positions: {name: idx}. Returns {name: Tensor[num_layers, intermediate] float32}.
    """
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, inputs):
            # inputs[0]: (1, seq_len, intermediate_size)
            captured[layer_idx] = inputs[0][0].detach()  # (seq_len, inter) bf16 on GPU
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(i)))

    try:
        attn = torch.ones_like(full_ids)
        with torch.no_grad():
            model.model(input_ids=full_ids, attention_mask=attn)
    finally:
        for h in hooks:
            h.remove()

    # Build {name: [num_layers, inter] float32}
    out: dict[str, torch.Tensor] = {}
    seq_len = full_ids.shape[1]
    for name, idx in positions.items():
        idx = max(0, min(idx, seq_len - 1))
        rows = []
        for l in range(num_layers):
            rows.append(captured[l][idx].to(torch.float32).cpu())
        out[name] = torch.stack(rows, dim=0)  # [num_layers, inter]
    del captured
    torch.cuda.empty_cache()
    return out


def process_sample(sample: dict, model, tokenizer, num_layers: int,
                   inter: int, max_new_tokens: int, output_dir: Path,
                   dump_template_to: Path | None) -> dict:
    pid = sample["problem_id"]
    level = int(sample["level"])
    problem = sample["problem_text"]

    messages = [{"role": "user", "content": problem}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if dump_template_to is not None:
        dump_template_to.write_text(
            f"=== chat_template_preview (problem_id={pid}, level={level}) ===\n"
            f"--- enable_thinking=True, add_generation_prompt=True ---\n\n"
            f"{prompt_str}\n\n=== END ===\n", encoding="utf-8")

    enc = tokenizer(prompt_str, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")
    prompt_ids = input_ids[0].tolist()
    prompt_len = input_ids.shape[1]

    # ── per-sample seed ──
    seed = per_sample_seed(pid)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0

    full_ids_list = out_ids[0].tolist()
    total_len = len(full_ids_list)
    gen_len = total_len - prompt_len
    last_tok = full_ids_list[-1]

    # finish_reason
    if gen_len >= max_new_tokens:
        finish_reason = "length"
    elif last_tok == tokenizer.eos_token_id:
        finish_reason = "stop"
    else:
        finish_reason = "unknown"
    truncated = (finish_reason == "length")

    # ── <think>/</think> markers ──
    open_ids, open_single, open_id = encode_marker(tokenizer, "<think>")
    close_ids, close_single, close_id = encode_marker(tokenizer, "</think>")

    # Whether template injected <think> into the prompt
    template_open_in_prompt = (open_id in prompt_ids)

    open_idx = find_first(full_ids_list, open_id)
    close_idx = find_first(full_ids_list, close_id)
    open_count = count_occurrences(full_ids_list, open_id)
    close_count = count_occurrences(full_ids_list, close_id)

    # Branch: if template put <think> at prompt end, t1_think = first generated tok
    think_status = "ok"
    if open_idx == -1 and template_open_in_prompt:
        # shouldn't happen (find_first searches full_ids incl prompt), but guard
        t1_think = prompt_len
    elif open_idx == -1:
        t1_think = -1
        think_status = "FAIL_no_open"
    else:
        t1_think = open_idx + 1

    if close_idx == -1:
        tK_think = -1
        think_status = "FAIL_no_close" if think_status == "ok" else think_status
    else:
        tK_think = close_idx - 1

    # validity
    think_valid = (0 <= t1_think < tK_think < total_len)
    if not think_valid and think_status == "ok":
        think_status = "FAIL_order"

    # positions to capture
    pos_t1_faithful = 0
    pos_tK_faithful = total_len - 1
    pos_prompt_last = prompt_len - 1
    # for think positions, if invalid, fall back to 0 (won't be used in analysis)
    cap_t1_think = t1_think if t1_think >= 0 else 0
    cap_tK_think = tK_think if tK_think >= 0 else 0

    positions = {
        "A_pos0":        pos_t1_faithful,
        "A_prompt_last": pos_prompt_last,
        "A_t1_think":    cap_t1_think,
        "A_tK_think":    cap_tK_think,
        "A_last":        pos_tK_faithful,
    }

    t1 = time.time()
    acts = capture_positions(model, out_ids, positions, num_layers)
    fwd_time = time.time() - t1

    # ── ΔA ──
    dA_faithful = acts["A_last"] - acts["A_pos0"]            # [L, inter]
    if think_valid:
        dA_thinking = acts["A_tK_think"] - acts["A_t1_think"]
    else:
        dA_thinking = torch.zeros_like(dA_faithful)

    def stats(t: torch.Tensor) -> dict:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "has_nan": bool(torch.isnan(t).any().item()),
            "has_inf": bool(torch.isinf(t).any().item()),
            "l2_per_layer_mean": float(t.norm(dim=1).mean().item()),
        }

    dA_faithful_stats = stats(dA_faithful)
    dA_thinking_stats = stats(dA_thinking)

    # per-layer L2 norms for report
    dA_faithful_norms = dA_faithful.norm(dim=1).tolist()
    dA_thinking_norms = dA_thinking.norm(dim=1).tolist()

    # decode generated text (truncate for storage)
    gen_text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=False)

    # save .pt
    save = {
        "problem_id": pid,
        "level": level,
        "subject": sample.get("subject", ""),
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "total_len": total_len,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "seed": seed,
        "open_id": open_id, "close_id": close_id,
        "open_idx": open_idx, "close_idx": close_idx,
        "t1_think": t1_think, "tK_think": tK_think,
        "think_span_len": (tK_think - t1_think + 1) if think_valid else -1,
        "think_valid": think_valid,
        "template_open_in_prompt": template_open_in_prompt,
        "dA_faithful": dA_faithful,      # [L, inter] float32
        "dA_thinking": dA_thinking,      # [L, inter] float32
        "A_pos0": acts["A_pos0"],
        "A_prompt_last": acts["A_prompt_last"],
        "A_t1_think": acts["A_t1_think"],
        "A_tK_think": acts["A_tK_think"],
        "A_last": acts["A_last"],
        "generated_text": gen_text,
        "gen_time_s": gen_time,
        "fwd_time_s": fwd_time,
    }
    shifts_dir = output_dir / "smoke_shifts"
    shifts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(save, shifts_dir / f"{pid}.pt")

    meta = {
        "problem_id": pid,
        "level": level,
        "subject": sample.get("subject", ""),
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "total_len": total_len,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "seed": seed,
        "open_id": open_id, "open_single": open_single, "open_ids": open_ids,
        "close_id": close_id, "close_single": close_single, "close_ids": close_ids,
        "open_idx": open_idx, "close_idx": close_idx,
        "open_count": open_count, "close_count": close_count,
        "t1_think": t1_think, "tK_think": tK_think,
        "think_span_len": (tK_think - t1_think + 1) if think_valid else -1,
        "think_valid": think_valid,
        "think_status": think_status,
        "template_open_in_prompt": template_open_in_prompt,
        "dA_faithful_stats": dA_faithful_stats,
        "dA_thinking_stats": dA_thinking_stats,
        "dA_faithful_norms": dA_faithful_norms,
        "dA_thinking_norms": dA_thinking_norms,
        "gen_time_s": round(gen_time, 2),
        "fwd_time_s": round(fwd_time, 2),
        "rank": None,  # filled by main
        "status": "ok",
    }
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.samples_parquet)
    shard = df.iloc[args.rank::args.world_size].reset_index(drop=True)
    print(f"[rank {args.rank}] shard size = {len(shard)} / {len(df)}", flush=True)

    model, tok, num_layers, inter = load_model(args.model_id)

    if torch.cuda.is_available():
        print(f"[rank {args.rank}] GPU mem after load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    meta_path = out_dir / f"smoke_meta_rank{args.rank}.jsonl"
    # fresh start each run
    if meta_path.exists():
        meta_path.unlink()

    template_dump = out_dir / "chat_template_preview.txt" if args.rank == 0 else None

    for i in range(len(shard)):
        sample = shard.iloc[i].to_dict()
        pid = sample["problem_id"]
        print(f"\n[rank {args.rank}] [{i+1}/{len(shard)}] id={pid} "
              f"level={sample['level']}", flush=True)
        try:
            dump = template_dump if (args.rank == 0 and i == 0) else None
            meta = process_sample(sample, model, tok, num_layers, inter,
                                  args.max_new_tokens, out_dir, dump)
            meta["rank"] = args.rank
            print(f"[rank {args.rank}]   gen_len={meta['gen_len']} "
                  f"finish={meta['finish_reason']} think_valid={meta['think_valid']} "
                  f"span={meta['think_span_len']} "
                  f"dA_f_nan={meta['dA_faithful_stats']['has_nan']} "
                  f"gen={meta['gen_time_s']}s fwd={meta['fwd_time_s']}s", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            meta = {"problem_id": pid, "level": int(sample["level"]),
                    "rank": args.rank, "status": "OOM", "error": str(e)[:200]}
            print(f"[rank {args.rank}]   OOM: {str(e)[:120]}", flush=True)
        except Exception as e:
            meta = {"problem_id": pid, "level": int(sample["level"]),
                    "rank": args.rank, "status": "error", "error": str(e)[:300]}
            print(f"[rank {args.rank}]   ERROR: {str(e)[:200]}", flush=True)

        with open(meta_path, "a") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"\n[rank {args.rank}] DONE → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
