#!/usr/bin/env python3
"""
extract_thinking_pilot.py
=========================
Phase Z-2 PILOT — Qwen3-8B thinking-mode activation extraction (production).

Extends extract_thinking_smoke.py with:
  (1) truncation t_k fix : if </think> never appears but generation is truncated,
      treat the LAST generated token as t_k. think_valid=True,
      think_status="ok_truncated". (faithful ΔA is unaffected — it always uses
      pos0 .. last token.)
  (2) is_correct scoring  : math_verify(compute_score) of generated answer vs
      ground-truth `answer`. Stores is_correct / gt_answer / extracted_answer.
  (3) resume / skip-existing : if shifts/{problem_id}.pt already exists, skip.
      → safe across time-limit-truncated chunk re-submission.
  (4) rich, joinable meta : sample_uid = "{spec_name}__{problem_id}", plus all
      labels (subject, subject_raw, level, r1_cot_token_count) and generation
      spec (spec_name, gen params, model, dtype). One pilot_meta_rank*.jsonl per
      shard → merged downstream.
  (5) chunked universe : --chunk-id selects rows where chunk_id == that value
      (round-robin balanced across cells), then --rank shards within the chunk.

Per-position raw activations + two ΔA definitions are identical to the smoke
script (NAIT-faithful and NAIT-inspired thinking-span).

Reference (read-only): extract_thinking_smoke.py (same dir).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.95
GEN_TOP_K = 20
DEFAULT_SPEC = "thinking_8k_v1"

BASE = Path("/scratch/lami2026/personal/jimin_2782")
VERL_REWARD = BASE / "src/verl-new"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-8B thinking-mode pilot extraction (1 shard)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--samples-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-id", type=int, default=-1,
                   help="select rows where chunk_id == this; -1 = all rows")
    p.add_argument("--limit", type=int, default=0,
                   help="cap shard size (smoke); 0 = no cap")
    p.add_argument("--spec-name", default=DEFAULT_SPEC)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    return p.parse_args()


# ── reward function loader (mirrors pass_rate_measurement.py) ────────────────
def load_reward_fn():
    try:
        from verl.utils.reward_score.math_verify import compute_score as _cs
        print("[INFO] reward: verl.utils.reward_score.math_verify", flush=True)
        return _cs
    except Exception:
        pass
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "math_verify_file",
            str(VERL_REWARD / "verl/utils/reward_score/math_verify.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("[INFO] reward: verl math_verify (file import)", flush=True)
        return mod.compute_score
    except Exception as e:
        print(f"[WARN] verl reward import failed ({str(e)[:120]}); trying math_verify pkg", flush=True)
        from math_verify.grader import verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

        def compute_score(model_output: str, ground_truth: str, **kw) -> float:
            try:
                gt_boxed = "\\boxed{" + str(ground_truth) + "}"
                eg = parse(gt_boxed, (LatexExtractionConfig(),))
                ep = parse(model_output, (ExprExtractionConfig(), LatexExtractionConfig()))
                if eg and ep:
                    return 1.0 if verify(eg, ep) else 0.0
                return 0.0
            except Exception:
                return 0.0
        return compute_score


def score_to_correct(score) -> bool:
    if isinstance(score, dict):
        score = score.get("score", score.get("acc", 0.0))
    try:
        return bool(float(score) >= 1.0)
    except Exception:
        return bool(score)


_BOX_RE = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str:
    """Return content of the LAST \\boxed{...} (brace-balanced), else ''."""
    last = ""
    for m in _BOX_RE.finditer(text):
        i = m.end()
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        last = "".join(buf)
    return last.strip()


# ── helpers (identical to smoke) ─────────────────────────────────────────────
def per_sample_seed(problem_id: str) -> int:
    return int(problem_id[:8], 16) % (2 ** 31)


def find_first(seq: list[int], token_id: int) -> int:
    try:
        return seq.index(token_id)
    except ValueError:
        return -1


def count_occurrences(seq: list[int], token_id: int) -> int:
    return sum(1 for t in seq if t == token_id)


def encode_marker(tokenizer, text: str):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids, (len(ids) == 1), (ids[0] if ids else -1)


def load_model(model_id: str, max_retries: int = 4):
    print(f"[MODEL] loading {model_id} (bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = None
    last_err = None
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                device_map={"": 0}, trust_remote_code=True)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 5 * (2 ** attempt)
            print(f"[MODEL] load attempt {attempt+1}/{max_retries} failed: "
                  f"{str(e)[:160]} → retry in {wait}s", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(wait)
    if model is None:
        raise RuntimeError(f"model load failed after {max_retries}: {last_err}")
    model.eval()
    num_layers = len(model.model.layers)
    inter = model.model.layers[0].mlp.down_proj.in_features
    print(f"[MODEL] num_layers={num_layers} intermediate_size={inter} "
          f"hidden={model.config.hidden_size}", flush=True)
    return model, tok, num_layers, inter


def capture_positions(model, full_ids, positions, num_layers):
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs):
            captured[layer_idx] = inputs[0][0].detach()
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

    out = {}
    seq_len = full_ids.shape[1]
    for name, idx in positions.items():
        idx = max(0, min(idx, seq_len - 1))
        rows = [captured[l][idx].to(torch.float32).cpu() for l in range(num_layers)]
        out[name] = torch.stack(rows, dim=0)
    del captured
    torch.cuda.empty_cache()
    return out


def process_sample(sample, model, tokenizer, num_layers, inter, max_new_tokens,
                   shifts_dir, reward_fn, spec_name, dump_template_to):
    pid = sample["problem_id"]
    level = int(sample["level"])
    problem = sample["problem_text"]
    gt_answer = str(sample.get("answer", "") or "")

    messages = [{"role": "user", "content": problem}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if dump_template_to is not None:
        dump_template_to.write_text(
            f"=== chat_template_preview (problem_id={pid}, level={level}) ===\n"
            f"{prompt_str}\n=== END ===\n", encoding="utf-8")

    enc = tokenizer(prompt_str, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")
    prompt_ids = input_ids[0].tolist()
    prompt_len = input_ids.shape[1]

    seed = per_sample_seed(pid)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
            top_k=GEN_TOP_K, pad_token_id=tokenizer.pad_token_id)
    gen_time = time.time() - t0

    full_ids_list = out_ids[0].tolist()
    total_len = len(full_ids_list)
    gen_len = total_len - prompt_len
    last_tok = full_ids_list[-1]

    if gen_len >= max_new_tokens:
        finish_reason = "length"
    elif last_tok == tokenizer.eos_token_id:
        finish_reason = "stop"
    else:
        finish_reason = "unknown"
    truncated = (finish_reason == "length")

    open_ids, open_single, open_id = encode_marker(tokenizer, "<think>")
    close_ids, close_single, close_id = encode_marker(tokenizer, "</think>")
    template_open_in_prompt = (open_id in prompt_ids)

    open_idx = find_first(full_ids_list, open_id)
    close_idx = find_first(full_ids_list, close_id)
    open_count = count_occurrences(full_ids_list, open_id)
    close_count = count_occurrences(full_ids_list, close_id)

    # ── t1 (start of thinking span) ──
    think_status = "ok"
    if open_idx == -1 and template_open_in_prompt:
        t1_think = prompt_len
    elif open_idx == -1:
        t1_think = -1
        think_status = "FAIL_no_open"
    else:
        t1_think = open_idx + 1

    # ── tK (end of thinking span) — TRUNCATION FIX ──
    if close_idx != -1:
        tK_think = close_idx - 1
    elif truncated and t1_think >= 0:
        # thinking never closed because generation hit the token budget:
        # use the last generated token as t_k.
        tK_think = total_len - 1
        if think_status == "ok":
            think_status = "ok_truncated"
    else:
        tK_think = -1
        if think_status == "ok":
            think_status = "FAIL_no_close"

    think_valid = (0 <= t1_think < tK_think < total_len)
    if not think_valid and think_status in ("ok", "ok_truncated"):
        think_status = "FAIL_order"

    pos_prompt_last = prompt_len - 1
    cap_t1_think = t1_think if t1_think >= 0 else 0
    cap_tK_think = tK_think if tK_think >= 0 else 0
    positions = {
        "A_pos0":        0,
        "A_prompt_last": pos_prompt_last,
        "A_t1_think":    cap_t1_think,
        "A_tK_think":    cap_tK_think,
        "A_last":        total_len - 1,
    }

    t1 = time.time()
    acts = capture_positions(model, out_ids, positions, num_layers)
    fwd_time = time.time() - t1

    dA_faithful = acts["A_last"] - acts["A_pos0"]
    dA_thinking = (acts["A_tK_think"] - acts["A_t1_think"]) if think_valid \
        else torch.zeros_like(dA_faithful)

    def stats(t):
        return {
            "shape": list(t.shape), "dtype": str(t.dtype),
            "has_nan": bool(torch.isnan(t).any().item()),
            "has_inf": bool(torch.isinf(t).any().item()),
            "l2_per_layer_mean": float(t.norm(dim=1).mean().item()),
        }

    dA_faithful_stats = stats(dA_faithful)
    dA_thinking_stats = stats(dA_thinking)

    gen_text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=False)

    # ── is_correct scoring ──
    extracted_answer = extract_boxed(gen_text)
    is_correct = None
    try:
        score = reward_fn(gen_text, gt_answer) if gt_answer else None
        if score is not None:
            is_correct = score_to_correct(score)
    except Exception as e:
        print(f"[WARN] scoring failed id={pid}: {str(e)[:120]}", flush=True)
        is_correct = None

    sample_uid = f"{spec_name}__{pid}"

    save = {
        "sample_uid": sample_uid, "spec_name": spec_name,
        "problem_id": pid, "level": level,
        "subject": sample.get("subject", ""),
        "subject_raw": sample.get("subject_raw", ""),
        "r1_cot_token_count": sample.get("r1_cot_token_count", -1),
        "prompt_len": prompt_len, "gen_len": gen_len, "total_len": total_len,
        "finish_reason": finish_reason, "truncated": truncated, "seed": seed,
        "open_id": open_id, "close_id": close_id,
        "open_idx": open_idx, "close_idx": close_idx,
        "t1_think": t1_think, "tK_think": tK_think,
        "think_span_len": (tK_think - t1_think + 1) if think_valid else -1,
        "think_valid": think_valid, "think_status": think_status,
        "template_open_in_prompt": template_open_in_prompt,
        "is_correct": is_correct, "gt_answer": gt_answer,
        "extracted_answer": extracted_answer,
        "max_new_tokens": max_new_tokens,
        "gen_temperature": GEN_TEMPERATURE, "gen_top_p": GEN_TOP_P, "gen_top_k": GEN_TOP_K,
        "dA_faithful": dA_faithful, "dA_thinking": dA_thinking,
        "A_pos0": acts["A_pos0"], "A_prompt_last": acts["A_prompt_last"],
        "A_t1_think": acts["A_t1_think"], "A_tK_think": acts["A_tK_think"],
        "A_last": acts["A_last"],
        "generated_text": gen_text,
        "gen_time_s": gen_time, "fwd_time_s": fwd_time,
    }
    torch.save(save, shifts_dir / f"{pid}.pt")

    meta = {
        "sample_uid": sample_uid, "spec_name": spec_name,
        "problem_id": pid, "level": level,
        "subject": sample.get("subject", ""),
        "subject_raw": sample.get("subject_raw", ""),
        "r1_cot_token_count": int(sample.get("r1_cot_token_count", -1) or -1),
        "prompt_len": prompt_len, "gen_len": gen_len, "total_len": total_len,
        "finish_reason": finish_reason, "truncated": truncated, "seed": seed,
        "open_id": open_id, "close_id": close_id,
        "open_idx": open_idx, "close_idx": close_idx,
        "open_count": open_count, "close_count": close_count,
        "t1_think": t1_think, "tK_think": tK_think,
        "think_span_len": (tK_think - t1_think + 1) if think_valid else -1,
        "think_valid": think_valid, "think_status": think_status,
        "template_open_in_prompt": template_open_in_prompt,
        "is_correct": is_correct, "gt_answer": gt_answer,
        "extracted_answer": extracted_answer,
        "max_new_tokens": max_new_tokens,
        "dA_faithful_stats": dA_faithful_stats,
        "dA_thinking_stats": dA_thinking_stats,
        "dA_faithful_l2_mean": dA_faithful_stats["l2_per_layer_mean"],
        "dA_thinking_l2_mean": dA_thinking_stats["l2_per_layer_mean"],
        "gen_time_s": round(gen_time, 2), "fwd_time_s": round(fwd_time, 2),
        "rank": None, "status": "ok",
    }
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    shifts_dir = out_dir / "shifts"
    shifts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.samples_parquet)
    if args.chunk_id >= 0 and "chunk_id" in df.columns:
        df = df[df["chunk_id"] == args.chunk_id].reset_index(drop=True)
    shard = df.iloc[args.rank::args.world_size].reset_index(drop=True)
    if args.limit > 0:
        shard = shard.iloc[:args.limit].reset_index(drop=True)
    print(f"[rank {args.rank}] chunk={args.chunk_id} shard={len(shard)} / "
          f"chunk_rows={len(df)} spec={args.spec_name}", flush=True)

    model, tok, num_layers, inter = load_model(args.model_id)
    reward_fn = load_reward_fn()

    if torch.cuda.is_available():
        print(f"[rank {args.rank}] GPU mem after load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    meta_path = out_dir / f"pilot_meta_rank{args.rank}.jsonl"
    template_dump = out_dir / "chat_template_preview.txt" if args.rank == 0 else None

    n_done = n_skip = 0
    for i in range(len(shard)):
        sample = shard.iloc[i].to_dict()
        pid = sample["problem_id"]
        pt_path = shifts_dir / f"{pid}.pt"
        if pt_path.exists():
            n_skip += 1
            if n_skip <= 5 or n_skip % 50 == 0:
                print(f"[rank {args.rank}] [{i+1}/{len(shard)}] skip existing {pid} "
                      f"(total skip={n_skip})", flush=True)
            continue
        print(f"[rank {args.rank}] [{i+1}/{len(shard)}] id={pid} level={sample['level']}",
              flush=True)
        try:
            dump = template_dump if (args.rank == 0 and n_done == 0) else None
            meta = process_sample(sample, model, tok, num_layers, inter,
                                  args.max_new_tokens, shifts_dir, reward_fn,
                                  args.spec_name, dump)
            meta["rank"] = args.rank
            n_done += 1
            print(f"[rank {args.rank}]   gen_len={meta['gen_len']} "
                  f"finish={meta['finish_reason']} think={meta['think_status']} "
                  f"span={meta['think_span_len']} correct={meta['is_correct']} "
                  f"dA_f_l2={meta['dA_faithful_l2_mean']:.3f} "
                  f"gen={meta['gen_time_s']}s", flush=True)
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

    print(f"\n[rank {args.rank}] DONE done={n_done} skip={n_skip} → {meta_path}",
          flush=True)


if __name__ == "__main__":
    main()
