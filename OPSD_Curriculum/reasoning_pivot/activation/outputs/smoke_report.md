# Phase Z-1 Smoke Report — Qwen3-8B Thinking-mode Activation Extraction

- Script: `src/OPSD_Curriculum/reasoning_pivot/activation/extract_thinking_smoke.py`
- Samples: `/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/reasoning_pivot/activation/outputs/smoke_samples.parquet`
- Shifts (.pt): `/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/reasoning_pivot/activation/outputs/smoke_shifts/{problem_id}.pt`
- Merged meta: `/scratch/lami2026/personal/jimin_2782/src/OPSD_Curriculum/reasoning_pivot/activation/outputs/smoke_meta.jsonl`
- max_new_tokens: **8192**, sampling: do_sample=True temp=0.6 top_p=0.95 top_k=20
- GPU: L40s ×2 (iREMB-C-07), per-sample seed = int(problem_id[:8],16) % 2**31

## Pass/Fail Gates

| Gate | Result |
|---|---|
| count: 7 unique problem_id, no dup/missing | ✅ PASS |
| hooks fire (dA_faithful L2 > 0 all) | ✅ PASS |
| ΔA shape [36, inter] all | ✅ PASS |
| no NaN/Inf (both ΔA) | ✅ PASS |
| think indexing valid (0<t1<tK<total) for non-truncated | ✅ PASS |
| template `<think>` handling clear + consistent | ✅ PASS |

**OVERALL: ✅ PASS**


## Chat Template `<think>` Handling

- `apply_chat_template(enable_thinking=True)` injects `<think>` into prompt: **NO** (consistent across samples: True)
- `<think>` token id = 151667 (single_token=True, ids=[151667])
- `</think>` token id = 151668 (single_token=True, ids=[151668])

<details><summary>chat_template_preview.txt (tail)</summary>

```
=== chat_template_preview (problem_id=2b21e67b278c0ac6, level=1) ===
--- enable_thinking=True, add_generation_prompt=True ---

<|im_start|>user
The radius  $r$  of a circle is increasing at a rate of  $2$  meters per minute. Find the rate of change, in  $\text{meters}^2/\text{minute}$ , of the area when  $r$  is  $6$  meters.<|im_end|>
<|im_start|>assistant


=== END ===

```
</details>

## Sample Table

| problem_id | level | subject | prompt_len | gen_len | total_len | open_idx | close_idx | think_span | finish | truncated | think_valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2b21e67b278c0ac6 | 1 | Other | 72 | 1098 | 1170 | 72 | 853 | 780 | stop | False | True |
| e55e3ed17afa1c94 | 2 | Intermediate Algebra | 32 | 3194 | 3226 | 32 | 2906 | 2873 | stop | False | True |
| 693a81acabd41e55 | 4 | Geometry | 71 | 8192 | 8263 | 71 | -1 | -1 | length | True | False |
| bc5fc33b3c96972a | 5 | Number Theory | 67 | 4748 | 4815 | 67 | 3928 | 3860 | stop | False | True |
| 61a6ab5c57145429 | 6 | Geometry | 83 | 8192 | 8275 | 83 | -1 | -1 | length | True | False |
| 3f0063d71420672d | 7 | Counting & Probability | 189 | 8192 | 8381 | 189 | -1 | -1 | length | True | False |
| d97bfcd7fd61ec99 | 8 | Geometry | 102 | 8192 | 8294 | 102 | -1 | -1 | length | True | False |

## Length Summary

- gen_len: median=8192, p95=8192, max=8192
- think_span_len: median=2873, p95=3761, max=3860
- truncation: **4/7** hit 8192 cap (hard L6-L8: 3/3)

## ΔA Norm Comparison (faithful vs thinking)

Per-sample mean per-layer L2 norm:

| problem_id | level | ‖dA_faithful‖ (mean/layer) | ‖dA_thinking‖ (mean/layer) | think_valid |
|---|---|---|---|---|
| 2b21e67b278c0ac6 | 1 | 406.758 | 70.785 | True |
| e55e3ed17afa1c94 | 2 | 406.024 | 71.682 | True |
| 693a81acabd41e55 | 4 | 408.191 | 0.000 | False |
| bc5fc33b3c96972a | 5 | 405.003 | 71.326 | True |
| 61a6ab5c57145429 | 6 | 400.390 | 0.000 | False |
| 3f0063d71420672d | 7 | 410.932 | 0.000 | False |
| d97bfcd7fd61ec99 | 8 | 428.953 | 0.000 | False |

- mean ‖dA_faithful‖ across samples = 409.464
- mean ‖dA_thinking‖ across samples = 71.264

- dtype: float32, shape: [36, intermediate_size] (per sample, both ΔA)

## Pilot Universe Context

- `correct` value_counts: {True: 3000} → **JSON-parsing-success flag only, NOT a correctness signal**; not usable as difficulty-bias indicator.
- `r1_cot_token_count` (DeepSeek-R1 CoT len): median=2878, p95=4761, max=4996

## max_new_tokens = 8192 Adequacy Opinion

- 3/3 hard samples hit the 8192 cap (truncated). → even 8192 is insufficient for hard MATH thinking. **Adopt a 'truncated-excluded' rule for think-span ΔA** (faithful ΔA still defined), and/or raise cap further only if hard-level think-span ΔA is required.

> Note: Qwen3 thinking generations can exceed DeepSeek-R1 CoT lengths; r1_cot is only a rough prior.
