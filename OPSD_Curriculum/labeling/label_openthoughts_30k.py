"""
label_openthoughts_30k.py
-------------------------
Openthoughts_math_30k_opsd 전체 (29,434 문제) 를 GPT-4.1-mini 로
(subject, level) 라벨링. async + resume.

SYSTEM_PROMPT 는 src/4.6_Task2/classifier/classify_full.py 의 것과
완전히 동일 (verbatim copy) — 기존 FastCuRL 분류와 분류 기준 일관성 확보.

ENV:
  LAMI_OPENAI_API_KEY    (required, 사용자가 export)
  HF_HOME / HF_DATASETS_CACHE  (optional; 기본 cache dir 사용)

실행 예:
  # smoke (200 sample)
  python label_openthoughts_30k.py --limit 200 --concurrency 20 \
         --output outputs/smoke200_labels.csv
  # full
  python label_openthoughts_30k.py --concurrency 20 \
         --output outputs/openthoughts_30k_labels.csv
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
HF_DATASET_ID = "siyanzhao/Openthoughts_math_30k_opsd"
HF_CACHE_DIR  = "/scratch/lami2026/personal/jimin_2782/.hf_cache_reasoning_pivot"
QWEN_TOKENIZER_ID = "Qwen/Qwen3-8B"

MODEL          = "gpt-4.1-mini-2025-04-14"
TEMPERATURE    = 0
MAX_TOKENS     = 50
MAX_RETRIES    = 4
SAVE_INTERVAL  = 200

# SYSTEM_PROMPT — classify_full.py 와 완전히 동일. 절대 수정 금지.
SYSTEM_PROMPT = """You are a mathematics problem classifier. For each problem, output two labels:

1. SUBJECT — one of 8 categories
2. LEVEL — integer 1 to 8 (absolute difficulty)

== SUBJECT CATEGORIES ==

- Algebra: equations, inequalities, polynomials, sequences, functions (basic), word problems
- Counting & Probability: combinatorics, permutations, probability, expected value
- Geometry: Euclidean geometry, triangles, circles, polygons, coordinate geometry, areas
- Intermediate Algebra: complex numbers, advanced functions, logarithms, polynomial roots
- Number Theory: divisibility, modular arithmetic, primes, GCD/LCM, Diophantine equations
- Prealgebra: arithmetic, fractions, percentages, ratios, basic operations, units
- Precalculus: trigonometry, vectors, matrices, parametric equations, polar coordinates, conic sections (parabola, ellipse, hyperbola)
- Other: only if the problem genuinely fits none of the above (e.g., pure logic puzzles, abstract algebra beyond standard scope, calculus-heavy problems)

Rules:
- Choose the SINGLE most representative subject. If a problem mixes two areas, pick the one that dominates the solution method.
- Use "Other" sparingly. Most problems fit one of the seven main categories.
- A "word problem" about digits/numbers/divisibility belongs to Number Theory, not Other.

== LEVEL DEFINITION (absolute difficulty, NOT source-based) ==

Evaluate the OBJECTIVE difficulty of solving the problem. The problem's source (AIME, Olympiad, etc.) is irrelevant — judge each problem on its own.

- L1: Arithmetic / elementary. Single-step computation.
- L2: Middle school basics. Simple equations, basic geometry.
- L3: High school foundations. Standard techniques, 1-2 steps.
- L4: High school advanced. Combining multiple concepts, non-trivial manipulation. (≈ MATH dataset Level 4) Routine application of Law of Sines/Cosines, vector operations, coordinate geometry formulas, standard trig identities belongs HERE, not L6.
- L5: College intro / competition entry. Insight or non-standard technique required. (≈ MATH Level 5, harder AMC, easy AIME #1-5)
- L6: Mid-tier competition. Requires a non-trivial KEY INSIGHT, not just multi-step computation. Standard techniques alone do NOT qualify. (≈ AIME #6-12, Olympiad P1)
- L7: Upper competition. Multiple deep steps, each requiring substantial insight. (≈ AIME #13-15, Olympiad P2-P4) Hard problems from competition datasets typically belong here.
- L8: Top-tier competition. Creative construction or novel approach beyond standard techniques. (≈ Hard IMO/Putnam, Olympiad P5-P6)

CRITICAL RULES:
- Source does not determine level. An easy AIME problem can be L4. A hard Omni-MATH problem can be L7.
- A LONG problem statement does NOT mean a high level. Many problems with lengthy setups are solvable by routine techniques and should be L3-L5. Judge by the DEPTH OF INSIGHT required, not by text length.
- L7-L8 should NOT be rare in competition data. Do not default to L6 just because the problem "looks competition-style." If the solution requires deep insight (not just standard manipulation), use L7 or L8.
- If a problem can be solved by ROUTINE application of a standard formula (Law of Sines, distance formula, basic vectors, standard substitution), it is at most L4-L5, regardless of how it is phrased.
- If between two levels, prefer the lower one ONLY when the problem genuinely sits at the boundary. Do NOT use this rule to systematically avoid L7-L8.

== OUTPUT FORMAT ==

Respond with ONLY a JSON object, no other text:
{"subject": "<one of the 8 categories>", "level": <integer 1-8>}

== EXAMPLES ==

Problem: "Compute 23 + 45."
{"subject": "Prealgebra", "level": 1}

Problem: "Solve for x: 3x + 7 = 22."
{"subject": "Algebra", "level": 2}

Problem: "Find the area of a triangle with vertices (0,0), (4,0), (0,3)."
{"subject": "Geometry", "level": 3}

Problem: "Given the parabola y^2 = 2x, find the equation of its directrix."
{"subject": "Precalculus", "level": 3}

Problem: "How many ways are there to arrange the letters of MISSISSIPPI?"
{"subject": "Counting & Probability", "level": 3}

Problem: "In triangle ABC, a = 2b and sin A, sin C, sin B form an arithmetic sequence. If the area is 8*sqrt(15)/3, find c."
{"subject": "Geometry", "level": 4}

Problem: "Find all real solutions to x^4 - 4x^3 + 6x^2 - 4x + 1 = 0."
{"subject": "Intermediate Algebra", "level": 4}

Problem: "Find the smallest positive integer n such that n^2 + n + 41 is divisible by 7."
{"subject": "Number Theory", "level": 4}

Problem: "Let f(x) = sin(x) + cos(2x). Find the maximum value of f on [0, 2*pi] and prove the bound is tight."
{"subject": "Precalculus", "level": 5}

Problem: "Petya's apartment number is a three-digit number. Rearranging its digits gives five other three-digit numbers, summing to 2017. Find Petya's apartment number."
{"subject": "Number Theory", "level": 6}

Problem: "Given a triangle ABC with sides a, b, c and area S, prove that a^2 + b^2 + c^2 >= 4*sqrt(3)*S."
{"subject": "Geometry", "level": 6}

Problem: "Determine all positive integer solutions (x, y, z) to x! + y! = z^2."
{"subject": "Number Theory", "level": 7}

Problem: "Let n >= 2. Prove that the polynomial x^n + 5x^{n-1} + 3 is irreducible over the integers."
{"subject": "Intermediate Algebra", "level": 7}

Problem: "In acute triangle ABC, let O be the circumcenter and H the orthocenter. Prove that the reflection of H over the midpoint of BC lies on the circumcircle."
{"subject": "Geometry", "level": 7}

Problem: "Let f: R -> R satisfy f(x+y) + f(x-y) = 2f(x)f(y) for all real x, y, with f not identically zero. Determine all such f."
{"subject": "Other", "level": 7}

Problem: "Compute the integer part of (sqrt(2)+sqrt(3))^{1000}."
{"subject": "Intermediate Algebra", "level": 8}

Problem: "Find the smallest positive integer that cannot be expressed in the form (2^a - 2^b)/(2^c - 2^d) where a,b,c,d are positive integers."
{"subject": "Number Theory", "level": 8}
"""

PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]

OUTPUT_COLUMNS = [
    "row_index",
    "source",
    "problem_text",
    "problem_char_len",
    "problem_qwen_tok_len",
    "r1_cot_token_count",
    "solution_char_len",
    "correct",
    "answer",
    "subject",
    "level",
    "raw_response",
    "error",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "latency_s",
    "attempts",
    "model",
    "prompt_sha",
]

# ─────────────────────────────────────────────────────────────────────
# Data loading + static feature extraction
# ─────────────────────────────────────────────────────────────────────
def load_dataset_rows(limit: int | None) -> pd.DataFrame:
    """HF dataset → pandas DF with row_index + static features (no GPT)."""
    from datasets import load_dataset

    print(f"[load] {HF_DATASET_ID}")
    ds = load_dataset(HF_DATASET_ID, cache_dir=HF_CACHE_DIR, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"       loaded {len(ds):,} rows")

    print(f"[tok ] {QWEN_TOKENIZER_ID}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_ID, trust_remote_code=True)

    rows = []
    for i, ex in enumerate(ds):
        problem = ex.get("problem") or ""
        solution = ex.get("solution") or ""
        rows.append({
            "row_index": i,
            "source": ex.get("source") or "",
            "problem_text": problem,
            "problem_char_len": len(problem),
            "problem_qwen_tok_len": len(tok.encode(problem, add_special_tokens=False)),
            "r1_cot_token_count": int(ex.get("generated_token_count") or 0),
            "solution_char_len": len(solution),
            "correct": bool(ex.get("correct")) if ex.get("correct") is not None else None,
            "answer": ex.get("Answer") or "",
        })
    df = pd.DataFrame(rows)
    print(f"[stat] problem_qwen_tok_len: median={df.problem_qwen_tok_len.median():.0f} "
          f"p95={df.problem_qwen_tok_len.quantile(0.95):.0f}")
    return df


# ─────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────
def save_csv(results: list[dict], path: Path) -> None:
    df = pd.DataFrame(results)
    # ensure all columns present
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[OUTPUT_COLUMNS].sort_values("row_index")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_resume(path: Path) -> dict[int, dict]:
    """Return {row_index: result_dict} for already-successful rows."""
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    ok = df[(df["error"].fillna("") == "") & df["subject"].notna() & df["level"].notna()]
    print(f"[resume] {path.name}: {len(ok):,} previously-successful rows will be skipped")
    out: dict[int, dict] = {}
    for r in ok.to_dict(orient="records"):
        out[int(r["row_index"])] = r
    return out


# ─────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────
async def classify_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    base_row: dict,
) -> dict:
    """Single classification. base_row already has static features."""
    out = dict(base_row)
    out.update({
        "subject": None,
        "level": None,
        "raw_response": "",
        "error": "",
        "finish_reason": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_s": None,
        "attempts": 0,
        "model": MODEL,
        "prompt_sha": PROMPT_SHA,
    })

    async with sem:
        t0 = time.monotonic()
        for attempt in range(1, MAX_RETRIES + 1):
            out["attempts"] = attempt
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Problem: {base_row['problem_text']}"},
                    ],
                )
                choice = resp.choices[0]
                raw = choice.message.content or ""
                out["raw_response"]      = raw
                out["finish_reason"]     = getattr(choice, "finish_reason", "") or ""
                u = getattr(resp, "usage", None)
                if u is not None:
                    out["prompt_tokens"]     = getattr(u, "prompt_tokens", None)
                    out["completion_tokens"] = getattr(u, "completion_tokens", None)

                try:
                    parsed = json.loads(raw)
                    out["subject"] = parsed.get("subject")
                    out["level"]   = parsed.get("level")
                    if out["subject"] is None or out["level"] is None:
                        out["error"] = "missing_fields"
                except json.JSONDecodeError as je:
                    out["error"] = f"json_parse_error: {je}"
                break  # success or non-retryable parse error

            except Exception as e:
                if attempt == MAX_RETRIES:
                    out["error"] = f"{type(e).__name__}: {e}"
                    break
                await asyncio.sleep(min(30.0, 2 ** attempt))  # exp backoff capped

        out["latency_s"] = round(time.monotonic() - t0, 3)
    return out


async def run(df: pd.DataFrame, output_path: Path, concurrency: int) -> None:
    api_key = os.environ.get("LAMI_OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "환경변수 LAMI_OPENAI_API_KEY 가 설정되지 않았습니다. "
            "터미널에서 `export LAMI_OPENAI_API_KEY=sk-...` 후 재실행하세요."
        )

    partial_path = output_path.with_suffix(".partial.csv")
    done = load_resume(partial_path)
    if not done:
        done = load_resume(output_path)

    todo_rows = [r for r in df.to_dict(orient="records") if r["row_index"] not in done]
    print(f"[run ] total={len(df):,} done={len(done):,} todo={len(todo_rows):,} "
          f"concurrency={concurrency}")

    if not todo_rows:
        print("[run ] nothing to do, just rewriting final output.")
        save_csv(list(done.values()), output_path)
        return

    client = AsyncOpenAI(api_key=api_key)
    sem    = asyncio.Semaphore(concurrency)

    results: list[dict] = list(done.values())
    tasks = [asyncio.create_task(classify_one(client, sem, r)) for r in todo_rows]

    completed_since_save = 0
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="GPT-4.1-mini label"):
        res = await coro
        results.append(res)
        completed_since_save += 1
        if completed_since_save >= SAVE_INTERVAL:
            save_csv(results, partial_path)
            completed_since_save = 0

    save_csv(results, partial_path)
    os.replace(partial_path, output_path)
    print(f"[done] saved → {output_path}")

    # quick summary
    rdf = pd.read_csv(output_path)
    n_ok  = (rdf["error"].fillna("") == "").sum()
    n_err = len(rdf) - n_ok
    print(f"\n=== Summary ===")
    print(f"total: {len(rdf):,}   ok: {n_ok:,}   err: {n_err:,}")
    if "subject" in rdf:
        print("\nsubject:")
        print(rdf["subject"].value_counts(dropna=False).to_string())
        print("\nlevel:")
        print(rdf["level"].value_counts(dropna=False).sort_index().to_string())
        if "source" in rdf:
            print("\nsource:")
            print(rdf["source"].value_counts(dropna=False).to_string())
    if "latency_s" in rdf and rdf["latency_s"].notna().any():
        l = rdf["latency_s"].dropna()
        print(f"\nlatency_s: mean={l.mean():.2f} p50={l.median():.2f} p95={l.quantile(0.95):.2f}")
    if "prompt_tokens" in rdf:
        tot_in  = rdf["prompt_tokens"].fillna(0).sum()
        tot_out = rdf["completion_tokens"].fillna(0).sum()
        print(f"tokens: prompt={int(tot_in):,} completion={int(tot_out):,}")

    await client.close()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True, help="output CSV path")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke)")
    ap.add_argument("--concurrency", type=int, default=20, help="async semaphore size")
    args = ap.parse_args()

    print(f"[cfg ] prompt_sha={PROMPT_SHA}  model={MODEL}")
    df = load_dataset_rows(limit=args.limit)
    asyncio.run(run(df, args.output, args.concurrency))


if __name__ == "__main__":
    main()
