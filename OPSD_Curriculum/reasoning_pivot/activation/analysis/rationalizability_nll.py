#!/usr/bin/env python3
"""
rationalizability_nll.py — H1 (1단계): 표현(g·level)이 base 난이도/친숙도를 예측하는가.

generation 0. base Qwen3-8B에 student 포맷 [문제만] 입력 + reference solution을
forced-decode(prompt_logprobs)하여 solution 토큰 평균 NLL을 잰다.
  NLL 낮음 = base가 그 정답 풀이를 자연스럽게 앎 = 쉬움/친숙.
unit(subject×level)별 평균 NLL → g·level과 Spearman 상관.

주의: 이건 student 관점(해답 없이)의 base 친숙도 = rationalizability의 "앞 화살표"
(표현→난이도). teacher 해답-조건부 rationalizability(뒤 화살표)는 B안(소규모 generation).

student 포맷은 data_collator.py:67과 동일. solution/problem은 OPSD dataset에서.
"""
import sys, glob, hashlib, json
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc
from scipy.stats import spearmanr

R = "/scratch/lami2026/personal/jimin_2782/"
sys.path.insert(0, R + "src/OPSD_Curriculum/training/stages_subjslack_20260624")
from build_stages_subjslack import compute_g_subject  # noqa: E402

OUT = R + "src/OPSD_Curriculum/reasoning_pivot/activation/analysis/rationalizability_nll_out"

def load_opsd():
    files = sorted(glob.glob(R + "cache/huggingface/siyanzhao___openthoughts_math_30k_opsd/default/0.0.0/*/*.arrow"))
    seen, tabs = set(), []
    for f in files:  # de-dup the two mirror dirs
        key = f.split("/")[-1]
        if key in seen: continue
        seen.add(key)
        tabs.append(ipc.open_stream(pa.memory_map(f, "r")).read_all())
    t = pa.concat_tables(tabs)
    prob = t.column("problem").to_pylist(); sol = t.column("solution").to_pylist()
    pid = [hashlib.sha1(p.encode()).hexdigest()[:16] for p in prob]
    return pd.DataFrame({"problem_id": pid, "problem": prob, "solution": sol})


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    opsd = load_opsd()
    print(f"[opsd] {len(opsd)} rows")

    # pilot universe (subject,level) — same population g was computed on
    pil = pd.read_parquet(R + "src/OPSD_Curriculum/labeling/outputs/pilot_universe_candidate.parquet")
    pil = pil[["problem_id", "subject", "level"]].drop_duplicates("problem_id")
    df = pil.merge(opsd, on="problem_id", how="inner").dropna(subset=["problem", "solution", "subject", "level"])
    g, _ = compute_g_subject()
    df["g"] = df["subject"].map(g)
    df = df[df["g"].notna()].reset_index(drop=True)
    print(f"[merged] {len(df)} problems with (solution, subject, level, g)")

    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.90, enforce_eager=False)
    tok = llm.get_tokenizer()

    def student_ids(problem):
        msg = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       tokenize=True, add_generation_prompt=True, enable_thinking=False)

    prompts, starts, rows = [], [], []
    for r in df.itertuples():
        sids = student_ids(r.problem)
        solids = tok(r.solution, add_special_tokens=False).input_ids
        full = sids + solids
        if len(full) > 8000 or len(solids) < 5:
            continue
        prompts.append({"prompt_token_ids": full}); starts.append(len(sids)); rows.append(r.Index)
    print(f"[prompts] {len(prompts)} (skipped {len(df)-len(prompts)} too-long/short)")

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outs = llm.generate(prompts, sp)

    nlls = []
    for out, start in zip(outs, starts):
        pt = out.prompt_token_ids; plp = out.prompt_logprobs
        lps = [plp[i][pt[i]].logprob for i in range(start, len(pt))
               if plp[i] is not None and pt[i] in plp[i]]
        nlls.append(-float(np.mean(lps)) if lps else np.nan)

    res = df.loc[rows].copy(); res["nll"] = nlls
    res = res.dropna(subset=["nll"])
    res["unit"] = res["subject"] + "|L" + res["level"].astype(int).astype(str)
    res[["problem_id", "subject", "level", "g", "nll"]].to_parquet(OUT + "/per_problem_nll.parquet")

    print("\n=== correlations (per-problem) ===")
    print(f"  Spearman(NLL, level)   = {spearmanr(res['level'], res['nll'])}")
    print(f"  Spearman(NLL, g)       = {spearmanr(res['g'], res['nll'])}")
    # partial: does g predict NLL beyond level?  (residualize NLL on level, corr with g)
    import numpy as _np
    lvl = res['level'].values.astype(float); nll = res['nll'].values; gg = res['g'].values
    b = _np.polyfit(lvl, nll, 1); nll_res = nll - _np.polyval(b, lvl)
    print(f"  Spearman(NLL|level-resid, g) = {spearmanr(gg, nll_res)}  <- subject-geometry beyond difficulty")

    print("\n=== per-subject mean NLL (vs g) ===")
    bs = res.groupby("subject").agg(nll=("nll", "mean"), g=("g", "first"), n=("nll", "size"))
    print(bs.sort_values("g").to_string())
    print("\n=== per-unit mean NLL ===")
    bu = res.groupby("unit").agg(nll=("nll", "mean"), level=("level", "first"), g=("g", "first"), n=("nll", "size"))
    print(bu.sort_values("nll").to_string())
    bs.to_csv(OUT + "/by_subject.csv"); bu.to_csv(OUT + "/by_unit.csv")
    print("\n[saved]", OUT)


if __name__ == "__main__":
    main()
