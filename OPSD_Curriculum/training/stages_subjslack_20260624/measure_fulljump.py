#!/usr/bin/env python3
"""
measure_fulljump.py — 전체표현(난이도+subject) consecutive-stage 점프 측정.

논문 주장 "minimize the representational jump between consecutive stages"를
LEVEL 미제거(=전체 표현) unit centroid로 직접 검증한다. 낮을수록 smoother.

윈도우 4종(W_ALL 전 레이어 / W_SL=W_SUBJ∪W_LEV / W_SUBJ / W_LEV)으로 측정해
전 레이어에서도 결론이 유지되는지 확인한다.

출처: Qwen3-8B THINKING pooled pilot1+pilot2 N=3025 (faithful 폐기). CPU-only.
의존: 같은 폴더 build_stages_subjslack.py (g 계산 + unit-atomic stage 빌드 재사용).

결과(2026-06-24, 기록): W_ALL diff=0.237 / 현ours=0.394 / 새ours(a=2.0)=0.226.
"""
import sys, os, time
from unittest import mock
import numpy as np, pandas as pd, json
from collections import Counter

# matplotlib not in opsd env; helpers we use are pure numpy -> mock it away.
sys.modules['matplotlib'] = mock.MagicMock()
sys.modules['matplotlib.pyplot'] = mock.MagicMock()
sys.modules['matplotlib.cm'] = mock.MagicMock()

R = "/scratch/lami2026/personal/jimin_2782/"
HERE = os.path.dirname(os.path.abspath(__file__))
AN = R + "src/OPSD_Curriculum/reasoning_pivot/activation/analysis"
sys.path.insert(0, HERE)
sys.path.insert(0, AN)

import pooled_analysis as pa, similarity_analysis as sa, clusterderive as cd
from build_stages_subjslack import compute_g_subject, build, ALPHA, PARQUET, N_STAGES

SUBJ7 = ['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra',
         'Number Theory', 'Prealgebra', 'Precalculus']
W_SUBJ = [9, 10, 11, 12, 14]
W_LEV = [20, 25, 26, 27, 29, 30, 31]


def main():
    t0 = time.time()
    _, DAT, md, _ = pa.load_pooled(None)
    assert len(md) == 3025, len(md)
    print(f"[load] {time.time()-t0:.0f}s DAT={DAT.shape}", flush=True)

    mask = md['subject'].isin(SUBJ7).to_numpy()
    md_sub = md.loc[mask].reset_index(drop=True).copy()
    DAT_sub = DAT[mask].astype(np.float32); del DAT
    unit_arr = (md_sub['subject'].astype(str) + "|L" + md_sub['level'].astype(int).astype(str)).to_numpy()
    cnt = Counter(unit_arr); units_keep = sorted([u for u, n in cnt.items() if n >= 30])
    DAn = sa.normalize_members(DAT_sub - DAT_sub.mean(0, keepdims=True))  # level NOT removed
    nL = DAn.shape[1]
    windows = {'W_ALL': list(range(nL)), 'W_SL': sorted(set(W_SUBJ + W_LEV)),
               'W_SUBJ': W_SUBJ, 'W_LEV': W_LEV}
    cents = {wn: cd.unit_centroid_matrix(DAn, unit_arr, units_keep, ly)[1] for wn, ly in windows.items()}

    g, _ = compute_g_subject()
    df = pd.read_parquet(PARQUET)
    df = df[df['in_setA']].reset_index(drop=True).copy()
    df['unit'] = df['subject'].astype(str) + "|L" + df['level'].astype(int).astype(str)
    N = len(df); uidx = {u: i for i, u in enumerate(units_keep)}
    ours, _ = build(df, g, lambda u: u['level'] + ALPHA * u['g'])
    diff, _ = build(df, g, lambda u: u['level'].astype(float))

    def consec(stage, c):
        W = np.zeros((N_STAGES, len(units_keep))); uv = df['unit'].values
        for i in range(N):
            if uv[i] in uidx:
                W[stage[i], uidx[uv[i]]] += 1
        W /= W.sum(1, keepdims=True)
        SC = W @ c; SC /= np.linalg.norm(SC, axis=1, keepdims=True) + 1e-9
        return float(np.mean([1 - SC[s] @ SC[s + 1] for s in range(N_STAGES - 1)]))

    print("\n=== FULL-REPRESENTATION consecutive jump (1-cos, lower=smoother) ===")
    print(f"{'config':>12}" + "".join(f"{w:>9}" for w in windows))
    for name, st in [("diff", diff), ("ours a=%.1f" % ALPHA, ours)]:
        print(f"{name:>12}" + "".join(f"{consec(st, cents[wn]):>9.3f}" for wn in windows), flush=True)
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
