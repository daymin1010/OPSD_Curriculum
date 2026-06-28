#!/usr/bin/env python
"""
MANIFEST_AUDIT — check fairness of mini50/mini100/q4/full subsamples.

For each (rung, arm) ∈ {full, q4, mini100, mini50} × {diff, ours}:
  - extract problem_id set from stage manifest
  - compute MD5 hash, N, per-stage N
  - join to row_table to get subject/level distribution per stage
  - compute schedule T_total and stage boundaries with B_glob=32, tail_policy=partial

Also checks:
  - identity: diff.problem_id_set == ours.problem_id_set per rung
  - nesting:  mini50 ⊂ mini100 ⊂ q4 ⊂ full (per arm)
  - subject/level proportions preserved across rungs

Output: src/OPSD_Curriculum/prompts/MANIFEST_AUDIT_2026-06-24.md
"""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd

REPO = Path('/scratch/lami2026/personal/jimin_2782')
STAGES_DIR = REPO / 'src/OPSD_Curriculum/training/stages_tiered_20260622'
ROW_TABLE = REPO / 'src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet'
OUT_PATH = REPO / 'src/OPSD_Curriculum/prompts/MANIFEST_AUDIT_2026-06-24.md'
B_GLOB = 32

RUNGS = [
    ('full',    'stages_cond2_diff.json',                'stages_cond3_ours_C2.json'),
    ('q4',      'stages_cond2_diff_q4.json',             'stages_cond3_ours_C2_q4.json'),
    ('mini100', 'stages_cond2_diff_mini100.json',        'stages_cond3_ours_C2_mini100.json'),
    ('mini50',  'stages_cond2_diff_mini50.json',         'stages_cond3_ours_C2_mini50.json'),
]


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        m = json.load(f)
    stages = []
    for st in m['stages']:
        stages.append({
            'stage_index': st.get('stage_index'),
            'order_index': st.get('order_index'),
            'n':            st.get('n'),
            'problem_ids':  list(st['problem_ids']),
        })
    return {'spec': m.get('spec', path.stem), 'stages': stages}


def md5_of_ids(ids: set[str]) -> str:
    h = hashlib.md5()
    for pid in sorted(ids):
        h.update(pid.encode())
    return h.hexdigest()[:12]


def stage_steps(n_problems: int, B_glob: int, tail_policy: str = 'partial') -> int:
    """Number of optimizer steps for a stage with n_problems and global batch B_glob."""
    if tail_policy == 'partial':
        return math.ceil(n_problems / B_glob)
    elif tail_policy == 'drop':
        return n_problems // B_glob
    raise ValueError(tail_policy)


def schedule(manifest: dict) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Return (T_total, [(order_index, stage_index, n, T_stage, cum_T)])."""
    stages = sorted(manifest['stages'], key=lambda s: s['order_index'])
    out, cum = [], 0
    for st in stages:
        T = stage_steps(st['n'], B_GLOB, 'partial')
        cum += T
        out.append((st['order_index'], st['stage_index'], st['n'], T, cum))
    return cum, out


def main() -> None:
    print('[load] row_table')
    row_df = pq.read_table(str(ROW_TABLE)).to_pandas()
    # Deduplicate by problem_id: row_table has one row per (problem_id, row_index). We
    # only need problem-level metadata (subject/level), so collapse to first row.
    pid_to_meta = (
        row_df.drop_duplicates('problem_id')
              .set_index('problem_id')[['subject', 'level', 'subject_cluster']]
    )
    print(f'[load] row_table: {len(row_df)} rows -> {len(pid_to_meta)} unique problem_ids')

    # Load all manifests
    data = {}  # rung -> arm -> manifest dict
    for rung, diff_name, ours_name in RUNGS:
        data[rung] = {
            'diff': load_manifest(STAGES_DIR / diff_name),
            'ours': load_manifest(STAGES_DIR / ours_name),
        }

    # ------------------------------------------------------------------ analyze
    lines: list[str] = []
    P = lines.append
    P('# MANIFEST AUDIT — universe identity, nesting, proportions, schedule')
    P('')
    P('Source row table: `src/OPSD_Curriculum/training/outputs/join_setA_rows.parquet`')
    P(f'B_glob = {B_GLOB}, tail_policy = partial')
    P('')

    # 1. Universe identity per rung
    P('## 1. Universe identity (diff vs ours, per rung)')
    P('')
    P('| rung | N(diff) | N(ours) | MD5(diff) | MD5(ours) | identical | symdiff |')
    P('|---|---|---|---|---|---|---|')
    universe = {}
    for rung in [r[0] for r in RUNGS]:
        d_ids = {pid for st in data[rung]['diff']['stages']  for pid in st['problem_ids']}
        o_ids = {pid for st in data[rung]['ours']['stages']  for pid in st['problem_ids']}
        sd = (d_ids ^ o_ids)
        universe[rung] = {'diff': d_ids, 'ours': o_ids, 'union': d_ids | o_ids}
        P(f'| {rung} | {len(d_ids)} | {len(o_ids)} | `{md5_of_ids(d_ids)}` | `{md5_of_ids(o_ids)}` | {"✅" if d_ids==o_ids else "❌"} | {len(sd)} |')
    P('')

    # 2. Nesting
    P('## 2. Nesting check: mini50 ⊂ mini100 ⊂ q4 ⊂ full (per arm)')
    P('')
    P('| arm | mini50⊂mini100 | mini100⊂q4 | q4⊂full | mini50⊂full |')
    P('|---|---|---|---|---|')
    for arm in ['diff', 'ours']:
        m50, m100, q4, fu = (universe[r][arm] for r in ['mini50','mini100','q4','full'])
        cells = [m50<=m100, m100<=q4, q4<=fu, m50<=fu]
        P(f'| {arm} | {"✅" if cells[0] else "❌"} | {"✅" if cells[1] else "❌"} | {"✅" if cells[2] else "❌"} | {"✅" if cells[3] else "❌"} |')
    P('')

    # 3. Stage proportions per (rung × arm × stage)
    P('## 3. Stage composition (N, mean level, subject mix)')
    P('')
    for rung in [r[0] for r in RUNGS]:
        P(f'### rung = {rung}')
        P('')
        P('| arm | order | stage_idx | N | level μ | level σ | top-3 subjects |')
        P('|---|---|---|---|---|---|---|')
        for arm in ['diff', 'ours']:
            stages = sorted(data[rung][arm]['stages'], key=lambda s: s['order_index'])
            for st in stages:
                sub = pid_to_meta.reindex(st['problem_ids']).dropna()
                lv = sub['level']
                subj = Counter(sub['subject'].astype(str))
                top3 = ', '.join(f'{k}({v})' for k, v in subj.most_common(3))
                P(f'| {arm} | {st["order_index"]} | {st["stage_index"]} | {st["n"]} | {lv.mean():.2f} | {lv.std():.2f} | {top3} |')
        P('')

    # 4. Schedule (T_total + stage boundaries)
    P('## 4. Schedule (B_glob=32, tail_policy=partial)')
    P('')
    P('| rung | arm | T_total | per-stage (order: n → T_stage → cum_T) |')
    P('|---|---|---|---|')
    for rung in [r[0] for r in RUNGS]:
        for arm in ['diff', 'ours']:
            T_total, sch = schedule(data[rung][arm])
            cells = ' ; '.join(f'{o}: {n}→{T}→{cum}' for o, _, n, T, cum in sch)
            P(f'| {rung} | {arm} | **{T_total}** | {cells} |')
    P('')

    # 5. Sanity: are q4/mini100/mini50 actually nested subsets w.r.t. proportional stratification?
    #    Show stage-N ratios across rungs (per stage) per arm.
    P('## 5. Per-stage N ratios across rungs (proportionality sanity)')
    P('')
    for arm in ['diff', 'ours']:
        P(f'### arm = {arm}')
        P('')
        P('| stage(order) | full | q4 | q4/full | mini100 | m100/q4 | mini50 | m50/q4 |')
        P('|---|---|---|---|---|---|---|---|')
        # Build dict: arm -> order -> n
        per_rung = {r: {st['order_index']: st['n'] for st in data[r][arm]['stages']} for r in ['full','q4','mini100','mini50']}
        for o in sorted(per_rung['full']):
            fu = per_rung['full'].get(o, 0)
            q4 = per_rung['q4'].get(o, 0)
            m1 = per_rung['mini100'].get(o, 0)
            m5 = per_rung['mini50'].get(o, 0)
            r_q4 = q4 / fu if fu else float('nan')
            r_m1 = m1 / q4 if q4 else float('nan')
            r_m5 = m5 / q4 if q4 else float('nan')
            P(f'| {o} | {fu} | {q4} | {r_q4:.3f} | {m1} | {r_m1:.3f} | {m5} | {r_m5:.3f} |')
        P('')

    # 6. Cross-arm overlap: are q4 and mini same in both arms?
    P('## 6. Cross-arm universe identity (same problems used in both arms?)')
    P('')
    P('| rung | |U(diff) ∩ U(ours)| | |U(diff) ∪ U(ours)| | Jaccard |')
    P('|---|---|---|---|')
    for rung in [r[0] for r in RUNGS]:
        i = len(universe[rung]['diff'] & universe[rung]['ours'])
        u = len(universe[rung]['union'])
        j = i / u if u else float('nan')
        P(f'| {rung} | {i} | {u} | {j:.4f} |')
    P('')

    # 7. Subject/level distribution across rungs (single arm = diff; ours mirror)
    P('## 7. Overall level histogram per rung (arm=diff; ours mirrors by §1)')
    P('')
    P('| rung | N | level=1 | 2 | 3 | 4 | 5 | 6 | 7 | mean | std |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for rung in [r[0] for r in RUNGS]:
        pids = universe[rung]['diff']
        sub = pid_to_meta.reindex(list(pids)).dropna()
        lv = sub['level']
        bins = Counter(int(x) for x in lv)
        row = [rung, len(pids)] + [bins.get(k, 0) for k in range(1, 8)] + [f'{lv.mean():.2f}', f'{lv.std():.2f}']
        P('| ' + ' | '.join(str(c) for c in row) + ' |')
    P('')

    P('## 8. Subject distribution per rung (arm=diff)')
    P('')
    all_subj = sorted(pid_to_meta['subject'].dropna().unique())
    header = '| rung | N | ' + ' | '.join(all_subj) + ' |'
    sep    = '|---|---|' + '|'.join(['---'] * len(all_subj)) + '|'
    P(header)
    P(sep)
    for rung in [r[0] for r in RUNGS]:
        pids = universe[rung]['diff']
        sub = pid_to_meta.reindex(list(pids)).dropna()
        cnt = Counter(sub['subject'].astype(str))
        row = [rung, len(pids)] + [cnt.get(s, 0) for s in all_subj]
        P('| ' + ' | '.join(str(c) for c in row) + ' |')
    P('')

    OUT_PATH.write_text('\n'.join(lines))
    print(f'[ok] wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
