#!/usr/bin/env python3
"""
curriculum_schedule.py
======================
Curriculum LAYER for OPSD self-distillation. **Data-ordering ONLY.**
Does NOT touch opsd_trainer.py / data_collator.py / loss / generation.

Two responsibilities:
  (1) Phase-0  : join OPSD train rows -> (subject, level) labels, map to
                 difficulty (D1..D4) + subject_cluster, filter Set-A
                 (exclude 'Other' + unlabeled). GATE on coverage >= 0.95.
  (2) Schedule : given a stages JSON + the joined row->(diff,subject) map +
                 (T, B_glob, seed), build a deterministic example-index
                 schedule that walks the stages in JSON order, each stage
                 getting round(T/num_stages) optimizer steps worth of slots,
                 filled by seeded shuffle + cycling, padded to a multiple of
                 B_glob so no optimizer step straddles two stages.

The schedule is consumed by CurriculumIndexDataset (curriculum_trainer.py),
which is fed to a SequentialSampler so the schedule literally IS the order.

CPU only. Deterministic. New file — OPSD_original untouched.

Join key (IMPORTANT):
  `siyanzhao/Openthoughts_math_30k_opsd` has NO native problem_id column
  (only problem/solution/source/Answer/generated_token_count). Our labels
  parquet's problem_id == sha1(problem_text)[:16] where problem_text is a
  verbatim copy of the OPSD `problem` field (see labeling/postprocess.py).
  So we RE-DERIVE problem_id on the OPSD side with the identical sha1 and
  join on it. This is deterministic; expected match rate ~= 1.0.
  Fallbacks (exact text, normalized text) are kept for auto-detect/report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                       # .../training/curriculum
TRAIN_ROOT = HERE.parent                                     # .../training
REPO_ROOT = Path("/scratch/lami2026/personal/jimin_2782")

LABELS_PARQUET = (
    REPO_ROOT
    / "src/OPSD_Curriculum/labeling/outputs/openthoughts_30k_labels_final.parquet"
)
STAGES_MAIN_SRC = (
    REPO_ROOT
    / "src/OPSD_Curriculum/reasoning_pivot/activation/analysis/clustering/stages_arm3_excludeOther.json"
)
STAGES_DIR = TRAIN_ROOT / "stages"
OUT_DIR = TRAIN_ROOT / "outputs"

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"

DIFFICULTY_AXIS = {"D1": [1, 2], "D2": [3, 4], "D3": [5, 6], "D4": [7, 8]}
DIFF_ORDER = ["D1", "D2", "D3", "D4"]
EXCLUDE_SUBJECT = "Other"
COVERAGE_GATE = 0.95


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sha1_id(s) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def norm_ws(s) -> str:
    """Whitespace-normalized text (fallback join key)."""
    return re.sub(r"\s+", " ", str(s)).strip()


def level_to_difficulty(level: int) -> str | None:
    for d, lv in DIFFICULTY_AXIS.items():
        if int(level) in lv:
            return d
    return None


def subject_to_cluster(subject: str, clusters: dict[str, list[str]]) -> str | None:
    for cname, members in clusters.items():
        if subject in members:
            return cname
    return None


def load_opsd_problems():
    """Return a pandas DataFrame with the OPSD train split's problem/solution
    text + an opsd_index column (positional row index in dataset order).

    NOTE (data-loading layer): we do NOT use `load_dataset()` because the cached
    `dataset_info.json` was written by a newer `datasets` (Feature type 'List')
    and is unreadable by the OPSD env's pinned datasets==3.6.0. `opsd_data`
    reads the Arrow shards directly (both shards, dataset order preserved).
    """
    from opsd_data import load_opsd_train

    ds = load_opsd_train()

    cols = ds.column_names
    print(f"[opsd] loaded {len(ds)} rows; columns = {cols}", flush=True)
    # keep only what we need for join (problem) + sanity (solution presence)
    problems = ds["problem"]
    df = pd.DataFrame({"opsd_index": np.arange(len(ds)), "problem": problems})
    df["has_solution"] = [bool(x) for x in ds["solution"]] if "solution" in cols else True
    df["opsd_columns"] = json.dumps(cols)  # carried for the report (constant)
    return df, cols


# ----------------------------------------------------------------------------
# Phase 0 — join + coverage + Set-A cell counts
# ----------------------------------------------------------------------------
def auto_join(opsd_df: pd.DataFrame, labels: pd.DataFrame):
    """
    Attach (subject, level, problem_id, row_index) from labels to each OPSD row.
    Auto-detect the join key, preferring the deterministic sha1 reconstruction.
    Returns (joined_df, join_info dict).
    """
    n_opsd = len(opsd_df)
    labels = labels.copy()

    # --- key (a'): re-derive problem_id on OPSD side via identical sha1 ---
    opsd_df = opsd_df.copy()
    opsd_df["problem_id"] = opsd_df["problem"].map(sha1_id)

    lab_by_pid = labels.drop_duplicates("problem_id").set_index("problem_id")
    merged = opsd_df.merge(
        lab_by_pid[["row_index", "subject", "level"]],
        left_on="problem_id",
        right_index=True,
        how="left",
    )
    n_match_pid = int(merged["subject"].notna().sum())
    rate_pid = n_match_pid / n_opsd

    join_info = {
        "n_opsd": n_opsd,
        "n_labels": len(labels),
        "key_used": None,
        "match_rate": None,
        "candidates": {
            "sha1_problem_id": {"matched": n_match_pid, "rate": rate_pid},
        },
    }

    if rate_pid >= COVERAGE_GATE:
        join_info["key_used"] = "sha1(problem)[:16] == labels.problem_id"
        join_info["match_rate"] = rate_pid
        return merged, join_info

    # --- fallback (b): exact problem-text match ---
    lab_by_text = labels.drop_duplicates("problem_text").set_index("problem_text")
    merged_b = opsd_df.drop(columns=["row_index", "subject", "level"], errors="ignore").merge(
        lab_by_text[["row_index", "subject", "level"]],
        left_on="problem",
        right_index=True,
        how="left",
    )
    n_match_text = int(merged_b["subject"].notna().sum())
    rate_text = n_match_text / n_opsd
    join_info["candidates"]["exact_text"] = {"matched": n_match_text, "rate": rate_text}

    if rate_text >= COVERAGE_GATE and rate_text > rate_pid:
        join_info["key_used"] = "exact problem-text match"
        join_info["match_rate"] = rate_text
        return merged_b, join_info

    # --- fallback (c): whitespace-normalized text match ---
    opsd_df["problem_norm"] = opsd_df["problem"].map(norm_ws)
    labels["problem_norm"] = labels["problem_text"].map(norm_ws)
    lab_by_norm = labels.drop_duplicates("problem_norm").set_index("problem_norm")
    merged_c = opsd_df.drop(columns=["row_index", "subject", "level"], errors="ignore").merge(
        lab_by_norm[["row_index", "subject", "level"]],
        left_on="problem_norm",
        right_index=True,
        how="left",
    )
    n_match_norm = int(merged_c["subject"].notna().sum())
    rate_norm = n_match_norm / n_opsd
    join_info["candidates"]["normalized_text"] = {"matched": n_match_norm, "rate": rate_norm}

    # pick the best of all candidates
    best = max(
        [("sha1_problem_id", rate_pid, merged),
         ("exact_text", rate_text, merged_b),
         ("normalized_text", rate_norm, merged_c)],
        key=lambda t: t[1],
    )
    join_info["key_used"] = best[0]
    join_info["match_rate"] = best[1]
    return best[2], join_info


def run_phase0(stages_main_path: Path, write_diffonly: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGES_DIR.mkdir(parents=True, exist_ok=True)

    assert LABELS_PARQUET.exists(), f"labels parquet missing: {LABELS_PARQUET}"
    assert stages_main_path.exists(), f"stages JSON missing: {stages_main_path}"

    stages_main = json.loads(stages_main_path.read_text())
    clusters = stages_main["subject_clusters"]            # C1..C4 -> members
    all_setA_subjects = sorted({s for m in clusters.values() for s in m})

    labels = pd.read_parquet(
        LABELS_PARQUET, columns=["problem_id", "row_index", "subject", "level", "problem_text"]
    )
    print(f"[labels] {len(labels)} rows; subjects={sorted(labels['subject'].unique())}", flush=True)

    opsd_df, opsd_cols = load_opsd_problems()
    joined, jinfo = auto_join(opsd_df, labels)

    print(f"[join] key={jinfo['key_used']} match_rate={jinfo['match_rate']:.4f}", flush=True)

    # ---- map difficulty + cluster ----
    joined["difficulty"] = joined["level"].map(
        lambda x: level_to_difficulty(x) if pd.notna(x) else None
    )
    joined["subject_cluster"] = joined["subject"].map(
        lambda s: subject_to_cluster(s, clusters) if pd.notna(s) else None
    )

    # ---- Set-A membership: matched, not Other, has difficulty + cluster ----
    matched = joined["subject"].notna()
    not_other = joined["subject"] != EXCLUDE_SUBJECT
    has_diff = joined["difficulty"].notna()
    has_cluster = joined["subject_cluster"].notna()
    in_setA = matched & not_other & has_diff & has_cluster
    joined["in_setA"] = in_setA

    # ---- stage_index assignment ----
    # main (③): difficulty x subject_cluster -> stage_index
    main_cell_to_stage = {
        (st["difficulty_cluster"], st["subject_cluster"]): st["stage_index"]
        for st in stages_main["stages"]
    }
    joined["stage_index_main"] = [
        main_cell_to_stage.get((d, c)) if (s and pd.notna(d) and pd.notna(c)) else None
        for d, c, s in zip(joined["difficulty"], joined["subject_cluster"], in_setA)
    ]
    # diffonly (②): difficulty -> stage_index (D1->0 .. D4->3)
    diff_to_stage = {d: i for i, d in enumerate(DIFF_ORDER)}
    joined["stage_index_diffonly"] = [
        diff_to_stage.get(d) if s and pd.notna(d) else None
        for d, s in zip(joined["difficulty"], in_setA)
    ]

    # ---- coverage gate ----
    gate_pass = jinfo["match_rate"] >= COVERAGE_GATE

    # ---- per-cell counts on Set-A ----
    setA = joined[in_setA]
    n_setA = int(in_setA.sum())
    n_total_opsd = len(joined)
    n_unmatched = int((~matched).sum())
    n_other = int((joined["subject"] == EXCLUDE_SUBJECT).sum())

    # main 4x4
    main_cells = {}
    for d in DIFF_ORDER:
        for cname in sorted(clusters.keys()):
            cnt = int(((setA["difficulty"] == d) & (setA["subject_cluster"] == cname)).sum())
            main_cells[f"{d}|{cname}"] = cnt
    # diffonly 4x1
    diff_cells = {d: int((setA["difficulty"] == d).sum()) for d in DIFF_ORDER}

    # ---- write diffonly stages JSON ----
    diffonly_path = STAGES_DIR / "stages_diffonly_setA.json"
    if write_diffonly:
        diffonly = {
            "spec": "arm2_diffonly_setA",
            "difficulty_axis": DIFFICULTY_AXIS,
            "exclude_subject": EXCLUDE_SUBJECT,
            "subject_set_all": all_setA_subjects,
            "stages": [
                {
                    "stage_index": i,
                    "level_set": DIFFICULTY_AXIS[d],
                    "subject_set": all_setA_subjects,   # subjects ignored
                    "difficulty_cluster": d,
                    "subject_cluster": "ALL",
                }
                for i, d in enumerate(DIFF_ORDER)
            ],
        }
        diffonly_path.write_text(json.dumps(diffonly, indent=2))
        print(f"[diffonly] wrote {diffonly_path}", flush=True)

    # ---- copy main stages JSON into training/stages for locality ----
    main_local = STAGES_DIR / "stages_arm3_excludeOther.json"
    main_local.write_text(json.dumps(stages_main, indent=2))

    # ---- per-row table ----
    row_cols = ["opsd_index", "problem_id", "row_index", "subject", "level",
                "difficulty", "subject_cluster", "stage_index_main",
                "stage_index_diffonly", "in_setA"]
    row_tab = joined[row_cols].copy()
    row_path = OUT_DIR / "join_setA_rows.parquet"
    row_tab.to_parquet(row_path, index=False)

    # ---- report ----
    L = []
    L.append("# Phase-0 — OPSD↔labels join + Set-A coverage (REPORT_join_setA)")
    L.append("")
    L.append("작성: curriculum_schedule.py run_phase0 (CPU, deterministic)")
    L.append(f"- OPSD dataset: `{OPSD_DATASET}` (train split)")
    L.append(f"- OPSD columns: `{opsd_cols}`")
    L.append(f"- labels parquet: `{LABELS_PARQUET.name}` ({len(labels)} rows)")
    L.append("")
    L.append("## 0. Join key auto-detect")
    L.append(f"- **key used: `{jinfo['key_used']}`**")
    L.append(f"- **match rate = {jinfo['match_rate']:.4f}** "
             f"({int(jinfo['match_rate']*n_total_opsd)}/{n_total_opsd})")
    L.append(f"- coverage gate (>= {COVERAGE_GATE}): "
             f"**{'PASS' if gate_pass else 'FAIL → STOP'}**")
    L.append("- candidate keys:")
    for k, v in jinfo["candidates"].items():
        L.append(f"    - {k}: matched={v['matched']} rate={v['rate']:.4f}")
    L.append("")
    L.append("## 1. Universe accounting")
    L.append(f"- OPSD train rows           : {n_total_opsd}")
    L.append(f"- unmatched (no label)      : {n_unmatched}")
    L.append(f"- matched but subject=Other : {n_other}")
    L.append(f"- **Set-A trainable total   : {n_setA}**")
    L.append(f"- labels-parquet Set-A ref  : ~28,771 (computed on parquet alone)")
    gap = 28771 - n_setA
    L.append(f"- gap vs 28,771             : {gap}  "
             f"({'OK / small' if abs(gap) < 500 else '⚠ LARGE — investigate'})")
    L.append("")
    L.append("## 2. Main (③-A) per-cell counts — difficulty × subject_cluster (Set-A)")
    L.append("")
    L.append("| difficulty | " + " | ".join(sorted(clusters.keys())) + " |")
    L.append("|---|" + "---|" * len(clusters))
    for d in DIFF_ORDER:
        row = " | ".join(str(main_cells[f"{d}|{c}"]) for c in sorted(clusters.keys()))
        L.append(f"| {d}{DIFFICULTY_AXIS[d]} | {row} |")
    L.append("")
    L.append("- clusters: " + "; ".join(f"{c}{{{', '.join(m)}}}" for c, m in clusters.items()))
    mn = min(main_cells.values()); n_empty = sum(1 for v in main_cells.values() if v == 0)
    L.append(f"- min cell = {mn}; #empty = {n_empty}")
    L.append("")
    L.append("## 3. Diff-only (②-A) per-stage counts (Set-A)")
    L.append("")
    L.append("| stage | difficulty | n |")
    L.append("|---|---|---|")
    for i, d in enumerate(DIFF_ORDER):
        L.append(f"| {i} | {d}{DIFFICULTY_AXIS[d]} | {diff_cells[d]} |")
    L.append("")
    L.append("## 4. Outputs")
    L.append(f"- per-row table: `outputs/{row_path.name}` ({len(row_tab)} rows)")
    L.append(f"- diff-only stages JSON: `stages/{diffonly_path.name}`")
    L.append(f"- main stages JSON (local copy): `stages/{main_local.name}`")
    L.append("")
    if not gate_pass:
        L.append("## ⚠ STOP — coverage below gate. Sample unmatched OPSD rows:")
        unm = joined[~matched].head(10)
        for _, r in unm.iterrows():
            snippet = str(r["problem"])[:160].replace("\n", " ")
            L.append(f"- opsd_index={r['opsd_index']} sha1={r['problem_id']}: {snippet}")

    report_path = OUT_DIR / "REPORT_join_setA.md"
    report_path.write_text("\n".join(L))
    print(f"[report] wrote {report_path}", flush=True)

    # also dump machine-readable join meta
    meta = {
        "join_info": jinfo,
        "gate_pass": bool(gate_pass),
        "n_setA": n_setA,
        "n_total_opsd": n_total_opsd,
        "n_unmatched": n_unmatched,
        "n_other": n_other,
        "main_cells": main_cells,
        "diff_cells": diff_cells,
        "diffonly_stages_json": str(diffonly_path),
        "main_stages_json": str(main_local),
        "row_table": str(row_path),
    }
    (OUT_DIR / "join_setA_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[phase0] gate_pass={gate_pass} n_setA={n_setA} "
          f"main_min_cell={mn} empty={n_empty}", flush=True)
    if not gate_pass:
        print("[phase0] COVERAGE GATE FAILED — not proceeding to training.", flush=True)
        sys.exit(2)
    return meta


# ----------------------------------------------------------------------------
# Schedule builder (stages-JSON-driven)
# ----------------------------------------------------------------------------
def build_schedule(stages_json_path: Path, row_table_path: Path, arm: str,
                   T: int, B_glob: int, seed: int = 42):
    """
    Build a deterministic example-index schedule of length ~= T*B_glob.

    arm: 'main' uses stage_index_main; 'diffonly' uses stage_index_diffonly.
    Each stage gets budget = round(T/num_stages) optimizer steps -> budget*B_glob
    example slots, padded to a multiple of B_glob (already is). Pools filled by
    seeded shuffle + cycling. Returns (schedule list[int over opsd_index], meta).
    """
    stages = json.loads(Path(stages_json_path).read_text())["stages"]
    num_stages = len(stages)
    stage_col = "stage_index_main" if arm == "main" else "stage_index_diffonly"

    rows = pd.read_parquet(row_table_path, columns=["opsd_index", stage_col, "in_setA"])
    rows = rows[rows["in_setA"]]

    budget = round(T / num_stages)              # optimizer steps per stage
    slots_per_stage = budget * B_glob

    schedule: list[int] = []
    meta_stages = []
    for st in stages:
        sidx = st["stage_index"]
        pool = rows.loc[rows[stage_col] == sidx, "opsd_index"].to_numpy()
        pool_size = int(len(pool))
        if pool_size == 0:
            raise ValueError(f"[schedule] arm={arm} stage {sidx} has EMPTY pool")

        filled = []
        cycles = 0
        sub = 0
        while len(filled) < slots_per_stage:
            rng = np.random.default_rng((seed ^ (sidx * 1_000_003)) + sub)
            perm = rng.permutation(pool)
            take = min(slots_per_stage - len(filled), len(perm))
            filled.extend(perm[:take].tolist())
            cycles += 1
            sub += 1
        # already a multiple of B_glob since slots_per_stage = budget*B_glob
        schedule.extend(int(x) for x in filled)

        meta_stages.append({
            "stage_index": sidx,
            "difficulty": st.get("difficulty_cluster"),
            "subject_cluster": st.get("subject_cluster"),
            "pool_size": pool_size,
            "slots": slots_per_stage,
            "cycles": cycles,
            "opt_steps": budget,
        })

    meta = {
        "arm": arm,
        "stages_json": str(stages_json_path),
        "T": T,
        "B_glob": B_glob,
        "seed": seed,
        "num_stages": num_stages,
        "budget_per_stage": budget,
        "schedule_len": len(schedule),
        "stages": meta_stages,
    }
    return schedule, meta


def derive_stage_maps(meta: dict, B_glob: int):
    """From a build_schedule() meta dict, derive the per-position and
    per-optimizer-step stage labels (in schedule order).

      stage_per_pos[i]      = stage of the i-th example in the (reordered) dataset
                              -> attached as the `stage_index` dataset column.
      stage_per_optstep[s]  = stage that optimizer step s should be training on
                              -> the monitor's source of truth for stage_expected.

    Invariant: slots == opt_steps * B_glob for every stage (build_schedule pads
    to a multiple of B_glob), so no optimizer step straddles two stages.
    """
    stage_per_pos: list[int] = []
    stage_per_optstep: list[int] = []
    for st in meta["stages"]:
        sidx = int(st["stage_index"])
        slots = int(st["slots"])
        opt_steps = int(st["opt_steps"])
        assert slots == opt_steps * B_glob, (sidx, slots, opt_steps, B_glob)
        stage_per_pos.extend([sidx] * slots)
        stage_per_optstep.extend([sidx] * opt_steps)
    return stage_per_pos, stage_per_optstep


def save_schedule_meta(meta: dict, arm: str):

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"schedule_meta_{arm}.json"
    p.write_text(json.dumps(meta, indent=2))
    print(f"[schedule] wrote {p}", flush=True)
    return p


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("phase0", help="join + coverage gate + Set-A counts")
    p0.add_argument("--stages_main", default=str(STAGES_MAIN_SRC))
    p0.add_argument("--no_diffonly", action="store_true")

    ps = sub.add_parser("schedule", help="build a schedule (debug/inspection)")
    ps.add_argument("--stages_json", required=True)
    ps.add_argument("--arm", choices=["main", "diffonly"], required=True)
    ps.add_argument("--T", type=int, required=True)
    ps.add_argument("--B_glob", type=int, required=True)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--row_table", default=str(OUT_DIR / "join_setA_rows.parquet"))

    args = ap.parse_args()
    if args.cmd == "phase0":
        run_phase0(Path(args.stages_main), write_diffonly=not args.no_diffonly)
    elif args.cmd == "schedule":
        sch, meta = build_schedule(Path(args.stages_json), Path(args.row_table),
                                   args.arm, args.T, args.B_glob, args.seed)
        save_schedule_meta(meta, args.arm)
        print(f"[schedule] len={len(sch)} (T*B_glob={args.T*args.B_glob})", flush=True)


if __name__ == "__main__":
    main()
