#!/usr/bin/env python3
"""
stagebuild.py — OPSD 5-stage pure stage construction (CPU only)
================================================================
Build conditions ①..⑤ from labeled OPSD curriculum data using the already-adopted
clusterderive residual unit-centroid pipeline. This script is bookkeeping/order
construction only: it reuses pooled_analysis/load_pooled, sa.normalize_members,
ssg.level_centroid_residual, and clusterderive centroid/Ward helpers.

Outputs are written only to .../training/stages/ by default:
  stages_cond1_random_seed{S}.json
  stages_cond2_diff.json
  stages_cond3_ours_C2.json
  stages_cond4_shuffle_seed{S}.json
  stages_cond5_diffmatched_seed{S}.json
  manifest.json
  REPORT_stagebuild_YYYY-MM-DD.md
  stagebuild_artifacts.npz
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

# analysis helpers (same directory)
import pooled_analysis as pa
import similarity_analysis as sa
import subject_similarity_gate as ssg
import subject_layer_resolved as slr
import clusterderive as cd

REPO_ROOT = Path("/scratch/lami2026/personal/jimin_2782")
ANALYSIS = Path(__file__).resolve().parent
CURRICULUM_DIR = REPO_ROOT / "src/OPSD_Curriculum/training/curriculum"
TRAIN_STAGES = REPO_ROOT / "src/OPSD_Curriculum/training/stages"
sys.path.insert(0, str(CURRICULUM_DIR))
import curriculum_schedule as cs  # noqa: E402

W_SUBJ = cd.W_SUBJ
MIN_UNIT_N = cd.MIN_UNIT_N
SUBJECT_CLUSTERS_ALL = {
    "C_alg": ["Algebra", "Intermediate Algebra", "Precalculus"],
    "C_geo": ["Geometry"],
    "C_disc": ["Counting & Probability", "Number Theory", "Prealgebra"],
    "C_other": ["Other"],
}


def parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def unit_of(subject: str, level: int) -> str:
    return f"{subject}|L{int(level)}"


def subj_of(unit: str) -> str:
    return unit.split("|L")[0]


def lev_of(unit: str) -> int:
    return int(unit.split("|L")[1])


def subject_cluster_map(include_other: bool) -> tuple[dict[str, str], dict[str, list[str]]]:
    clusters = {k: v[:] for k, v in SUBJECT_CLUSTERS_ALL.items() if include_other or k != "C_other"}
    smap = {}
    for c, members in clusters.items():
        for s in members:
            smap[s] = c
    return smap, clusters


def md_table_from_counts(df: pd.DataFrame, row: str, col: str, row_order=None, col_order=None) -> str:
    ct = pd.crosstab(df[row], df[col])
    if row_order is not None:
        ct = ct.reindex(row_order, fill_value=0)
    if col_order is not None:
        ct = ct.reindex(columns=col_order, fill_value=0)
    return "```\n" + ct.to_string() + "\n```"


def load_training_universe(include_other: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_parquet(
        cs.LABELS_PARQUET,
        columns=["problem_id", "row_index", "subject", "level", "problem_text"],
    )
    opsd_df, opsd_cols = cs.load_opsd_problems()
    joined, jinfo = cs.auto_join(opsd_df, labels)
    matched = joined["subject"].notna() & joined["level"].notna()
    if include_other:
        keep = matched
    else:
        keep = matched & (joined["subject"] != "Other")
    df = joined.loc[keep, ["opsd_index", "problem_id", "subject", "level"]].copy()
    df["level"] = df["level"].astype(int)
    df["unit"] = [unit_of(s, l) for s, l in zip(df["subject"], df["level"])]
    df = df.sort_values(["level", "subject", "opsd_index"]).reset_index(drop=True)
    info = {
        "opsd_columns": opsd_cols,
        "join_info": jinfo,
        "n_total_opsd": int(len(joined)),
        "n_matched": int(matched.sum()),
        "n_unmatched": int((~matched).sum()),
        "n_universe": int(len(df)),
        "include_other": bool(include_other),
    }
    return df, info


def compute_residual_centroids(include_other: bool):
    DAF, DAT, md, ninfo = pa.load_pooled(None)
    del DAF; gc.collect()
    if len(md) != 3025:
        raise RuntimeError(f"[GATE] expected pooled N=3025, got {len(md)}")
    smap, clusters = subject_cluster_map(include_other)
    if include_other:
        mask = np.ones(len(md), dtype=bool)
    else:
        mask = md["subject"].isin(sorted(smap)).to_numpy()
    md_sub = md.loc[mask].reset_index(drop=True).copy()
    DAT_sub = DAT[mask].astype(np.float32)
    unit_arr = (md_sub["subject"].astype(str) + "|L" + md_sub["level"].astype(int).astype(str)).to_numpy()
    cnt = Counter(unit_arr)
    units_all = sorted(cnt)
    units_keep = sorted([u for u, n in cnt.items() if n >= MIN_UNIT_N])
    if len(units_keep) < 4:
        raise RuntimeError(f"[GATE] units_keep<4: {len(units_keep)}")

    mu = DAT_sub.mean(axis=0, keepdims=True)
    DA_c = DAT_sub - mu
    DA_resid = ssg.level_centroid_residual(DA_c, md_sub)
    DAn_r = sa.normalize_members(DA_resid)
    M_res, cents_res = cd.unit_centroid_matrix(DAn_r, unit_arr, units_keep, W_SUBJ)

    # Pilot ARI k=4 on residual centroid pipeline, common units with n>=30 each pilot.
    pilot_col = "_pilot" if "_pilot" in md_sub.columns else "pilot"
    pilot_ari = float("nan")
    units_common: list[str] = []
    if pilot_col in md_sub.columns:
        p1 = md_sub[pilot_col].to_numpy() == "pilot1"
        p2 = md_sub[pilot_col].to_numpy() == "pilot2"
        cnt1 = Counter(unit_arr[p1]); cnt2 = Counter(unit_arr[p2])
        units_common = [u for u in units_keep if cnt1.get(u, 0) >= MIN_UNIT_N and cnt2.get(u, 0) >= MIN_UNIT_N]
        if len(units_common) >= 4:
            # recompute residual separately within each pilot to mirror residual-centroid pipeline.
            def _pilot_labels(pm):
                mdp = md_sub.loc[pm].reset_index(drop=True).copy()
                DATp = DAT_sub[pm].astype(np.float32)
                DAc = DATp - DATp.mean(axis=0, keepdims=True)
                DAr = ssg.level_centroid_residual(DAc, mdp)
                DNr = sa.normalize_members(DAr)
                uarr = (mdp["subject"].astype(str) + "|L" + mdp["level"].astype(int).astype(str)).to_numpy()
                M, _ = cd.unit_centroid_matrix(DNr, uarr, units_common, W_SUBJ)
                lab, _ = cd.ward_labels(M, 4)
                return lab
            pilot_ari = float(adjusted_rand_score(_pilot_labels(p1), _pilot_labels(p2)))

    # full residual labels k=4 (or k=3 excluding Other for reporting only)
    labels4, Z4 = cd.ward_labels(M_res, 4 if include_other else 3)
    info = {
        "ninfo": ninfo,
        "analysis_N_subset": int(len(md_sub)),
        "units_analysis_all": units_all,
        "units_keep": units_keep,
        "n_per_unit_keep": [int(cnt[u]) for u in units_keep],
        "pilot_ari_k4_resid": pilot_ari,
        "units_common_p1p2": units_common,
        "ward_labels_resid": labels4.astype(int).tolist(),
        "subject_clusters": clusters,
    }
    return units_keep, cents_res.astype(np.float32), M_res.astype(np.float32), info


def build_unit_vectors(train_units: list[str], units_keep: list[str], cents: np.ndarray, smap: dict[str, str]):
    keep_idx = {u: i for i, u in enumerate(units_keep)}
    train_set = sorted(train_units, key=lambda u: (lev_of(u), subj_of(u)))
    vectors: dict[str, np.ndarray] = {}
    nearest_info = []
    for u in train_set:
        if u in keep_idx:
            vectors[u] = cents[keep_idx[u]].copy()
            continue
        subj, lv = subj_of(u), lev_of(u)
        cl = smap.get(subj)
        # fallback preference: same subject-cluster and same level, then same cluster, then all.
        cand = [ku for ku in units_keep if smap.get(subj_of(ku)) == cl and lev_of(ku) == lv]
        method = "same_cluster_level_nearest"
        if not cand:
            cand = [ku for ku in units_keep if smap.get(subj_of(ku)) == cl]
            method = "same_cluster_nearest"
        if not cand:
            cand = units_keep[:]
            method = "global_nearest"
        # Approx nearest in residual centroid space; if no own vector, compare by same level/cluster candidates.
        # Pick candidate closest to average candidate set if multiple tied by metadata; deterministic.
        # For missing unit with no observed centroid, its proxy vector is the selected candidate vector.
        # If same subject+neighboring level exists, prefer closest level gap before cosine tie.
        cand = sorted(cand, key=lambda x: (abs(lev_of(x) - lv), x))
        chosen = cand[0]
        vectors[u] = cents[keep_idx[chosen]].copy()
        nearest_info.append({
            "unit": u, "assigned_centroid_unit": chosen, "method": method,
            "subject": subj, "level": lv,
            "assigned_subject": subj_of(chosen), "assigned_level": lev_of(chosen),
        })
    return train_set, vectors, nearest_info


def dist(u: str, v: str, vectors: dict[str, np.ndarray]) -> float:
    return float(1.0 - np.dot(vectors[u], vectors[v]))


def order_units_c2(units: list[str], vectors: dict[str, np.ndarray], counts: dict[str, int], smap: dict[str, str], r: int):
    remaining = set(units)
    ordered: list[str] = []
    relax_events = 0
    while remaining:
        if not ordered:
            cur = min(remaining, key=lambda u: (lev_of(u), subj_of(u), u))
            ordered.append(cur); remaining.remove(cur)
        cur = ordered[-1]
        cur_cl = smap[subj_of(cur)]
        # stay within cluster up to r consecutive including current cluster tail.
        tail_same = 0
        for x in reversed(ordered):
            if smap[subj_of(x)] == cur_cl:
                tail_same += 1
            else:
                break
        placed = False
        if tail_same < r:
            cand = [u for u in remaining if smap[subj_of(u)] == cur_cl and lev_of(u) >= lev_of(cur)]
            if not cand:
                cand = [u for u in remaining if smap[subj_of(u)] == cur_cl]
            if cand:
                nxt = min(cand, key=lambda u: (max(0, lev_of(cur)-lev_of(u)), dist(cur, u, vectors), lev_of(u), u))
                ordered.append(nxt); remaining.remove(nxt); placed = True
        if placed:
            continue
        cand = [u for u in remaining if smap[subj_of(u)] != cur_cl and lev_of(u) >= lev_of(cur) - 1]
        if not cand:
            cand = [u for u in remaining if smap[subj_of(u)] != cur_cl]
            relax_events += 1
        if not cand:
            cand = list(remaining)
            relax_events += 1
        nxt = min(cand, key=lambda u: (dist(cur, u, vectors), max(0, lev_of(cur)-lev_of(u)), lev_of(u), u))
        ordered.append(nxt); remaining.remove(nxt)
    return ordered, relax_events


def split_ordered_units_by_mass(unit_order: list[str], counts: dict[str, int], n_stages: int) -> dict[str, int]:
    """Assign whole units to contiguous stages while approximating equal problem mass.

    Units remain atomic, so exact equal mass is not always possible.  The previous
    target-crossing loop could miss a boundary forever when the pre-crossing
    cumulative mass was closer to the first target than the post-crossing mass;
    that collapsed most units into the last stage.  Here each target boundary is
    chosen independently as the unused cut whose cumulative mass is closest to
    that target.
    """
    total = int(sum(counts[u] for u in unit_order))
    if len(unit_order) < n_stages:
        raise AssertionError(f"cannot split {len(unit_order)} units into {n_stages} stages")
    cum_by_cut = np.cumsum([counts[u] for u in unit_order[:-1]], dtype=np.float64)
    cuts: list[int] = []
    for k in range(1, n_stages):
        target = total * k / n_stages
        candidates = [c for c in range(1, len(unit_order)) if c not in cuts]
        if not candidates:
            raise AssertionError("no remaining split candidates")
        best = min(candidates, key=lambda c: (abs(cum_by_cut[c - 1] - target), c))
        cuts.append(int(best))
    cuts = sorted(cuts)
    if len(set(cuts)) != n_stages - 1:
        raise AssertionError(f"non-unique cuts: {cuts}")
    u2stage: dict[str, int] = {}
    start = 0
    for s, end in enumerate(cuts + [len(unit_order)]):
        if end <= start:
            raise AssertionError(f"empty unit slice at stage={s}, cuts={cuts}")
        for u in unit_order[start:end]:
            u2stage[u] = s
        start = end
    return u2stage


def equal_mass_split_indices(n: int, n_stages: int) -> list[np.ndarray]:
    idx = np.arange(n)
    return [x.astype(int) for x in np.array_split(idx, n_stages)]


def problem_level_stage_by_difficulty(df: pd.DataFrame, n_stages: int) -> np.ndarray:
    """Cond2: level ascending, equal-mass contiguous problem bands."""
    order = df.sort_values(["level", "subject", "opsd_index"]).index.to_numpy()
    stage = np.empty(len(df), dtype=int)
    for s, part in enumerate(np.array_split(order, n_stages)):
        stage[part] = s
    return stage


def stage_mean_levels(df: pd.DataFrame, stage_idx: np.ndarray, n_stages: int) -> np.ndarray:
    tmp = df.copy()
    tmp["stage"] = stage_idx
    return tmp.groupby("stage")["level"].mean().reindex(range(n_stages)).to_numpy(dtype=np.float64)


def mean_level_gate(means: np.ndarray) -> tuple[bool, list[float], int, float]:
    """Gate for mostly non-decreasing mean difficulty.

    Allows at most one small dip.  A dip below -0.25 mean level is treated as severe.
    """
    diffs = np.diff(means).astype(float)
    dip_mask = diffs < -0.10
    n_dips = int(dip_mask.sum())
    min_diff = float(diffs.min()) if len(diffs) else 0.0
    ok = (n_dips <= 1) and (min_diff >= -0.25)
    return ok, [float(x) for x in diffs], n_dips, min_diff


def order_units_nearest_path(
    units: list[str],
    vectors: dict[str, np.ndarray],
    start_after: str | None = None,
    start_unit: str | None = None,
) -> list[str]:
    """Greedy residual nearest-neighbor path over a fixed tier."""
    remaining = set(units)
    if not remaining:
        return []
    if start_unit is not None:
        if start_unit not in remaining:
            raise AssertionError(f"start_unit not in tier: {start_unit}")
        cur = start_unit
    elif start_after is None:
        cur = min(remaining, key=lambda u: (lev_of(u), subj_of(u), u))
    else:
        cur = min(remaining, key=lambda u: (dist(start_after, u, vectors), lev_of(u), subj_of(u), u))
    ordered = [cur]
    remaining.remove(cur)
    while remaining:
        cur = ordered[-1]
        nxt = min(remaining, key=lambda u: (dist(cur, u, vectors), lev_of(u), subj_of(u), u))
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def choose_tier_start(tus: list[str], vectors: dict[str, np.ndarray], prev_tail: str | None, mode: str) -> str | None:
    if mode == "default":
        return None
    if mode == "low":
        return min(tus, key=lambda u: (lev_of(u), dist(prev_tail, u, vectors) if prev_tail else 0.0, subj_of(u), u))
    if mode == "high":
        return max(tus, key=lambda u: (lev_of(u), -dist(prev_tail, u, vectors) if prev_tail else 0.0, subj_of(u), u))
    if mode == "nearest":
        if prev_tail is None:
            return None
        return min(tus, key=lambda u: (dist(prev_tail, u, vectors), lev_of(u), subj_of(u), u))
    raise ValueError(f"unknown tier start mode: {mode}")


def order_units_c2_tiered(
    units: list[str],
    vectors: dict[str, np.ndarray],
    counts: dict[str, int],
    n_tiers: int,
    start_units: dict[int, str] | None = None,
    start_modes: dict[int, str] | None = None,
):
    """Difficulty backbone + subject/residual perturbation.

    1) Split units sorted by level into equal-mass difficulty tiers.
    2) Within each tier only, order units by residual nearest-neighbor path.
    3) Stitch tiers low->high.  Optional start variants are used only to choose
       the first unit of a tier; the within-tier path is still residual-nearest.
    """
    start_units = start_units or {}
    start_modes = start_modes or {}
    difficulty_order = sorted(units, key=lambda u: (lev_of(u), subj_of(u), u))
    u2tier = split_ordered_units_by_mass(difficulty_order, counts, n_tiers)
    tier_units = [[u for u in difficulty_order if u2tier[u] == t] for t in range(n_tiers)]
    ordered: list[str] = []
    tier_info = []
    prev_tail = None
    for t, tus in enumerate(tier_units):
        start_unit = start_units.get(t)
        start_mode = start_modes.get(t, "default")
        if start_unit is None:
            start_unit = choose_tier_start(tus, vectors, prev_tail, start_mode)
        part = order_units_nearest_path(tus, vectors, prev_tail, start_unit=start_unit)
        if not part:
            raise AssertionError(f"empty tier {t}")
        ordered.extend(part)
        prev_tail = part[-1]
        tier_info.append({
            "tier": int(t),
            "n_units": int(len(part)),
            "n_problems": int(sum(counts[u] for u in part)),
            "min_level": int(min(lev_of(u) for u in part)),
            "max_level": int(max(lev_of(u) for u in part)),
            "start_mode": start_mode,
            "first_unit": part[0],
            "last_unit": part[-1],
            "units": part,
        })
    return ordered, u2tier, tier_info


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def tier_start_variants(train_units: list[str], counts: dict[str, int], n_tiers: int, max_first_tier_starts: int = 100):
    """Yield start variants.  Exhaustive for n_tiers=2, default elsewhere.

    The extra n_tiers=2 variants are still within the requested construction: tier
    ordering is fixed low->high, and only the within-tier residual path start is
    changed to tune ②↔③ similarity without violating the difficulty backbone.
    """
    yield {}, {}, "default"
    if n_tiers != 2:
        return
    difficulty_order = sorted(train_units, key=lambda u: (lev_of(u), subj_of(u), u))
    u2tier = split_ordered_units_by_mass(difficulty_order, counts, n_tiers)
    tier0 = [u for u in difficulty_order if u2tier[u] == 0]
    # Try all starts when small; otherwise deterministic coverage across levels/subjects.
    starts0 = tier0[:max_first_tier_starts]
    for s0 in starts0:
        for mode1 in ["nearest", "low", "high"]:
            yield {0: s0}, {1: mode1}, f"t0={s0};t1={mode1}"


def select_tiered_c2(df: pd.DataFrame, train_units: list[str], vectors: dict[str, np.ndarray], counts: dict[str, int], args):
    """Try n_tiers/start variants and select a configuration satisfying gates."""
    n_stages = int(args.n_stages)
    stage2 = problem_level_stage_by_difficulty(df, n_stages)
    candidates = sorted(set(parse_int_list(args.tier_candidates)))
    if int(args.n_tiers) not in candidates:
        candidates.append(int(args.n_tiers))
        candidates = sorted(set(candidates))
    records = []
    built: dict[tuple[int, str], tuple[list[str], dict[str, int], dict[str, int], list[dict[str, Any]], dict[str, Any]]] = {}
    for nt in candidates:
        if nt < 1 or nt > len(train_units):
            continue
        for start_units, start_modes, variant_name in tier_start_variants(train_units, counts, nt):
            unit_order, u2tier, tier_info = order_units_c2_tiered(train_units, vectors, counts, nt, start_units=start_units, start_modes=start_modes)
            u2stage = split_ordered_units_by_mass(unit_order, counts, n_stages)
            stage3 = df["unit"].map(u2stage).to_numpy(dtype=int)
            rho, p = spearmanr(stage2, stage3)
            means = stage_mean_levels(df, stage3, n_stages)
            mean_ok, diffs, n_dips, min_diff = mean_level_gate(means)
            rho_f = float(rho)
            rho_ok = 0.4 <= rho_f <= 0.7
            pass_gate = bool(rho_ok and mean_ok)
            rho_pen = 0.0 if rho_ok else (0.4 - rho_f if rho_f < 0.4 else rho_f - 0.7)
            mean_pen = 0.0 if mean_ok else (max(0, n_dips - 1) + max(0.0, -0.25 - min_diff) * 10.0)
            penalty = rho_pen * 10.0 + mean_pen * 10.0 + abs(rho_f - 0.55) + 0.01 * abs(nt - int(args.n_tiers))
            rec = {
                "n_tiers": int(nt),
                "variant": variant_name,
                "rho": rho_f,
                "p": float(p),
                "stage_mean_levels": [float(x) for x in means],
                "stage_mean_level_diffs": diffs,
                "n_dips": int(n_dips),
                "min_diff": float(min_diff),
                "rho_gate_ok": bool(rho_ok),
                "mean_level_gate_ok": bool(mean_ok),
                "gate_pass": pass_gate,
                "penalty": float(penalty),
            }
            records.append(rec)
            built[(nt, variant_name)] = (unit_order, u2stage, u2tier, tier_info, rec)
    passes = [r for r in records if r["gate_pass"]]
    if passes:
        chosen_rec = min(passes, key=lambda r: (abs(r["rho"] - 0.55), abs(r["n_tiers"] - int(args.n_tiers)), r["n_tiers"], r["variant"]))
        unit_order, u2stage, u2tier, tier_info, rec = built[(chosen_rec["n_tiers"], chosen_rec["variant"])]
        return unit_order, u2stage, u2tier, tier_info, rec, records
    if not records:
        raise RuntimeError("[GATE] no valid n_tiers candidates were evaluated")
    # Keep the failure explicit; never silently emit stages if the decision gate fails.
    preview = sorted(records, key=lambda r: (r["penalty"], abs(r["rho"] - 0.55)))[:20]
    raise RuntimeError("[GATE] no tiered C2 candidate satisfied rho(+0.4..+0.7) and mean-level gates. Best candidates:\n" + json.dumps(preview, indent=2, ensure_ascii=False))


def make_stage_json(spec: str, df: pd.DataFrame, stage_idx: np.ndarray, params: dict[str, Any], out_path: Path):
    n_stages = int(params["n_stages"])
    stages = []
    for s in range(n_stages):
        sub = df.loc[stage_idx == s, ["problem_id", "opsd_index", "unit", "subject", "level"]].copy()
        sub = sub.sort_values(["level", "subject", "opsd_index"])
        items = [
            {"problem_id": str(r.problem_id), "opsd_index": int(r.opsd_index),
             "unit": str(r.unit), "subject": str(r.subject), "level": int(r.level)}
            for r in sub.itertuples(index=False)
        ]
        stages.append({
            "stage_index": int(s),
            "order_index": int(s),
            "n": int(len(sub)),
            "problem_ids": [it["problem_id"] for it in items],
            "opsd_indices": [it["opsd_index"] for it in items],
            "items": items,
        })
    obj = {"spec": spec, **params, "stages": stages}
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def assert_partition(name: str, df: pd.DataFrame, stage_idx: np.ndarray, n_stages: int, ref_ids: set[str]):
    if len(stage_idx) != len(df):
        raise AssertionError(f"{name}: stage_idx length mismatch")
    if set(np.unique(stage_idx).tolist()) != set(range(n_stages)):
        raise AssertionError(f"{name}: stages not exactly 0..{n_stages-1}: {sorted(np.unique(stage_idx))}")
    ids = df["problem_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{name}: duplicate problem_id in universe")
    if set(ids) != ref_ids:
        raise AssertionError(f"{name}: problem set differs")
    sizes = np.array([int((stage_idx == s).sum()) for s in range(n_stages)], dtype=int)
    for s, sz in enumerate(sizes):
        if sz == 0:
            raise AssertionError(f"{name}: empty stage {s}")
    # Hard guard against silent unit-cut collapse.  Cond3/4/5 can be imperfect
    # because units are atomic, but no stage should swallow most of the universe.
    target = math.ceil(len(df) / n_stages)
    if sizes.max() > 2 * target:
        raise AssertionError(f"{name}: stage mass collapse sizes={sizes.tolist()} target={target}")


def build_conditions(df: pd.DataFrame, unit_order: list[str], u2stage: dict[str, int], vectors: dict[str, np.ndarray], args, smap):
    n = len(df); n_stages = args.n_stages
    ref_ids = set(df["problem_id"].astype(str))
    outputs = {}
    stage3 = df["unit"].map(u2stage).to_numpy(dtype=int)
    assert_partition("cond3", df, stage3, n_stages, ref_ids)
    outputs["cond3"] = stage3

    stage2 = problem_level_stage_by_difficulty(df, n_stages)
    assert_partition("cond2", df, stage2, n_stages, ref_ids)
    outputs["cond2"] = stage2

    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # cond1 random equal-mass split
        perm = rng.permutation(n)
        st = np.empty(n, dtype=int)
        for s, part in enumerate(np.array_split(perm, n_stages)):
            st[part] = s
        assert_partition(f"cond1_seed{seed}", df, st, n_stages, ref_ids)
        outputs[f"cond1_seed{seed}"] = st

        # cond4 stage order shuffle (membership from cond3, labels permuted)
        while True:
            p = rng.permutation(n_stages)
            if not np.array_equal(p, np.arange(n_stages)) and not np.array_equal(p, np.arange(n_stages)[::-1]):
                break
        st4 = np.array([int(np.where(p == x)[0][0]) for x in stage3], dtype=int)
        # This means new stage 0 contains old stage p[0], etc.
        assert_partition(f"cond4_seed{seed}", df, st4, n_stages, ref_ids)
        outputs[f"cond4_seed{seed}"] = st4

        # cond5 level-distribution matched to cond3, random within each level.
        st5 = np.full(n, -1, dtype=int)
        for lv in sorted(df["level"].unique()):
            level_idx = df.index[df["level"] == lv].to_numpy()
            rng.shuffle(level_idx)
            pos = 0
            for s in range(n_stages):
                need = int(((stage3 == s) & (df["level"].to_numpy() == lv)).sum())
                if need:
                    st5[level_idx[pos:pos+need]] = s
                    pos += need
            if pos != len(level_idx):
                raise AssertionError(f"cond5 seed={seed} lv={lv}: allocation mismatch")
        assert (st5 >= 0).all()
        assert_partition(f"cond5_seed{seed}", df, st5, n_stages, ref_ids)
        outputs[f"cond5_seed{seed}"] = st5
    return outputs


def stage_centroid_distances(df: pd.DataFrame, stage_idx: np.ndarray, vectors: dict[str, np.ndarray], counts: dict[str, int]):
    cents = []
    for s in sorted(np.unique(stage_idx)):
        units = sorted(df.loc[stage_idx == s, "unit"].unique())
        w = np.array([counts[u] for u in units], dtype=np.float64)
        V = np.stack([vectors[u] for u in units])
        c = (V * w[:, None]).sum(axis=0) / max(w.sum(), 1.0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        cents.append(c)
    return np.array([1.0 - float(np.dot(cents[i], cents[i+1])) for i in range(len(cents)-1)], dtype=np.float32)


def random_stage_distance_mean(df, vectors, counts, n_stages, n_rep=200, seed=123):
    units = sorted(df["unit"].unique())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        order = rng.permutation(units).tolist()
        u2s = split_ordered_units_by_mass(order, counts, n_stages)
        st = df["unit"].map(u2s).to_numpy(dtype=int)
        vals.append(stage_centroid_distances(df, st, vectors, counts).mean())
    return float(np.mean(vals)), float(np.std(vals))


def write_report(out_dir: Path, df: pd.DataFrame, outputs: dict[str, np.ndarray], unit_order, counts, nearest_info, params, ari, rand_mean, rand_std, vectors):
    today = date.today().isoformat()
    L = []
    L.append(f"# OPSD stage build — {today}")
    L.append("")
    L.append("CPU-only pure stage construction. Existing training/OPSD/extraction/prior npz artifacts were not modified.")
    L.append("")
    L.append("## 0. Parameters / universe")
    for k, v in params.items():
        L.append(f"- {k}: `{v}`")
    L.append(f"- N problems: **{len(df):,}**")
    L.append(f"- units: **{df['unit'].nunique()}**")
    L.append(f"- residual cluster pilot ARI(k=4): **{ari:+.3f}**")
    L.append("")

    rho, p = spearmanr(outputs["cond2"], outputs["cond3"])
    means3 = stage_mean_levels(df, outputs["cond3"], params["n_stages"])
    mean_ok, mean_diffs, n_dips, min_diff = mean_level_gate(means3)
    L.append("## 1. Decision gate: tiered ③ C2")
    L.append(f"- selected n_tiers = **{params.get('selected_n_tiers')}**")
    L.append(f"- Spearman ρ(stage_cond2, stage_cond3) = **{float(rho):+.4f}** (target +0.4~+0.7; p={float(p):.3g})")
    L.append(f"- ③ stage mean levels = **{[round(float(x), 3) for x in means3]}**")
    L.append(f"- mean-level diffs = `{[round(float(x), 3) for x in mean_diffs]}`; dips={n_dips}; min_diff={min_diff:+.3f}")
    L.append(f"- gate_pass = **{bool(0.4 <= float(rho) <= 0.7 and mean_ok)}**")
    L.append("- Interpretation: ③ should preserve an ascending difficulty trend while remaining clearly distinct from pure ② difficulty order.")
    L.append("")

    for name in ["cond3", "cond2"]:
        label = "③ ours_C2" if name == "cond3" else "② difficulty"
        tmp = df.copy(); tmp["stage"] = outputs[name]
        L.append(f"## 2. {label}: stage×subject / stage×level")
        L.append("### stage × subject")
        L.append(md_table_from_counts(tmp, "stage", "subject", row_order=list(range(params["n_stages"]))))
        L.append("### stage × level")
        L.append(md_table_from_counts(tmp, "stage", "level", row_order=list(range(params["n_stages"])), col_order=sorted(df["level"].unique())))
        means = tmp.groupby("stage")["level"].mean().reindex(range(params["n_stages"]))
        L.append("### stage mean level")
        L.append("| stage | mean_level | n |")
        L.append("|---|---:|---:|")
        for s in range(params["n_stages"]):
            L.append(f"| {s} | {means.loc[s]:.3f} | {int((outputs[name] == s).sum())} |")
        L.append("")

    dseq = stage_centroid_distances(df, outputs["cond3"], vectors, counts)
    L.append("## 3. ③ residual stage-to-stage distance")
    L.append(f"- consecutive stage residual distances: `{[round(float(x), 4) for x in dseq]}`")
    L.append(f"- mean consecutive distance = **{float(dseq.mean()):.4f}**")
    L.append(f"- random unit-order mean±sd over 200 reps = **{rand_mean:.4f} ± {rand_std:.4f}**")
    L.append("")

    tmp = df.copy(); tmp["stage"] = outputs["cond3"]
    L.append("## 4. ③ stage-level distribution table (⑤ fixed spec)")
    L.append(md_table_from_counts(tmp, "stage", "level", row_order=list(range(params["n_stages"])), col_order=sorted(df["level"].unique())))
    L.append("")

    L.append("## 5. Missing/no-centroid unit assignment")
    if nearest_info:
        L.append("| unit | assigned_centroid_unit | method |")
        L.append("|---|---|---|")
        for x in nearest_info:
            L.append(f"| {x['unit']} | {x['assigned_centroid_unit']} | {x['method']} |")
    else:
        L.append("- None: every training unit had n≥30 residual centroid in analysis subset.")
    L.append("")

    L.append("## 6. Output files")
    for pth in sorted(out_dir.glob("stages_cond*.json")):
        L.append(f"- `{pth.name}`")
    L.append("- `manifest.json`")
    L.append("- `stagebuild_artifacts.npz`")
    L.append("")
    (out_dir / f"REPORT_stagebuild_{today}.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-other", action="store_true", default=False)
    ap.add_argument("--n-stages", type=int, default=5)
    ap.add_argument("--subject-run-r", type=int, default=2, help="deprecated; kept for compatibility")
    ap.add_argument("--n-tiers", type=int, default=3)
    ap.add_argument("--tier-candidates", default="2,3,4,5,6,7,8")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-dir", default=str(TRAIN_STAGES))
    args = ap.parse_args()
    t0 = time.time()
    if args.n_stages != 5:
        raise RuntimeError("[GATE] requested n_stages must be 5 for this build")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Do not overwrite prior/new artifacts silently.
    expected = [
        out_dir / "stages_cond2_diff.json",
        out_dir / "stages_cond3_ours_C2.json",
        out_dir / "manifest.json",
        out_dir / "stagebuild_artifacts.npz",
        out_dir / f"REPORT_stagebuild_{date.today().isoformat()}.md",
    ]
    expected += [out_dir / f"stages_cond1_random_seed{s}.json" for s in parse_seeds(args.seeds)]
    expected += [out_dir / f"stages_cond4_shuffle_seed{s}.json" for s in parse_seeds(args.seeds)]
    expected += [out_dir / f"stages_cond5_diffmatched_seed{s}.json" for s in parse_seeds(args.seeds)]
    existing = [str(p) for p in expected if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs:\n" + "\n".join(existing))

    print("[1/6] load training universe", flush=True)
    df, train_info = load_training_universe(args.include_other)
    smap, clusters = subject_cluster_map(args.include_other)
    df["subject_cluster"] = df["subject"].map(smap)
    if df["subject_cluster"].isna().any():
        raise RuntimeError("subject cluster mapping failed")
    counts = df.groupby("unit").size().astype(int).to_dict()

    print("[2/6] compute residual centroids", flush=True)
    units_keep, cents, M_res, cinfo = compute_residual_centroids(args.include_other)

    print("[3/6] proxy missing units and select tiered C2", flush=True)
    train_units, vectors, nearest_info = build_unit_vectors(sorted(counts), units_keep, cents, smap)
    unit_order, u2stage, u2tier, tier_info, selected_tier_record, tier_gate_records = select_tiered_c2(df, train_units, vectors, counts, args)
    relax_events = 0

    print("[4/6] build conditions", flush=True)
    outputs = build_conditions(df, unit_order, u2stage, vectors, args, smap)

    params = {
        "include_other": bool(args.include_other),
        "n_stages": int(args.n_stages),
        "construction": "tiered_difficulty_backbone_residual_within_tier",
        "n_tiers_default": int(args.n_tiers),
        "tier_candidates": parse_int_list(args.tier_candidates),
        "selected_n_tiers": int(selected_tier_record["n_tiers"]),
        "selected_rho_cond2_cond3": float(selected_tier_record["rho"]),
        "selected_stage_mean_levels": selected_tier_record["stage_mean_levels"],
        "selected_stage_mean_level_diffs": selected_tier_record["stage_mean_level_diffs"],
        "tier_gate_target_rho": "+0.4..+0.7",
        "tier_gate_record": selected_tier_record,
        "seeds": parse_seeds(args.seeds),
        "W_SUBJ": W_SUBJ,
        "MIN_UNIT_N": MIN_UNIT_N,
        "clusters": clusters,
        "relax_events": int(relax_events),
        "elapsed_start_date": date.today().isoformat(),
    }
    common_json_params = {k: v for k, v in params.items() if k not in ["elapsed_start_date"]}

    print("[5/6] write JSON outputs", flush=True)
    make_stage_json("stages_cond2_diff", df, outputs["cond2"], common_json_params, out_dir / "stages_cond2_diff.json")
    make_stage_json("stages_cond3_ours_C2", df, outputs["cond3"], common_json_params | {"unit_order": unit_order, "unit_to_stage": u2stage, "unit_to_tier": u2tier, "tier_info": tier_info}, out_dir / "stages_cond3_ours_C2.json")
    for seed in parse_seeds(args.seeds):
        make_stage_json(f"stages_cond1_random_seed{seed}", df, outputs[f"cond1_seed{seed}"], common_json_params | {"seed": seed}, out_dir / f"stages_cond1_random_seed{seed}.json")
        make_stage_json(f"stages_cond4_shuffle_seed{seed}", df, outputs[f"cond4_seed{seed}"], common_json_params | {"seed": seed, "base": "cond3_membership_stage_order_shuffled"}, out_dir / f"stages_cond4_shuffle_seed{seed}.json")
        make_stage_json(f"stages_cond5_diffmatched_seed{seed}", df, outputs[f"cond5_seed{seed}"], common_json_params | {"seed": seed, "level_distribution_source": "cond3"}, out_dir / f"stages_cond5_diffmatched_seed{seed}.json")

    manifest = {
        "params": params,
        "training_info": train_info,
        "centroid_info": cinfo,
        "unit_counts_full": {k: int(v) for k, v in sorted(counts.items())},
        "unit_order_cond3": unit_order,
        "unit_to_stage_cond3": {k: int(v) for k, v in sorted(u2stage.items())},
        "unit_to_tier_cond3": {k: int(v) for k, v in sorted(u2tier.items())},
        "tier_info_cond3": tier_info,
        "tier_gate_records": tier_gate_records,
        "dropped_or_missing_unit_assignments": nearest_info,
        "condition_files": [p.name for p in sorted(out_dir.glob("stages_cond*.json"))],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("[6/6] diagnostics/report/artifacts", flush=True)
    rand_mean, rand_std = random_stage_distance_mean(df, vectors, counts, args.n_stages)
    rho, p = spearmanr(outputs["cond2"], outputs["cond3"])
    dseq = stage_centroid_distances(df, outputs["cond3"], vectors, counts)
    write_report(out_dir, df, outputs, unit_order, counts, nearest_info, params, cinfo["pilot_ari_k4_resid"], rand_mean, rand_std, vectors)

    np.savez(
        out_dir / "stagebuild_artifacts.npz",
        cond2_stage=outputs["cond2"].astype(np.int16),
        cond3_stage=outputs["cond3"].astype(np.int16),
        unit_order=np.array(unit_order),
        unit_order_counts=np.array([counts[u] for u in unit_order], dtype=np.int32),
        cond3_stage_distances=dseq.astype(np.float32),
        random_stage_distance_mean=np.array([rand_mean], dtype=np.float32),
        random_stage_distance_std=np.array([rand_std], dtype=np.float32),
        spearman_cond2_cond3=np.array([float(rho)], dtype=np.float32),
        spearman_cond2_cond3_p=np.array([float(p)], dtype=np.float32),
        pilot_ari_k4_resid=np.array([cinfo["pilot_ari_k4_resid"]], dtype=np.float32),
        residual_M_keep=M_res.astype(np.float32),
        units_keep=np.array(units_keep),
        train_units=np.array(train_units),
        selected_n_tiers=np.array([int(selected_tier_record["n_tiers"])], dtype=np.int16),
        selected_stage_mean_levels=np.array(selected_tier_record["stage_mean_levels"], dtype=np.float32),
        tier_gate_records_json=np.array([json.dumps(tier_gate_records, ensure_ascii=False)]),
        **{f"cond1_seed{s}_stage": outputs[f"cond1_seed{s}"].astype(np.int16) for s in parse_seeds(args.seeds)},
        **{f"cond4_seed{s}_stage": outputs[f"cond4_seed{s}"].astype(np.int16) for s in parse_seeds(args.seeds)},
        **{f"cond5_seed{s}_stage": outputs[f"cond5_seed{s}"].astype(np.int16) for s in parse_seeds(args.seeds)},
    )
    print(f"[done] wrote stagebuild outputs to {out_dir} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
