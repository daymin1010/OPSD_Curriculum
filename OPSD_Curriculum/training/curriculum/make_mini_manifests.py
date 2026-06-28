#!/usr/bin/env python3
"""
make_mini_manifests.py
======================
Build mini-scale manifests (`mini50`, `mini100`) for fair A/B
(cond2_diff vs cond3_ours_C2) — much smaller than the existing `q4`
manifests so total optimizer step count lands near the OPSD original
peak region (~50–150 step).

Why a new sampler (vs reusing make_quarter_manifests.py)
-------------------------------------------------------
The existing quarter sampler picks a random subset of the global
universe. For very small N the resulting stage sizes become noisy
and some `unit`s (subject|level cells) may drop out entirely.
This script instead does **unit-stratified sampling** within each
stage of the source `q4` manifests:

  - We start from the existing `*_q4.json` (universe = 7,193,
    `diff_set == ours_set`, fair).
  - For each `unit` present in the q4 universe, we sample exactly
    `round(frac * n_unit)` problem_ids (>=1 if n_unit > 0) using a
    seeded RNG. The picked ids form the common subset (same for both
    arms).
  - We then intersect each arm's stage problem lists with that common
    subset, **also filtering the `items` array** (fixes a latent bug in
    `make_quarter_manifests.py` that left q4 `items` unfiltered).

Result invariants (asserted at exit):
  - diff_set == ours_set                       (fair A/B)
  - sampled ids are a subset of q4 universe    (no leakage)
  - within each unit, |sampled| / |universe|  ≈ frac
  - per-stage `n` == len(problem_ids) == len(items)

Scales produced
---------------
  mini50  : frac_of_q4 ≈ 2/9  → N ≈ 1,598, T_total ≈ 50  (B_glob=32)
  mini100 : frac_of_q4 ≈ 4/9  → N ≈ 3,197, T_total ≈ 100

Outputs:
  stages_tiered_20260622/stages_cond2_diff_mini50.json
  stages_tiered_20260622/stages_cond3_ours_C2_mini50.json
  stages_tiered_20260622/stages_cond2_diff_mini100.json
  stages_tiered_20260622/stages_cond3_ours_C2_mini100.json

Usage
-----
  python make_mini_manifests.py            # writes both mini50 and mini100
  python make_mini_manifests.py --scale 50 # only mini50
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
STAGES_DIR = REPO / "training" / "stages_tiered_20260622"


def union_ids(manifest: dict) -> set:
    s = set()
    for st in manifest["stages"]:
        s.update(str(x) for x in st["problem_ids"])
    return s


def build_id_to_unit(manifest: dict) -> dict:
    """Map problem_id -> unit (e.g. 'Algebra|L4') using `items`."""
    out = {}
    for st in manifest["stages"]:
        for it in st["items"]:
            out[str(it["problem_id"])] = it["unit"]
    return out


def stratified_pick(id_to_unit: dict, frac: float, seed: int):
    """For each unit, draw round(frac * n_unit) ids (>=1 if n_unit>0)."""
    by_unit = defaultdict(list)
    for pid, u in id_to_unit.items():
        by_unit[u].append(pid)
    rng = np.random.default_rng(seed)
    selected = set()
    per_unit_report = []
    for u in sorted(by_unit.keys()):
        ids = sorted(by_unit[u])  # deterministic order
        n_u = len(ids)
        k_u = max(1, int(round(frac * n_u))) if n_u > 0 else 0
        k_u = min(k_u, n_u)
        # use rng.choice over indices to be deterministic with seed
        idx = rng.choice(n_u, size=k_u, replace=False) if k_u > 0 else np.array([], dtype=int)
        idx.sort()
        picked = [ids[i] for i in idx]
        selected.update(picked)
        per_unit_report.append((u, n_u, k_u))
    return selected, per_unit_report


def filter_manifest(manifest: dict, keep: set, suffix: str) -> tuple:
    new_stages = []
    total = 0
    for st in manifest["stages"]:
        # Keep order of original problem_ids; filter items the same way
        kept_ids = [str(x) for x in st["problem_ids"] if str(x) in keep]
        kept_items = [it for it in st["items"] if str(it["problem_id"]) in keep]
        kept_opsd = []
        if "opsd_indices" in st:
            # opsd_indices is aligned 1:1 with original problem_ids
            keep_mask = [str(x) in keep for x in st["problem_ids"]]
            kept_opsd = [oi for oi, m in zip(st["opsd_indices"], keep_mask) if m]
        # Sanity check alignment
        assert len(kept_ids) == len(kept_items), (
            f"stage {st['stage_index']}: problem_ids({len(kept_ids)}) != items({len(kept_items)})"
        )
        if kept_opsd:
            assert len(kept_opsd) == len(kept_ids), (
                f"stage {st['stage_index']}: opsd_indices({len(kept_opsd)}) != ids({len(kept_ids)})"
            )
        new_st = dict(st)
        new_st["problem_ids"] = kept_ids
        new_st["items"] = kept_items
        if kept_opsd:
            new_st["opsd_indices"] = kept_opsd
        new_st["n"] = len(kept_ids)
        new_st["source_n_q4"] = len(st["problem_ids"])
        new_stages.append(new_st)
        total += len(kept_ids)
    out = dict(manifest)
    out["stages"] = new_stages
    out["n_stages"] = len(new_stages)
    out["spec"] = (manifest.get("spec", "") + f"_{suffix}").lstrip("_")
    return out, total


def build_scale(scale: str, frac: float, seed: int, B_glob: int = 32) -> None:
    """scale in {'mini50','mini100'}. frac is fraction of q4 universe."""
    diff_src = STAGES_DIR / "stages_cond2_diff_q4.json"
    ours_src = STAGES_DIR / "stages_cond3_ours_C2_q4.json"
    diff_dst = STAGES_DIR / f"stages_cond2_diff_{scale}.json"
    ours_dst = STAGES_DIR / f"stages_cond3_ours_C2_{scale}.json"

    diff = json.loads(diff_src.read_text())
    ours = json.loads(ours_src.read_text())

    diff_ids = union_ids(diff)
    ours_ids = union_ids(ours)
    if diff_ids != ours_ids:
        only_d = len(diff_ids - ours_ids)
        only_o = len(ours_ids - diff_ids)
        raise SystemExit(
            f"[{scale}] q4 universes differ: diff_only={only_d}, ours_only={only_o}"
        )

    # Build id->unit map from the (richer) ours manifest items;
    # diff_items should agree on those problem_ids it shares (it does, fair set).
    id_to_unit = build_id_to_unit(ours)
    # All ids in the shared universe must have a unit; q4 items WAS unfiltered
    # so map covers full original universe — restrict to q4 universe just to be safe.
    id_to_unit = {pid: u for pid, u in id_to_unit.items() if pid in diff_ids}
    missing_unit = diff_ids - set(id_to_unit.keys())
    if missing_unit:
        raise SystemExit(
            f"[{scale}] {len(missing_unit)} q4 ids missing unit info; first 3: {list(missing_unit)[:3]}"
        )

    N_universe = len(id_to_unit)
    print(f"[{scale}] q4 universe N={N_universe}, frac={frac:.4f}, seed={seed}")

    selected, per_unit = stratified_pick(id_to_unit, frac=frac, seed=seed)
    print(f"[{scale}] selected N={len(selected)} (target~{int(round(frac*N_universe))}); "
          f"|units|={len(per_unit)}")
    # Per-unit stats summary
    n_unit_kept = sum(1 for _, _, k in per_unit if k > 0)
    n_unit_zero = sum(1 for _, _, k in per_unit if k == 0)
    sizes = [k for _, _, k in per_unit if k > 0]
    print(f"[{scale}] units kept={n_unit_kept}, dropped={n_unit_zero}, "
          f"per-unit n: min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}")

    diff_q, diff_total = filter_manifest(diff, selected, scale)
    ours_q, ours_total = filter_manifest(ours, selected, scale)
    # Fairness checks
    diff_q_ids = union_ids(diff_q)
    ours_q_ids = union_ids(ours_q)
    assert diff_q_ids == selected, f"diff_{scale} missing {len(selected - diff_q_ids)}, extra {len(diff_q_ids - selected)}"
    assert ours_q_ids == selected, f"ours_{scale} missing {len(selected - ours_q_ids)}, extra {len(ours_q_ids - selected)}"
    assert diff_q_ids == ours_q_ids, "diff and ours universes differ"
    assert diff_total == ours_total == len(selected)

    # metadata
    for m, src in ((diff_q, diff_src), (ours_q, ours_src)):
        m["subsample_frac_of_q4"] = frac
        m["subsample_seed"] = seed
        m["subsample_strategy"] = "unit_stratified_round"
        m["source_universe_n_q4"] = N_universe
        m["sampled_n"] = len(selected)
        m["source_manifest_q4"] = str(src)
        m["scale"] = scale

    diff_dst.write_text(json.dumps(diff_q, indent=2))
    ours_dst.write_text(json.dumps(ours_q, indent=2))

    print(f"[{scale}] wrote {diff_dst.name}: N={diff_total}, T_total~{math.ceil(diff_total/B_glob)}")
    for st in diff_q["stages"]:
        print(f"    diff stage{st['stage_index']} order={st['order_index']}: "
              f"q4_n={st['source_n_q4']} -> n={st['n']}  T_stage~{math.ceil(st['n']/B_glob)}")
    print(f"[{scale}] wrote {ours_dst.name}: N={ours_total}, T_total~{math.ceil(ours_total/B_glob)}")
    for st in ours_q["stages"]:
        print(f"    ours stage{st['stage_index']} order={st['order_index']}: "
              f"q4_n={st['source_n_q4']} -> n={st['n']}  T_stage~{math.ceil(st['n']/B_glob)}")
    print(f"[{scale}] OK: identical universe of {len(diff_q_ids)} problems across arms\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["50", "100", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B_glob", type=int, default=32)
    args = ap.parse_args()

    # Target T_total values: 50 and 100. q4 has N=7193, B=32 → T_q4=225.
    # frac to hit T_target ≈ B*T_target / N_q4
    N_Q4 = 7193
    cfgs = {
        "50": ("mini50", args.B_glob * 50 / N_Q4),   # ≈ 0.2225 → N~1600
        "100": ("mini100", args.B_glob * 100 / N_Q4),  # ≈ 0.4450 → N~3200
    }
    scales = ["50", "100"] if args.scale == "both" else [args.scale]
    for s in scales:
        name, frac = cfgs[s]
        build_scale(name, frac=frac, seed=args.seed, B_glob=args.B_glob)


if __name__ == "__main__":
    main()
