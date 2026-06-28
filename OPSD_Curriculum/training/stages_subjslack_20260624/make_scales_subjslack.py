#!/usr/bin/env python3
"""
make_scales_subjslack.py
========================
Build fair A/B subsample manifests for the NEW "ours" curriculum
(level_backbone_residual_subject_slack), scales: q4 / mini50 / mini100 / mini150.

Chain (identical to make_quarter + make_mini, unit-stratified):
  full (28,771)  --frac 0.25-->  q4 (~7,193, T~225)
  q4             --frac-->       mini50 (~1,603, T~50)
                                 mini100 (~3,198, T~100)
                                 mini150 (~4,800, T~150)

FAIRNESS (asserted): at every scale, `stratified_pick` draws ONE shared id set,
then both arms are filtered to it -> diff_ids == ours_ids == selected.
Difference between arms is stage assignment ONLY.

Reuses the exact helper logic of curriculum/make_mini_manifests.py.
Outputs to this folder: stages_cond2_diff_{scale}.json,
                        stages_cond3_ours_subjslack_{scale}.json
"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

SD = Path(__file__).resolve().parent
DIFF_FULL = SD / "stages_cond2_diff.json"
OURS_FULL = SD / "stages_cond3_ours_subjslack.json"
SEED = 42
B_GLOB = 32


def union_ids(m):
    s = set()
    for st in m["stages"]:
        s.update(str(x) for x in st["problem_ids"])
    return s


def build_id_to_unit(m):
    out = {}
    for st in m["stages"]:
        for it in st["items"]:
            out[str(it["problem_id"])] = it["unit"]
    return out


def stratified_pick(id_to_unit, frac, seed):
    by_unit = defaultdict(list)
    for pid, u in id_to_unit.items():
        by_unit[u].append(pid)
    rng = np.random.default_rng(seed)
    selected = set()
    for u in sorted(by_unit):
        ids = sorted(by_unit[u])
        n_u = len(ids)
        k_u = min(max(1, int(round(frac * n_u))) if n_u > 0 else 0, n_u)
        idx = rng.choice(n_u, size=k_u, replace=False) if k_u > 0 else np.array([], int)
        idx.sort()
        selected.update(ids[i] for i in idx)
    return selected


def filter_manifest(m, keep, base_spec):
    new_stages, total = [], 0
    for st in m["stages"]:
        keep_mask = [str(x) in keep for x in st["problem_ids"]]
        kept_ids = [str(x) for x, k in zip(st["problem_ids"], keep_mask) if k]
        kept_items = [it for it in st["items"] if str(it["problem_id"]) in keep]
        new_st = dict(st)
        new_st["problem_ids"] = kept_ids
        new_st["items"] = kept_items
        if "opsd_indices" in st:
            new_st["opsd_indices"] = [oi for oi, k in zip(st["opsd_indices"], keep_mask) if k]
            assert len(new_st["opsd_indices"]) == len(kept_ids)
        assert len(kept_ids) == len(kept_items)
        new_st["source_n"] = len(st["problem_ids"])
        new_st["n"] = len(kept_ids)
        new_stages.append(new_st)
        total += len(kept_ids)
    out = dict(m)
    out["stages"] = new_stages
    out["n_stages"] = len(new_stages)
    out["spec"] = base_spec
    return out, total


def subsample(diff, ours, frac, seed, scale):
    di, oi = union_ids(diff), union_ids(ours)
    assert di == oi, f"[{scale}] source universes differ: {len(di ^ oi)}"
    id2u = {p: u for p, u in build_id_to_unit(ours).items() if p in di}
    assert not (di - set(id2u)), f"[{scale}] missing unit info"
    selected = stratified_pick(id2u, frac, seed)
    ds, dt = filter_manifest(diff, selected, f"stages_cond2_diff_{scale}")
    os_, ot = filter_manifest(ours, selected, f"stages_cond3_ours_subjslack_{scale}")
    assert union_ids(ds) == union_ids(os_) == selected, f"[{scale}] arm universe mismatch"
    assert dt == ot == len(selected)
    for m in (ds, os_):
        m["subsample"] = {"scale": scale, "frac": frac, "seed": seed,
                          "sampled_n": len(selected), "strategy": "unit_stratified_round"}
    return ds, os_, selected


def write(m, name):
    (SD / name).write_text(json.dumps(m))
    sizes = [st["n"] for st in m["stages"]]
    print(f"  wrote {name:<46} N={sum(sizes)} T~{math.ceil(sum(sizes)/B_GLOB)} stages={sizes}")


def main():
    diff_full = json.loads(DIFF_FULL.read_text())
    ours_full = json.loads(OURS_FULL.read_text())
    print(f"full universe N={len(union_ids(diff_full))}")

    # full -> q4 (25%)
    dq, oq, selq = subsample(diff_full, ours_full, 0.25, SEED, "q4")
    write(dq, "stages_cond2_diff_q4.json")
    write(oq, "stages_cond3_ours_subjslack_q4.json")
    Nq4 = len(selq)
    print(f"q4 universe N={Nq4}\n")

    # q4 -> mini{50,100,150}  (frac chosen so T_total ~ {50,100,150})
    for scale, T in [("mini50", 50), ("mini100", 100), ("mini150", 150)]:
        frac = B_GLOB * T / Nq4
        dm, om, selm = subsample(dq, oq, frac, SEED, scale)
        write(dm, f"stages_cond2_diff_{scale}.json")
        write(om, f"stages_cond3_ours_subjslack_{scale}.json")
        print(f"  [{scale}] frac_of_q4={frac:.4f} sampled_n={len(selm)} "
              f"(subset of q4: {selm <= selq})\n")


if __name__ == "__main__":
    main()
