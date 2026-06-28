#!/usr/bin/env python3
"""
opsd_data.py
============
Data-LOADING layer ONLY for the OPSD curriculum stack.

WHY THIS EXISTS
---------------
The cached HF dataset `siyanzhao/Openthoughts_math_30k_opsd` was written by a
NEWER `datasets` version: its `dataset_info.json` encodes a feature with type
`'List'`, which the OPSD env's pinned `datasets==3.6.0` cannot parse:

    ValueError: Feature type 'List' not found.

So `load_dataset("siyanzhao/Openthoughts_math_30k_opsd")` hard-fails inside our
env, even though the underlying Arrow shards are perfectly valid.

This module bypasses the broken metadata by reading the Arrow shards directly
with pyarrow and wrapping them in a real `datasets.Dataset`. The returned object
is a genuine HF Dataset, so everything downstream (`ds["problem"]`,
`ds.column_names`, `ds.select(schedule)`, the OPSD collator) works unchanged.

CONTRACT (must not silently drift)
-----------------------------------
* Reads BOTH shards (00000-of-00002, 00001-of-00002) in lexical/name order so
  positional row index (`opsd_index`) matches what `load_dataset(...)["train"]`
  would have produced (shard 0 then shard 1).
* Phase-0 reference: full == 29,434 rows. A hard guard refuses to return a
  single-shard (23,719) dataset — training on shard-1-only is forbidden.
* Pure CPU. Deterministic. No network. No modification of upstream opsd_src.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
REPO_ROOT = Path("/scratch/lami2026/personal/jimin_2782")
DATASET_CACHE_GLOB = (
    "cache/huggingface/datasets/siyanzhao___openthoughts_math_30k_opsd/"
    "**/openthoughts_math_30k_opsd-train-*.arrow"
)
EXPECTED_TOTAL_ROWS = 29434          # Phase-0 reference (full Openthoughts 30k)
EXPECTED_NUM_SHARDS = 2              # 00000-of-00002 + 00001-of-00002
MIN_SAFE_ROWS = 29000               # below this we assume a missing shard


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _read_arrow_table(path: str) -> pa.Table:
    """Read a single .arrow shard, tolerating both IPC file and stream framing."""
    try:
        return ipc.open_file(path).read_all()
    except Exception:
        with pa.memory_map(path, "r") as src:
            return ipc.open_stream(src).read_all()


def find_shards(repo_root: Path | str = REPO_ROOT) -> list[str]:
    """Return the OPSD train Arrow shards in deterministic (name) order."""
    repo_root = Path(repo_root)
    pattern = str(repo_root / DATASET_CACHE_GLOB)
    shards = sorted(glob.glob(pattern, recursive=True))
    return shards


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def load_opsd_train(repo_root: Path | str = REPO_ROOT, strict: bool = True):
    """Load the OPSD train split as a real `datasets.Dataset`, bypassing the
    incompatible cached `dataset_info.json`.

    Parameters
    ----------
    repo_root : project root (defaults to the personal workspace).
    strict    : if True (default), enforce the full-dataset guard (29,434 rows,
                2 shards). Set False only for intentional debugging.

    Returns
    -------
    datasets.Dataset with the same columns/order as load_dataset(...)["train"].
    """
    from datasets import Dataset

    shards = find_shards(repo_root)
    if not shards:
        raise FileNotFoundError(
            f"[opsd_data] no Arrow shards under {Path(repo_root)/DATASET_CACHE_GLOB}"
        )

    tables = [_read_arrow_table(p) for p in shards]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    # The Arrow schema carries embedded HF metadata (info.features) written by a
    # newer `datasets`; it encodes a 'List' feature that datasets==3.6.0 cannot
    # parse. Stripping the schema metadata forces datasets to RE-INFER features
    # from the raw pyarrow column types (string/int/etc.), which is exactly what
    # we want — the actual data is fine, only the metadata is incompatible.
    table = table.replace_schema_metadata(None)
    n_rows = table.num_rows


    msg = (
        f"[opsd_data] shards={len(shards)} rows={n_rows} "
        f"(expected {EXPECTED_NUM_SHARDS} shards / {EXPECTED_TOTAL_ROWS} rows)"
    )
    print(msg, flush=True)
    for p, t in zip(shards, tables):
        print(f"[opsd_data]   {os.path.basename(p)}: {t.num_rows} rows", flush=True)

    if strict:
        if len(shards) < EXPECTED_NUM_SHARDS:
            raise RuntimeError(
                f"[opsd_data] only {len(shards)} shard(s) found; refusing to train "
                f"on a partial dataset (shard-1-only is forbidden). {msg}"
            )
        if n_rows < MIN_SAFE_ROWS:
            raise RuntimeError(
                f"[opsd_data] only {n_rows} rows (< {MIN_SAFE_ROWS}); a shard is "
                f"likely missing. Refusing partial-dataset training. {msg}"
            )
        if n_rows != EXPECTED_TOTAL_ROWS:
            # Not fatal (dataset could legitimately be revised) but loud.
            print(
                f"[opsd_data] WARNING: row count {n_rows} != Phase-0 reference "
                f"{EXPECTED_TOTAL_ROWS}. Verify before trusting cell counts.",
                flush=True,
            )

    ds = Dataset(table)
    return ds


if __name__ == "__main__":
    ds = load_opsd_train()
    print(f"[opsd_data] OK: {len(ds)} rows; columns={ds.column_names}")
    print(f"[opsd_data] row0.problem[:80]={str(ds[0]['problem'])[:80]!r}")
