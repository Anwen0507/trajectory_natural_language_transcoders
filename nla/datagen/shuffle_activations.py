"""Random-activation baseline — permute activation bundles across rows.

Keeps prompts/responses/provenance fixed, shuffles the legacy activation_vector
or every named joint-checkpoint column with the SAME row permutation. If
training on this gives the same MSE as the real dataset, the
injection signal isn't doing anything (model ignores the vector). If MSE is
much worse, the activation vector carries real information.

This is the baseline from docs/design.md §7:
"Random-activation baseline — shuffle activation vectors across rows to
measure how much the signal matters."

Seeded for reproducibility. Sidecar gets a `_shuf_activations` suffix so
training can tell it's the baseline.
"""

import argparse
import hashlib
import random
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.sidecar import read_sidecar, write_sidecar
from nla.datagen.stage_shuffle import _TAKE_VALUES_BYTES_LIMIT, _take_fixed_size_list_via_numpy, _values_nbytes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="any stage3 output (av_sft/ar_sft/rl)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)

    table = pq.read_table(storage.open_read(args.input))
    if "activation_vector" in table.column_names:
        activation_columns = ["activation_vector"]
    else:
        activation_columns = [
            checkpoint.column_name
            for checkpoint in in_meta.extraction.checkpoints
        ]
        missing = [column for column in activation_columns if column not in table.column_names]
        assert not missing, (
            f"input is missing activation columns {missing}; not a compatible "
            f"stage3 output? Columns: {table.column_names}"
        )

    # Deterministic permutation keyed on (seed, dataset_id) — same input +
    # same seed → same shuffle, across environments.
    rng = random.Random(
        hashlib.sha256(f"{args.seed}|{in_meta.dataset_id}|activ".encode()).digest()
    )
    perm = list(range(table.num_rows))
    rng.shuffle(perm)

    # Apply one permutation to the whole checkpoint bundle. Independently
    # shuffling columns would create combinations that never co-occurred in a
    # target-model forward and would test a different baseline.
    perm_np = np.asarray(perm, dtype=np.int64)
    perm_pa = pa.array(perm, type=pa.int64())
    out_table = table
    for column in activation_columns:
        av_col = table.column(column)
        if _values_nbytes(av_col) > _TAKE_VALUES_BYTES_LIMIT:
            print(f"  {column}: {_values_nbytes(av_col) / 2**30:.2f} GiB — numpy gather")
            shuffled = _take_fixed_size_list_via_numpy(av_col, perm_np)
        else:
            shuffled = av_col.take(perm_pa)
        col_idx = out_table.column_names.index(column)
        out_table = out_table.set_column(col_idx, column, shuffled)

    storage.ensure_parent(args.output)
    pq.write_table(out_table, storage.open_write(args.output), row_group_size=65536)

    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__shuf_activations{args.seed}",
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.shuffle_activations",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"shuffled {activation_columns} across {table.num_rows} rows → {args.output}")
    print(f"  (prompts/responses/provenance UNCHANGED — this is the random-baseline dataset)")


if __name__ == "__main__":
    main()
