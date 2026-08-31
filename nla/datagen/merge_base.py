"""Merge stage0 shard parquets into a single base parquet + sidecar.

Stage0's per-doc keyed RNG means N parallel runs on disjoint --corpus-start
slices produce row-for-row identical output to a single serial run, just
faster. This merges those shards back into one file with a sidecar that
looks exactly like what a serial run would have written — same dataset_id
hash, merged corpus_slice.

Shards must be contiguous (no gaps, no overlaps) and must agree on all
extraction params. Any mismatch is a hard failure.
"""

import argparse
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.checkpoints import base_dataset_id
from nla.datagen.sidecar import NLADatasetMeta, read_sidecar, write_sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="shard parquet paths")
    p.add_argument("--output", required=True, help="merged parquet path")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)

    metas: list[NLADatasetMeta] = [read_sidecar(storage, path) for path in args.inputs]
    for m, path in zip(metas, args.inputs, strict=True):
        assert m.stage == "base", f"{path}: expected stage=base, got stage={m.stage!r}"

    # Sort by corpus_slice.start so concat order matches serial-run doc order.
    order = sorted(range(len(metas)), key=lambda i: metas[i].extraction.corpus_slice["start"])
    metas = [metas[i] for i in order]
    paths = [args.inputs[i] for i in order]

    ref = metas[0].extraction
    for m, path in zip(metas[1:], paths[1:], strict=True):
        e = m.extraction
        assert e.base_model == ref.base_model, f"{path}: base_model {e.base_model!r} != {ref.base_model!r}"
        assert e.layer_index == ref.layer_index, f"{path}: layer_index {e.layer_index} != {ref.layer_index}"
        assert e.checkpoints == ref.checkpoints, (
            f"{path}: checkpoints {e.checkpoints} != {ref.checkpoints}"
        )
        assert e.d_model == ref.d_model, f"{path}: d_model {e.d_model} != {ref.d_model}"
        assert e.norm == ref.norm, f"{path}: norm {e.norm!r} != {ref.norm!r}"
        assert e.corpus == ref.corpus, f"{path}: corpus {e.corpus!r} != {ref.corpus!r}"
        assert e.positions_per_doc == ref.positions_per_doc, (
            f"{path}: positions_per_doc {e.positions_per_doc} != {ref.positions_per_doc}"
        )

    slices = [m.extraction.corpus_slice for m in metas]
    for i in range(len(slices) - 1):
        expected_next = slices[i]["start"] + slices[i]["length"]
        actual_next = slices[i + 1]["start"]
        assert actual_next == expected_next, (
            f"shards not contiguous: {paths[i]} covers [{slices[i]['start']}, {expected_next}) "
            f"but {paths[i + 1]} starts at {actual_next} (gap or overlap)"
        )

    merged_slice = {"start": slices[0]["start"], "length": sum(s["length"] for s in slices)}

    tables = [pq.read_table(storage.open_read(path)) for path in paths]
    merged = pa.concat_tables(tables)
    row_count = merged.num_rows
    assert row_count == sum(m.row_count for m in metas), (
        f"merged rows {row_count} != sum of sidecar row_counts {sum(m.row_count for m in metas)}"
    )

    storage.ensure_parent(args.output)
    # Smaller row groups so downstream readers don't hit int32 list-offset
    # overflow on large activation columns (1M rows × d=3584 > 2.1B elements).
    pq.write_table(merged, storage.open_write(args.output), row_group_size=65536)

    out_meta = replace(
        metas[0],
        dataset_id=base_dataset_id(
            ref.base_model,
            ref.checkpoints,
            ref.corpus,
            merged_slice,
            legacy_layer_index=ref.layer_index,
        ),
        row_count=row_count,
        extraction=replace(ref, corpus_slice=merged_slice),
        parent_datasets=[m.dataset_id for m in metas],
        created_by="nla.datagen.merge_base",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)

    print(f"merged {len(paths)} shards → {row_count} rows → {args.output}")
    print(f"  corpus_slice: [{merged_slice['start']}, {merged_slice['start'] + merged_slice['length']})")
    print(f"  dataset_id: {out_meta.dataset_id}")


if __name__ == "__main__":
    main()
