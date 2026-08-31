import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen import merge_base
from nla.datagen.checkpoints import base_dataset_id, checkpoints_for_depths
from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    read_sidecar,
    write_sidecar,
)
from nla.datagen.stage0_extract import _schema
from nla.datagen.storage import LocalStorage


class MergeMultiCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.storage = LocalStorage()
        self.checkpoints = checkpoints_for_depths([0, 24], 24)

    def _write_shard(
        self,
        path,
        *,
        start,
        length,
        dataset_id,
        checkpoints=None,
        row_count=1,
        stage="base",
    ):
        checkpoints = list(checkpoints or self.checkpoints)
        schema = _schema(2, self.checkpoints, legacy_single_layer=False)
        rows = {
            "n_raw_tokens": [51 + start] * row_count,
            "detokenized_text_truncated": [f"text-{start}"] * row_count,
            "activation_embedding": [[float(start), 0.0]] * row_count,
            "activation_block_24": [[float(start), 24.0]] * row_count,
            "doc_id": [f"doc-{start}"] * row_count,
        }
        pq.write_table(pa.table(rows, schema=schema), path)
        extraction = NLAExtractionMeta(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            d_model=2,
            layer_index=None,
            norm="none",
            corpus="corpus",
            corpus_slice={"start": start, "length": length},
            positions_per_doc=1,
            checkpoints=checkpoints,
        )
        meta = NLADatasetMeta(dataset_id, stage, row_count, extraction)
        write_sidecar(self.storage, str(path), meta)
        return meta

    def _run(self, inputs, output):
        argv = [
            "merge_base",
            "--inputs",
            *map(str, inputs),
            "--output",
            str(output),
        ]
        with patch.object(sys, "argv", argv):
            merge_base.main()

    def test_merges_reversed_multi_checkpoint_shards_and_rebuilds_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard0 = Path(temp_dir) / "shard0.parquet"
            shard1 = Path(temp_dir) / "shard1.parquet"
            self._write_shard(shard0, start=0, length=2, dataset_id="shard-zero")
            self._write_shard(shard1, start=2, length=3, dataset_id="shard-one")
            output = Path(temp_dir) / "merged" / "base.parquet"

            self._run([shard1, shard0], output)

            table = pq.read_table(output)
            self.assertEqual(table.column("doc_id").to_pylist(), ["doc-0", "doc-2"])
            self.assertEqual(table.column("activation_embedding").to_pylist(), [[0.0, 0.0], [2.0, 0.0]])
            meta = read_sidecar(self.storage, str(output))
            self.assertEqual(meta.row_count, 2)
            self.assertEqual(meta.extraction.corpus_slice, {"start": 0, "length": 5})
            self.assertEqual(meta.extraction.checkpoints, self.checkpoints)
            self.assertEqual(meta.parent_datasets, ["shard-zero", "shard-one"])
            self.assertEqual(
                meta.dataset_id,
                base_dataset_id(
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    self.checkpoints,
                    "corpus",
                    {"start": 0, "length": 5},
                ),
            )

    def test_rejects_checkpoint_mismatch_before_concatenation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard0 = Path(temp_dir) / "shard0.parquet"
            shard1 = Path(temp_dir) / "shard1.parquet"
            self._write_shard(shard0, start=0, length=1, dataset_id="zero")
            self._write_shard(
                shard1,
                start=1,
                length=1,
                dataset_id="one",
                checkpoints=checkpoints_for_depths([0, 4]),
            )
            with self.assertRaisesRegex(AssertionError, "checkpoints .* !="):
                self._run([shard0, shard1], Path(temp_dir) / "merged.parquet")

    def test_rejects_non_base_noncontiguous_and_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard0 = Path(temp_dir) / "shard0.parquet"
            shard1 = Path(temp_dir) / "shard1.parquet"
            meta0 = self._write_shard(shard0, start=0, length=1, dataset_id="zero")
            self._write_shard(shard1, start=2, length=1, dataset_id="one")
            output = Path(temp_dir) / "merged.parquet"
            with self.assertRaisesRegex(AssertionError, "not contiguous"):
                self._run([shard0, shard1], output)

            write_sidecar(self.storage, str(shard0), replace(meta0, stage="not_base"))
            with self.assertRaisesRegex(AssertionError, "expected stage=base"):
                self._run([shard0], output)

            write_sidecar(self.storage, str(shard0), replace(meta0, row_count=2))
            with self.assertRaisesRegex(AssertionError, "merged rows 1 !="):
                self._run([shard0], output)


if __name__ == "__main__":
    unittest.main()
