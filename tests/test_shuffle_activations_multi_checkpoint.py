import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen import shuffle_activations
from nla.datagen.checkpoints import checkpoints_for_depths
from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    read_sidecar,
    write_sidecar,
)
from nla.datagen.storage import LocalStorage


class ShuffleActivationBundleTest(unittest.TestCase):
    def test_joint_columns_move_with_one_shared_row_permutation(self):
        checkpoints = checkpoints_for_depths([0, 4, 8], num_layers=8)
        storage = LocalStorage()
        rows = 8

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "joint.parquet"
            output_path = Path(temp_dir) / "joint_shuffled.parquet"
            data = {
                "row_id": list(range(rows)),
                "activation_embedding": [[float(i), float(i)] for i in range(rows)],
                "activation_block_04": [[100.0 + i, 100.0 + i] for i in range(rows)],
                "activation_block_08": [[200.0 + i, 200.0 + i] for i in range(rows)],
            }
            schema = pa.schema([
                ("row_id", pa.int64()),
                ("activation_embedding", pa.list_(pa.float32(), 2)),
                ("activation_block_04", pa.list_(pa.float32(), 2)),
                ("activation_block_08", pa.list_(pa.float32(), 2)),
            ])
            pq.write_table(pa.table(data, schema=schema), input_path)
            meta = NLADatasetMeta(
                dataset_id="joint",
                stage="base",
                row_count=rows,
                extraction=NLAExtractionMeta(
                    base_model="model",
                    d_model=2,
                    layer_index=None,
                    norm="none",
                    corpus="corpus",
                    corpus_slice={"start": 0, "length": rows},
                    positions_per_doc=1,
                    checkpoints=checkpoints,
                ),
            )
            write_sidecar(storage, str(input_path), meta)

            argv = [
                "shuffle_activations",
                "--input", str(input_path),
                "--output", str(output_path),
                "--seed", "23",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(shuffle_activations, "make_storage", return_value=storage),
            ):
                shuffle_activations.main()

            output = pq.read_table(output_path).to_pydict()
            self.assertEqual(output["row_id"], list(range(rows)))
            embedding_order = [int(vector[0]) for vector in output["activation_embedding"]]
            self.assertNotEqual(embedding_order, list(range(rows)))
            self.assertEqual(sorted(embedding_order), list(range(rows)))
            self.assertEqual(
                [int(vector[0] - 100) for vector in output["activation_block_04"]],
                embedding_order,
            )
            self.assertEqual(
                [int(vector[0] - 200) for vector in output["activation_block_08"]],
                embedding_order,
            )

            output_meta = read_sidecar(storage, str(output_path))
            self.assertEqual(output_meta.extraction.checkpoints, checkpoints)
            self.assertEqual(output_meta.parent_datasets, ["joint"])
            self.assertIn("shuf_activations23", output_meta.dataset_id)

    def test_legacy_column_uses_numpy_gather_for_large_values(self):
        storage = LocalStorage()
        rows = 5
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "legacy.parquet"
            output_path = Path(temp_dir) / "legacy_shuffled.parquet"
            table = pa.table({
                "row_id": list(range(rows)),
                "activation_vector": pa.array(
                    [[float(i), float(i + 10)] for i in range(rows)],
                    type=pa.list_(pa.float32(), 2),
                ),
            })
            pq.write_table(table, input_path)
            meta = NLADatasetMeta(
                dataset_id="legacy",
                stage="base",
                row_count=rows,
                extraction=NLAExtractionMeta(
                    base_model="model",
                    d_model=2,
                    layer_index=3,
                    norm="none",
                    corpus="corpus",
                    corpus_slice={"start": 0, "length": rows},
                    positions_per_doc=1,
                ),
            )
            write_sidecar(storage, str(input_path), meta)

            argv = [
                "shuffle_activations",
                "--input", str(input_path),
                "--output", str(output_path),
                "--seed", "11",
            ]
            real_take = shuffle_activations._take_fixed_size_list_via_numpy
            with (
                patch.object(sys, "argv", argv),
                patch.object(shuffle_activations, "make_storage", return_value=storage),
                patch.object(
                    shuffle_activations,
                    "_values_nbytes",
                    return_value=shuffle_activations._TAKE_VALUES_BYTES_LIMIT + 1,
                ),
                patch.object(
                    shuffle_activations,
                    "_take_fixed_size_list_via_numpy",
                    wraps=real_take,
                ) as numpy_take,
            ):
                shuffle_activations.main()

            numpy_take.assert_called_once()
            output = pq.read_table(output_path).to_pydict()
            order = [int(vector[0]) for vector in output["activation_vector"]]
            self.assertEqual(sorted(order), list(range(rows)))
            self.assertNotEqual(order, list(range(rows)))
            self.assertEqual(output["row_id"], list(range(rows)))

    def test_rejects_joint_sidecar_when_named_column_is_missing(self):
        storage = LocalStorage()
        checkpoints = checkpoints_for_depths([0, 4], num_layers=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "bad.parquet"
            output_path = Path(temp_dir) / "unused.parquet"
            pq.write_table(
                pa.table({
                    "activation_embedding": pa.array(
                        [[1.0, 2.0]], type=pa.list_(pa.float32(), 2)
                    ),
                }),
                input_path,
            )
            write_sidecar(
                storage,
                str(input_path),
                NLADatasetMeta(
                    dataset_id="bad",
                    stage="base",
                    row_count=1,
                    extraction=NLAExtractionMeta(
                        base_model="model",
                        d_model=2,
                        layer_index=None,
                        norm="none",
                        corpus="corpus",
                        corpus_slice={"start": 0, "length": 1},
                        positions_per_doc=1,
                        checkpoints=checkpoints,
                    ),
                ),
            )
            argv = [
                "shuffle_activations",
                "--input", str(input_path),
                "--output", str(output_path),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(shuffle_activations, "make_storage", return_value=storage),
                self.assertRaisesRegex(
                    AssertionError, "missing activation columns.*activation_block_04"
                ),
            ):
                shuffle_activations.main()


if __name__ == "__main__":
    unittest.main()
