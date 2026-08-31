import gc
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class _Sample:
    def __init__(self, *, prompt, metadata):
        self.prompt = prompt
        self.metadata = metadata
        self.group_index = None
        self.index = None


def _import_data_source_with_miles_stubs():
    miles = types.ModuleType("miles")
    rollout = types.ModuleType("miles.rollout")
    rollout_data_source = types.ModuleType("miles.rollout.data_source")
    utils = types.ModuleType("miles.utils")
    processing = types.ModuleType("miles.utils.processing_utils")
    types_module = types.ModuleType("miles.utils.types")

    rollout_data_source.RolloutDataSource = type("RolloutDataSource", (), {})
    processing.load_tokenizer = lambda *_args, **_kwargs: object()
    types_module.Sample = _Sample
    stubs = {
        "miles": miles,
        "miles.rollout": rollout,
        "miles.rollout.data_source": rollout_data_source,
        "miles.utils": utils,
        "miles.utils.processing_utils": processing,
        "miles.utils.types": types_module,
    }
    sys.modules.pop("nla.data_source", None)
    with patch.dict(sys.modules, stubs):
        return importlib.import_module("nla.data_source")


class CheckpointDataSourceTest(unittest.TestCase):
    @staticmethod
    def _load_table(table, cfg):
        data_source = _import_data_source_with_miles_stubs()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training.parquet"
            pq.write_table(table, path)
            args = types.SimpleNamespace(
                prompt_data=str(path),
                hf_checkpoint="model",
                nla_sidecar_source=None,
                input_key="prompt",
                rollout_seed=7,
                rollout_shuffle=False,
                n_samples_per_prompt=1,
            )
            with (
                patch.object(data_source, "load_tokenizer", return_value=object()),
                patch.object(
                    data_source, "resolve_sidecar_source", return_value=str(path)
                ),
                patch.object(data_source, "load_nla_config", return_value=cfg),
                patch.object(gc, "freeze"),
            ):
                return data_source.NLADataSource(args)

    def test_stage3_checkpoint_provenance_is_carried_into_sample_metadata(self):
        data_source = _import_data_source_with_miles_stubs()
        prompt_type = pa.list_(
            pa.struct([("role", pa.string()), ("content", pa.string())])
        )
        schema = pa.schema([
            ("prompt", prompt_type),
            ("response", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), 2)),
            ("n_raw_tokens", pa.int64()),
            ("activation_checkpoint", pa.string()),
            ("activation_depth", pa.int64()),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
        ])
        rows = {
            "prompt": [
                [{"role": "user", "content": "explain <INJECT>"}],
                [{"role": "user", "content": "decode <INJECT>"}],
            ],
            "response": ["first", "second"],
            "activation_vector": [[1.0, 2.0], [3.0, 4.0]],
            "n_raw_tokens": [51, 52],
            "activation_checkpoint": ["embedding", "block_24"],
            "activation_depth": [0, 24],
            "activation_layer": [None, 23],
            "doc_id": ["doc-0", "doc-1"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training.parquet"
            pq.write_table(pa.table(rows, schema=schema), path)
            args = types.SimpleNamespace(
                prompt_data=str(path),
                hf_checkpoint="model",
                nla_sidecar_source=None,
                input_key="prompt",
                rollout_seed=7,
                rollout_shuffle=False,
                n_samples_per_prompt=1,
            )
            with (
                patch.object(data_source, "load_tokenizer", return_value=object()),
                patch.object(data_source, "resolve_sidecar_source", return_value=str(path)),
                patch.object(
                    data_source,
                    "load_nla_config",
                    return_value=types.SimpleNamespace(
                        injection_char="㊗",
                        d_model=2,
                        activation_checkpoint_names=("activation",),
                    ),
                ),
                patch.object(gc, "freeze"),
            ):
                source = data_source.NLADataSource(args)

        samples = source.dataset.samples
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].prompt[0]["content"], "explain ㊗")
        self.assertEqual(samples[1].prompt[0]["content"], "decode ㊗")
        np.testing.assert_array_equal(samples[0].metadata["activation_vector"], [1.0, 2.0])
        np.testing.assert_array_equal(
            samples[0].metadata["activation_vectors"], [[1.0, 2.0]]
        )
        self.assertEqual(samples[0].metadata["activation_checkpoint"], "embedding")
        self.assertEqual(samples[0].metadata["activation_depth"], 0)
        self.assertIsNone(samples[0].metadata["activation_layer"])
        self.assertEqual(samples[1].metadata["activation_checkpoint"], "block_24")
        self.assertEqual(samples[1].metadata["activation_depth"], 24)
        self.assertEqual(samples[1].metadata["activation_layer"], 23)
        self.assertEqual(samples[1].metadata["response"], "second")

    def test_joint_checkpoint_columns_stack_in_sidecar_order(self):
        data_source = _import_data_source_with_miles_stubs()
        prompt_type = pa.list_(
            pa.struct([("role", pa.string()), ("content", pa.string())])
        )
        schema = pa.schema([
            ("prompt", prompt_type),
            ("activation_embedding", pa.list_(pa.float32(), 2)),
            ("activation_block_24", pa.list_(pa.float32(), 2)),
            ("n_raw_tokens", pa.int64()),
            ("doc_id", pa.string()),
        ])
        rows = {
            "prompt": [[{
                "role": "user",
                "content": "embedding <INJECT> then block <INJECT>",
            }]],
            "activation_embedding": [[1.0, 2.0]],
            "activation_block_24": [[3.0, 4.0]],
            "n_raw_tokens": [51],
            "doc_id": ["doc-0"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "joint.parquet"
            pq.write_table(pa.table(rows, schema=schema), path)
            args = types.SimpleNamespace(
                prompt_data=str(path),
                hf_checkpoint="model",
                nla_sidecar_source=None,
                input_key="prompt",
                rollout_seed=7,
                rollout_shuffle=False,
                n_samples_per_prompt=1,
            )
            cfg = types.SimpleNamespace(
                injection_char="㊗",
                d_model=2,
                activation_checkpoint_names=("embedding", "block_24"),
            )
            with (
                patch.object(data_source, "load_tokenizer", return_value=object()),
                patch.object(data_source, "resolve_sidecar_source", return_value=str(path)),
                patch.object(data_source, "load_nla_config", return_value=cfg),
                patch.object(gc, "freeze"),
            ):
                source = data_source.NLADataSource(args)

        sample = source.dataset.samples[0]
        self.assertEqual(sample.prompt[0]["content"].count("㊗"), 2)
        np.testing.assert_array_equal(
            sample.metadata["activation_vectors"],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertNotIn("activation_vector", sample.metadata)

    def test_legacy_config_without_checkpoint_names_defaults_to_one_site(self):
        table = pa.table({
            "prompt": [[{"role": "user", "content": "single <INJECT>"}]],
            "activation_vector": pa.array(
                [[1.0, 2.0]], type=pa.list_(pa.float32(), 2)
            ),
        })
        cfg = types.SimpleNamespace(injection_char="㊗", d_model=2)

        source = self._load_table(table, cfg)

        sample = source.dataset.samples[0]
        self.assertEqual(sample.prompt[0]["content"], "single ㊗")
        np.testing.assert_array_equal(
            sample.metadata["activation_vectors"], [[1.0, 2.0]]
        )

    def test_rejects_legacy_column_for_multi_site_sidecar(self):
        table = pa.table({
            "prompt": [[{"role": "user", "content": "one <INJECT>"}]],
            "activation_vector": pa.array(
                [[1.0, 2.0]], type=pa.list_(pa.float32(), 2)
            ),
        })
        cfg = types.SimpleNamespace(
            injection_char="㊗",
            d_model=2,
            activation_checkpoint_names=("embedding", "block_24"),
        )
        with self.assertRaisesRegex(
            AssertionError, "legacy 'activation_vector'.*declares 2 checkpoints"
        ):
            self._load_table(table, cfg)

    def test_rejects_missing_joint_column_and_wrong_vector_width(self):
        cfg = types.SimpleNamespace(
            injection_char="㊗",
            d_model=2,
            activation_checkpoint_names=("embedding", "block_24"),
        )
        missing = pa.table({
            "prompt": [[{
                "role": "user",
                "content": "first <INJECT> second <INJECT>",
            }]],
            "activation_embedding": pa.array(
                [[1.0, 2.0]], type=pa.list_(pa.float32(), 2)
            ),
        })
        with self.assertRaisesRegex(
            AssertionError, "missing joint activation columns.*activation_block_24"
        ):
            self._load_table(missing, cfg)

        wrong_width = pa.table({
            "prompt": [[{
                "role": "user",
                "content": "first <INJECT> second <INJECT>",
            }]],
            "activation_embedding": pa.array(
                [[1.0, 2.0, 3.0]], type=pa.list_(pa.float32(), 3)
            ),
            "activation_block_24": pa.array(
                [[4.0, 5.0, 6.0]], type=pa.list_(pa.float32(), 3)
            ),
        })
        with self.assertRaisesRegex(
            AssertionError, r"activation bundle has shape \(1, 2, 3\).*expected"
        ):
            self._load_table(wrong_width, cfg)

    def test_rejects_prompt_marker_count_mismatch(self):
        cfg = types.SimpleNamespace(
            injection_char="㊗",
            d_model=2,
            activation_checkpoint_names=("embedding", "block_24"),
        )
        table = pa.table({
            "prompt": [[{"role": "user", "content": "only <INJECT>"}]],
            "activation_embedding": pa.array(
                [[1.0, 2.0]], type=pa.list_(pa.float32(), 2)
            ),
            "activation_block_24": pa.array(
                [[3.0, 4.0]], type=pa.list_(pa.float32(), 2)
            ),
        })
        with self.assertRaisesRegex(
            AssertionError, "prompt contains 1 '<INJECT>' markers, expected 2"
        ):
            self._load_table(table, cfg)


if __name__ == "__main__":
    unittest.main()
