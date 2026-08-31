import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nla.datagen import stage3_build
from nla.datagen.checkpoints import (
    ActivationCheckpoint,
    base_dataset_id,
    checkpoint_for_depth,
    checkpoints_for_depths,
)
from nla.datagen.model_presets import resolve as resolve_model_preset
from nla.datagen.run_pipeline import _paths, _training_path
from nla.datagen.sidecar import (
    NLAApiSummaryMeta,
    NLADatasetMeta,
    NLAExtractionMeta,
    deserialize_sidecar,
    read_sidecar,
    read_sidecar_local,
    serialize_sidecar,
    write_sidecar,
    write_sidecar_local,
)
from nla.datagen.stage0_extract import _schema
from nla.datagen.stage3_build import _resolve_activation_source, _schema_for
from nla.datagen.storage import LocalStorage
from nla.schema import NLATokenMeta


class CheckpointMetadataTest(unittest.TestCase):
    def setUp(self):
        self.depths = [0, 4, 8, 12, 16, 20, 24]
        self.checkpoints = checkpoints_for_depths(self.depths, num_layers=24)
        self.extraction = NLAExtractionMeta(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            d_model=896,
            layer_index=None,
            norm="none",
            corpus="example/corpus",
            corpus_slice={"start": 0, "length": 10},
            positions_per_doc=2,
            checkpoints=self.checkpoints,
        )

    def test_checkpoint_names_and_columns(self):
        self.assertEqual(
            [checkpoint.name for checkpoint in self.checkpoints],
            ["embedding", "block_04", "block_08", "block_12", "block_16", "block_20", "block_24"],
        )
        self.assertEqual(self.checkpoints[-1].layer_index, 23)
        self.assertFalse(self.checkpoints[-1].final_norm_applied)
        schema = _schema(896, self.checkpoints, legacy_single_layer=False)
        self.assertEqual(schema.names[2:-1], [checkpoint.column_name for checkpoint in self.checkpoints])
        for checkpoint in self.checkpoints:
            self.assertTrue(pa.types.is_fixed_size_list(schema.field(checkpoint.column_name).type))
            self.assertEqual(schema.field(checkpoint.column_name).type.list_size, 896)

    def test_embedding_and_arbitrary_block_checkpoint_semantics(self):
        embedding = checkpoint_for_depth(0)
        self.assertEqual(
            embedding,
            ActivationCheckpoint("embedding", 0, "embedding", None, False),
        )
        self.assertEqual(embedding.column_name, "activation_embedding")

        block = checkpoint_for_depth(1)
        self.assertEqual(block.name, "block_01")
        self.assertEqual(block.kind, "block_output")
        self.assertEqual(block.layer_index, 0)
        self.assertFalse(block.final_norm_applied)
        self.assertEqual(checkpoint_for_depth(100).name, "block_100")
        with self.assertRaisesRegex(AssertionError, "non-negative"):
            checkpoint_for_depth(-1)

    def test_checkpoint_depth_validation_accepts_boundaries_and_rejects_bad_lists(self):
        self.assertEqual(
            [checkpoint.depth for checkpoint in checkpoints_for_depths((0, 24), 24)],
            [0, 24],
        )
        cases = [
            ([], 24, "at least one"),
            ([4, 4], 24, "sorted and unique"),
            ([8, 4], 24, "sorted and unique"),
            ([-1, 4], 24, "non-negative"),
            ([0, 25], 24, "exceed model depth"),
            ([0], -1, "exceed model depth"),
        ]
        for depths, num_layers, message in cases:
            with (
                self.subTest(depths=depths, num_layers=num_layers),
                self.assertRaisesRegex(AssertionError, message),
            ):
                checkpoints_for_depths(depths, num_layers)

    def test_legacy_stage0_schema_is_unchanged(self):
        schema = _schema(3, [checkpoint_for_depth(11)], legacy_single_layer=True)
        self.assertEqual(
            schema.names,
            [
                "n_raw_tokens",
                "detokenized_text_truncated",
                "activation_vector",
                "activation_layer",
                "doc_id",
            ],
        )
        self.assertEqual(schema.field("activation_vector").type.list_size, 3)

    def test_qwen05b_preset_preserves_multi_checkpoint_target(self):
        cfg = resolve_model_preset({
            "model": "qwen05b",
            "checkpoint_depths": self.depths,
            "stage0": {},
        })
        self.assertEqual(cfg["base_model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertNotIn("layer_index", cfg)
        self.assertEqual(cfg["stage0"]["extractor_kwargs"]["max_length"], 1024)
        paths = _paths("/tmp/example")
        self.assertEqual(
            _training_path(cfg | {"output_dir": "/tmp/example"}, paths, "rl", 24),
            "/tmp/example/checkpoints/block_24/rl.parquet",
        )
        self.assertEqual(
            _training_path(
                cfg | {"output_dir": "/tmp/example"},
                paths,
                "av_sft",
                0,
                shuffled=True,
            ),
            "/tmp/example/checkpoints/embedding/av_sft_shuf.parquet",
        )

    def test_model_preset_legacy_defaults_and_explicit_values_win(self):
        legacy = resolve_model_preset({"model": "qwen05b"})
        self.assertEqual(legacy["layer_index"], 16)
        self.assertEqual(legacy["stage0"]["extractor_kwargs"]["batch_size"], 4)
        paths = _paths("/tmp/legacy")
        self.assertEqual(_training_path(legacy, paths, "rl", None), "/tmp/legacy/rl.parquet")
        self.assertEqual(
            _training_path(legacy, paths, "rl", None, shuffled=True),
            "/tmp/legacy/rl_shuf.parquet",
        )

        explicit = resolve_model_preset({
            "model": "qwen05b",
            "base_model": "/models/local",
            "layer_index": 7,
            "stage0": {"extractor_kwargs": {"batch_size": 1}},
        })
        self.assertEqual(explicit["base_model"], "/models/local")
        self.assertEqual(explicit["layer_index"], 7)
        self.assertEqual(explicit["stage0"]["extractor_kwargs"], {"batch_size": 1})
        passthrough = {"base_model": "manual"}
        self.assertIs(resolve_model_preset(passthrough), passthrough)
        with self.assertRaisesRegex(AssertionError, "unknown model preset"):
            resolve_model_preset({"model": "missing"})

    def test_v2_sidecar_round_trip(self):
        meta = NLADatasetMeta(
            dataset_id="multi",
            stage="base",
            row_count=20,
            extraction=self.extraction,
        )
        loaded = deserialize_sidecar(serialize_sidecar(meta))
        self.assertEqual(loaded.schema_version, 2)
        self.assertEqual(loaded.extraction.checkpoints, self.checkpoints)
        self.assertIsNone(loaded.extraction.layer_index)

    def test_v2_sidecar_round_trip_restores_nested_metadata_and_ignores_unknown_keys(self):
        token_meta = NLATokenMeta("x", 1, 2, 3, [4, 5])
        api_meta = NLAApiSummaryMeta("provider", 100, 0.25, "instruction")
        meta = NLADatasetMeta(
            dataset_id="nested",
            stage="base",
            row_count=2,
            extraction=self.extraction,
            tokens=token_meta,
            api_summaries=api_meta,
        )
        serialized = serialize_sidecar(meta) + "future_top_level_key: ignored\n"
        loaded = deserialize_sidecar(serialized)
        self.assertEqual(loaded.tokens, token_meta)
        self.assertEqual(loaded.api_summaries, api_meta)
        self.assertFalse(hasattr(loaded, "future_top_level_key"))

    def test_v1_sidecar_maps_layer_index_to_one_checkpoint(self):
        loaded = deserialize_sidecar("""
kind: nla_dataset
schema_version: 1
dataset_id: legacy
stage: base
row_count: 1
extraction:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  d_model: 896
  layer_index: 10
  norm: none
  corpus: example/corpus
  corpus_slice: {start: 0, length: 1}
  positions_per_doc: 1
""")
        self.assertEqual(loaded.schema_version, 2)
        self.assertEqual(loaded.extraction.checkpoints[0].depth, 11)
        self.assertEqual(loaded.extraction.checkpoints[0].layer_index, 10)

    def test_legacy_metadata_construction_is_still_serializable(self):
        extraction = NLAExtractionMeta(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            d_model=896,
            layer_index=10,
            norm="none",
            corpus="example/corpus",
            corpus_slice={"start": 0, "length": 1},
            positions_per_doc=1,
        )
        meta = NLADatasetMeta(
            dataset_id="legacy-construction",
            stage="base",
            row_count=1,
            extraction=extraction,
        )
        loaded = deserialize_sidecar(serialize_sidecar(meta))
        self.assertEqual(loaded.extraction.checkpoints[0].depth, 11)
        self.assertEqual(extraction.checkpoints, [checkpoint_for_depth(11)])

    def test_sidecar_rejects_invalid_checkpoint_contracts(self):
        def meta_with(extraction):
            return NLADatasetMeta("bad", "base", 1, extraction)

        with self.assertRaisesRegex(AssertionError, "must not be empty"):
            serialize_sidecar(meta_with(replace(self.extraction, checkpoints=[])))
        with self.assertRaisesRegex(AssertionError, "sorted and unique"):
            serialize_sidecar(meta_with(replace(
                self.extraction,
                checkpoints=[checkpoint_for_depth(4), checkpoint_for_depth(4)],
            )))
        with self.assertRaisesRegex(AssertionError, "sorted and unique"):
            serialize_sidecar(meta_with(replace(
                self.extraction,
                checkpoints=[checkpoint_for_depth(8), checkpoint_for_depth(4)],
            )))
        with self.assertRaisesRegex(AssertionError, "only valid for a single"):
            serialize_sidecar(meta_with(replace(self.extraction, layer_index=3)))
        with self.assertRaisesRegex(AssertionError, "disagrees"):
            serialize_sidecar(meta_with(replace(
                self.extraction,
                layer_index=8,
                checkpoints=[checkpoint_for_depth(4)],
            )))

    def test_deserialize_rejects_malformed_checkpoint_contracts(self):
        valid = yaml.safe_load(serialize_sidecar(NLADatasetMeta(
            "valid",
            "base",
            1,
            self.extraction,
        )))
        cases = []

        unsorted = yaml.safe_load(yaml.safe_dump(valid))
        unsorted["extraction"]["checkpoints"] = list(
            reversed(unsorted["extraction"]["checkpoints"])
        )
        cases.append((unsorted, "sorted and unique"))

        negative = yaml.safe_load(yaml.safe_dump(valid))
        negative["extraction"]["checkpoints"][0]["depth"] = -1
        cases.append((negative, "non-negative"))

        layer_with_many = yaml.safe_load(yaml.safe_dump(valid))
        layer_with_many["extraction"]["layer_index"] = 3
        cases.append((layer_with_many, "only valid for a single"))

        mismatch = yaml.safe_load(yaml.safe_dump(valid))
        mismatch["extraction"]["checkpoints"] = [
            mismatch["extraction"]["checkpoints"][1]
        ]
        mismatch["extraction"]["layer_index"] = 99
        cases.append((mismatch, "disagrees"))

        for data, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(AssertionError, message),
            ):
                deserialize_sidecar(yaml.safe_dump(data))

    def test_sidecar_rejects_invalid_kind_and_schema_version(self):
        for text, message in [
            ("kind: other\nschema_version: 2\n", "not an NLA dataset"),
            ("kind: nla_dataset\nschema_version: 999\n", "unsupported sidecar schema"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                deserialize_sidecar(text)

    def test_training_sidecar_requirements_still_apply(self):
        base = NLADatasetMeta("train", "rl", 1, self.extraction)
        with self.assertRaisesRegex(AssertionError, "requires NLATokenMeta"):
            serialize_sidecar(base)
        tokens = NLATokenMeta("x", 1, 2, 3)
        with self.assertRaisesRegex(AssertionError, r"requires prompt_templates\['actor'\]"):
            serialize_sidecar(replace(base, tokens=tokens))
        actor_template = "\n".join(
            "<concept>{injection_char}</concept>"
            for _ in self.extraction.checkpoints
        )
        ar = replace(
            base,
            stage="ar_sft",
            tokens=tokens,
            prompt_templates={"actor": actor_template},
        )
        with self.assertRaisesRegex(AssertionError, "requires critic_suffix_ids"):
            serialize_sidecar(ar)
        tokens_with_suffix = replace(tokens, critic_suffix_ids=[4])
        with self.assertRaisesRegex(AssertionError, r"requires prompt_templates\['critic'\]"):
            serialize_sidecar(replace(ar, tokens=tokens_with_suffix))

    def test_sidecar_storage_and_pathlib_io_round_trip(self):
        meta = NLADatasetMeta("io", "base", 1, self.extraction)
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "nested" / "data.parquet"
            parquet_path.parent.mkdir()
            storage = LocalStorage()
            write_sidecar(storage, str(parquet_path), meta)
            self.assertEqual(read_sidecar(storage, str(parquet_path)).dataset_id, "io")

            local_parquet = Path(temp_dir) / "local.parquet"
            write_sidecar_local(local_parquet, meta)
            self.assertEqual(read_sidecar_local(local_parquet).dataset_id, "io")

    def test_legacy_dataset_id_is_unchanged(self):
        corpus_slice = {"start": 0, "length": 10}
        checkpoint = checkpoints_for_depths([11])
        actual = base_dataset_id(
            "Qwen/Qwen2.5-0.5B-Instruct",
            checkpoint,
            "example/corpus",
            corpus_slice,
            legacy_layer_index=10,
        )
        digest = hashlib.sha256(
            f"Qwen/Qwen2.5-0.5B-Instruct|10|example/corpus|{corpus_slice}".encode()
        ).hexdigest()[:8]
        self.assertEqual(actual, f"base_Qwen2.5-0.5B-Instruct_L10_{digest}")

    def test_multi_checkpoint_dataset_id_is_stable_and_target_sensitive(self):
        corpus_slice = {"start": 2, "length": 10}
        actual = base_dataset_id(
            "Qwen/Qwen2.5-0.5B-Instruct",
            self.checkpoints,
            "example/corpus",
            corpus_slice,
        )
        identity = (
            "Qwen/Qwen2.5-0.5B-Instruct|"
            "checkpoint_depths=(0, 4, 8, 12, 16, 20, 24)|"
            f"example/corpus|{corpus_slice}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
        self.assertEqual(
            actual,
            f"base_Qwen2.5-0.5B-Instruct_C0-4-8-12-16-20-24_{digest}",
        )
        other = base_dataset_id(
            "Qwen/Qwen2.5-0.5B-Instruct",
            self.checkpoints[:-1],
            "example/corpus",
            corpus_slice,
        )
        self.assertNotEqual(actual, other)

    def test_stage3_selects_one_checkpoint(self):
        meta = NLADatasetMeta(
            dataset_id="multi",
            stage="base",
            row_count=20,
            extraction=self.extraction,
        )
        columns = ["n_raw_tokens", "doc_id", *(checkpoint.column_name for checkpoint in self.checkpoints)]
        checkpoint, column, is_multi = _resolve_activation_source(meta, columns, 24)
        self.assertEqual(checkpoint.name, "block_24")
        self.assertEqual(column, "activation_block_24")
        self.assertTrue(is_multi)
        out_schema = _schema_for("rl", False, 896, multi_checkpoint_input=True)
        self.assertIn("activation_checkpoint", out_schema.names)
        self.assertIn("activation_depth", out_schema.names)

    def test_stage3_materializes_selected_vector_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "base.parquet"
            output_path = Path(temp_dir) / "rl_block24.parquet"
            schema = _schema(896, self.checkpoints, legacy_single_layer=False)
            rows = {
                "n_raw_tokens": [51, 52],
                "detokenized_text_truncated": ["alpha", "beta"],
                "doc_id": ["doc-a", "doc-b"],
            }
            for checkpoint in self.checkpoints:
                rows[checkpoint.column_name] = [
                    [float(checkpoint.depth)] * 896,
                    [float(checkpoint.depth + 1)] * 896,
                ]
            pq.write_table(pa.table(rows, schema=schema), input_path)
            meta = NLADatasetMeta(
                dataset_id="multi",
                stage="base",
                row_count=2,
                extraction=self.extraction,
            )
            storage = LocalStorage()
            write_sidecar(storage, str(input_path), meta)

            token_meta = NLATokenMeta(
                injection_char="x",
                injection_token_id=1,
                injection_left_neighbor_id=2,
                injection_right_neighbor_id=3,
            )
            argv = [
                "stage3_build",
                "--input", str(input_path),
                "--stage", "rl",
                "--checkpoint-depth", "24",
                "--output", str(output_path),
                "--no-keep-debug-metadata",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(stage3_build, "load_tokenizer", return_value=object()),
                patch.object(stage3_build, "build_token_meta", return_value=token_meta),
                patch.object(stage3_build, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
            ):
                stage3_build.main()

            output = pq.read_table(output_path)
            self.assertEqual(output.column("activation_checkpoint").to_pylist(), ["block_24"] * 2)
            self.assertEqual(output.column("activation_depth").to_pylist(), [24, 24])
            self.assertEqual(output.column("activation_layer").to_pylist(), [23, 23])
            first_vector = output.column("activation_vector")[0].as_py()
            self.assertEqual(first_vector, [24.0] * 896)
            output_meta = read_sidecar(storage, str(output_path))
            self.assertEqual(output_meta.extraction.checkpoints, [self.checkpoints[-1]])
            self.assertEqual(output_meta.extraction.layer_index, 23)


if __name__ == "__main__":
    unittest.main()
