import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen import stage3_build
from nla.datagen.checkpoints import checkpoint_for_depth, checkpoints_for_depths
from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    read_sidecar,
    write_sidecar,
)
from nla.datagen.stage0_extract import _schema as stage0_schema
from nla.datagen.stage3_build import (
    _build_ar_sft_cols,
    _build_av_sft_cols,
    _build_rl_cols,
    _resolve_activation_source,
    _resolve_activation_sources,
    _schema_for,
)
from nla.datagen.storage import LocalStorage
from nla.schema import NLATokenMeta, wrap_explanation

DEPTHS = [0, 4, 8, 12, 16, 20, 24]


class _Tokenizer:
    def __init__(self, suffix_ids=(8, 9), *, bad_tail=False):
        self.suffix_ids = list(suffix_ids)
        self.bad_tail = bad_tail
        self.prompts = []

    def __call__(self, prompt, *, add_special_tokens):
        self.prompts.append((prompt, add_special_tokens))
        tail = [99] if self.bad_tail else self.suffix_ids
        return {"input_ids": [1, 2, *tail]}


def _token_meta(suffix=(8, 9)):
    return NLATokenMeta(
        injection_char="x",
        injection_token_id=1,
        injection_left_neighbor_id=2,
        injection_right_neighbor_id=3,
        critic_suffix_ids=list(suffix),
    )


def _extraction(checkpoints=None, *, layer_index=None, norm="none"):
    return NLAExtractionMeta(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        d_model=2,
        layer_index=layer_index,
        norm=norm,
        corpus="corpus",
        corpus_slice={"start": 0, "length": 2},
        positions_per_doc=1,
        checkpoints=list(checkpoints or []),
    )


class Stage3HelpersTest(unittest.TestCase):
    def setUp(self):
        self.checkpoints = checkpoints_for_depths(DEPTHS, 24)
        self.meta = NLADatasetMeta("base_multi", "base", 2, _extraction(self.checkpoints))

    def test_all_output_schema_variants(self):
        expected_core = {
            "av_sft": ["prompt", "response", "activation_vector"],
            "ar_sft": ["prompt", "activation_vector"],
            "rl": ["prompt", "activation_vector"],
        }
        for stage, core in expected_core.items():
            for multi in (False, True):
                for debug in (False, True):
                    with self.subTest(stage=stage, multi=multi, debug=debug):
                        schema = _schema_for(
                            stage,
                            debug,
                            2,
                            multi_checkpoint_input=multi,
                        )
                        provenance = (
                            [
                                "n_raw_tokens",
                                "activation_checkpoint",
                                "activation_depth",
                                "activation_layer",
                                "doc_id",
                            ]
                            if multi
                            else ["n_raw_tokens", "activation_layer", "doc_id"]
                        )
                        expected = core + provenance
                        if debug:
                            expected.append("detokenized_text_truncated")
                        self.assertEqual(schema.names, expected)
                        self.assertEqual(schema.field("activation_vector").type.list_size, 2)
        with self.assertRaisesRegex(AssertionError, "unreachable"):
            _schema_for("unknown", False, 2)

        joint = _schema_for(
            "av_sft",
            False,
            2,
            multi_checkpoint_input=True,
            joint_checkpoints=self.checkpoints,
        )
        self.assertEqual(
            joint.names,
            [
                "prompt",
                "response",
                *[checkpoint.column_name for checkpoint in self.checkpoints],
                "n_raw_tokens",
                "doc_id",
            ],
        )
        with self.assertRaisesRegex(AssertionError, "joint-checkpoint AR"):
            _schema_for(
                "ar_sft", False, 2, joint_checkpoints=self.checkpoints
            )

    def test_resolves_legacy_source_with_optional_matching_depth(self):
        checkpoint = checkpoint_for_depth(11)
        meta = replace(self.meta, extraction=_extraction([checkpoint], layer_index=10))
        for requested in (None, 11):
            with self.subTest(requested=requested):
                self.assertEqual(
                    _resolve_activation_source(meta, ["activation_vector"], requested),
                    (checkpoint, "activation_vector", False),
                )
        with self.assertRaisesRegex(AssertionError, "does not match"):
            _resolve_activation_source(meta, ["activation_vector"], 10)

    def test_resolves_every_named_multi_checkpoint_source(self):
        columns = [checkpoint.column_name for checkpoint in self.checkpoints]
        for checkpoint in self.checkpoints:
            with self.subTest(depth=checkpoint.depth):
                self.assertEqual(
                    _resolve_activation_source(self.meta, columns, checkpoint.depth),
                    (checkpoint, checkpoint.column_name, True),
                )

    def test_resolves_joint_sources_in_checkpoint_order(self):
        columns = [checkpoint.column_name for checkpoint in self.checkpoints]
        self.assertEqual(
            _resolve_activation_sources(
                self.meta, columns, None, allow_joint=True
            ),
            (self.checkpoints, columns, True, True),
        )
        with self.assertRaisesRegex(AssertionError, "joint AR/reward"):
            _resolve_activation_sources(
                self.meta, columns, None, allow_joint=False
            )

    def test_activation_source_contract_errors(self):
        empty = replace(self.meta, extraction=_extraction([]))
        duplicate = replace(
            self.meta,
            extraction=_extraction([checkpoint_for_depth(4), checkpoint_for_depth(4)]),
        )
        cases = [
            (empty, [], 0, "no extraction checkpoints"),
            (self.meta, ["activation_vector"], None, "declares multiple checkpoints"),
            (self.meta, [cp.column_name for cp in self.checkpoints], None, "requires --checkpoint-depth"),
            (self.meta, [cp.column_name for cp in self.checkpoints], 3, "not present exactly once"),
            (duplicate, ["activation_block_04"], 4, "not present exactly once"),
            (self.meta, ["activation_embedding"], 24, "input has no 'activation_block_24'"),
        ]
        for meta, columns, requested, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                _resolve_activation_source(meta, columns, requested)

    def test_column_builders_cover_empty_and_nonempty_batches(self):
        batch = pa.record_batch(
            [pa.array(["first", "second"], type=pa.string())],
            names=["api_explanation"],
        )
        av = _build_av_sft_cols(batch, "actor <INJECT>")
        self.assertEqual(av["prompt"].to_pylist()[0][0]["content"], "actor <INJECT>")
        self.assertEqual(
            av["response"].to_pylist(),
            [wrap_explanation("first"), wrap_explanation("second")],
        )
        rl = _build_rl_cols(batch, "actor <INJECT>")
        self.assertEqual(list(rl), ["prompt"])
        self.assertEqual(len(rl["prompt"]), 2)

        tokenizer = _Tokenizer()
        ar = _build_ar_sft_cols(batch, "prefix {explanation} suffix", [8, 9], tokenizer)
        self.assertEqual(
            ar["prompt"].to_pylist(),
            ["prefix first suffix", "prefix second suffix"],
        )
        self.assertTrue(all(not add_special for _, add_special in tokenizer.prompts))

        empty = pa.record_batch(
            [pa.array([], type=pa.string())], names=["api_explanation"]
        )
        self.assertEqual(len(_build_av_sft_cols(empty, "actor")["prompt"]), 0)
        self.assertEqual(len(_build_rl_cols(empty, "actor")["prompt"]), 0)
        self.assertEqual(
            len(_build_ar_sft_cols(empty, "{explanation} suffix", [8, 9], tokenizer)["prompt"]),
            0,
        )

    def test_critic_builder_fails_on_bpe_suffix_mismatch(self):
        batch = pa.record_batch(
            [pa.array(["explanation"], type=pa.string())],
            names=["api_explanation"],
        )
        with self.assertRaisesRegex(AssertionError, "does not end with expected suffix"):
            _build_ar_sft_cols(
                batch,
                "{explanation} suffix",
                [8, 9],
                _Tokenizer(bad_tail=True),
            )


class Stage3MainTest(unittest.TestCase):
    def setUp(self):
        self.storage = LocalStorage()
        self.checkpoints = checkpoints_for_depths(DEPTHS, 24)

    def _write_multi_input(
        self,
        path,
        *,
        row_count=2,
        include_api=True,
        include_debug=True,
        omit_column=None,
        meta=None,
    ):
        fields = [
            ("n_raw_tokens", pa.int64()),
            ("detokenized_text_truncated", pa.string()),
            *[(cp.column_name, pa.list_(pa.float32(), 2)) for cp in self.checkpoints],
            ("doc_id", pa.string()),
        ]
        if include_api:
            fields.append(("api_explanation", pa.string()))
        schema = pa.schema([field for field in fields if field[0] != omit_column])
        rows = {
            "n_raw_tokens": [51 + i for i in range(row_count)],
            "detokenized_text_truncated": [f"text-{i}" for i in range(row_count)],
            "doc_id": [f"doc-{i}" for i in range(row_count)],
        }
        for checkpoint in self.checkpoints:
            rows[checkpoint.column_name] = [
                [float(checkpoint.depth), float(checkpoint.depth + i + 0.5)]
                for i in range(row_count)
            ]
        if include_api:
            rows["api_explanation"] = [f"explanation-{i}" for i in range(row_count)]
        rows.pop(omit_column, None)
        pq.write_table(pa.table(rows, schema=schema), path)
        if meta is None:
            meta = NLADatasetMeta(
                "base_multi",
                "base",
                row_count,
                _extraction(self.checkpoints),
            )
        write_sidecar(self.storage, str(path), meta)
        return meta

    def _write_legacy_input(self, path):
        checkpoint = checkpoint_for_depth(11)
        schema = stage0_schema(2, [checkpoint], legacy_single_layer=True).append(
            pa.field("api_explanation", pa.string())
        )
        rows = {
            "n_raw_tokens": [51],
            "detokenized_text_truncated": ["legacy text"],
            "activation_vector": [[10.0, 11.0]],
            "activation_layer": [10],
            "doc_id": ["legacy-doc"],
            "api_explanation": ["legacy explanation"],
        }
        pq.write_table(pa.table(rows, schema=schema), path)
        meta = NLADatasetMeta(
            "base_legacy",
            "base",
            1,
            _extraction([checkpoint], layer_index=10),
        )
        write_sidecar(self.storage, str(path), meta)

    def _run(self, input_path, output_path, stage, *, depth="unset", debug=True, **patches):
        tokenizer = patches.pop("tokenizer", _Tokenizer())
        token_meta = patches.pop("token_meta", _token_meta())
        argv = [
            "stage3_build",
            "--input", str(input_path),
            "--stage", stage,
            "--output", str(output_path),
            "--keep-debug-metadata" if debug else "--no-keep-debug-metadata",
        ]
        if depth != "unset":
            argv += ["--checkpoint-depth", str(depth)]
        for key, value in patches.items():
            argv += [f"--{key.replace('_', '-')}", value]
        with (
            patch.object(sys, "argv", argv),
            patch.object(stage3_build, "load_tokenizer", return_value=tokenizer),
            patch.object(stage3_build, "build_token_meta", return_value=token_meta),
            patch.object(stage3_build, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
        ):
            stage3_build.main()
        return tokenizer

    def test_materializes_embedding_and_final_block_for_every_training_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "multi.parquet"
            self._write_multi_input(input_path)

            for depth in (0, 24):
                for stage in ("av_sft", "ar_sft", "rl"):
                    with self.subTest(depth=depth, stage=stage):
                        debug = depth == 0
                        output = Path(temp_dir) / f"{stage}_{depth}" / "data.parquet"
                        tokenizer = self._run(
                            input_path,
                            output,
                            stage,
                            depth=depth,
                            debug=debug,
                        )
                        table = pq.read_table(output)
                        self.assertEqual(table.num_rows, 2)
                        expected_name = "embedding" if depth == 0 else "block_24"
                        self.assertEqual(
                            table.column("activation_checkpoint").to_pylist(),
                            [expected_name, expected_name],
                        )
                        self.assertEqual(table.column("activation_depth").to_pylist(), [depth, depth])
                        expected_layer = None if depth == 0 else 23
                        self.assertEqual(
                            table.column("activation_layer").to_pylist(),
                            [expected_layer, expected_layer],
                        )
                        self.assertEqual(
                            table.column("activation_vector")[0].as_py(),
                            [float(depth), float(depth + 0.5)],
                        )
                        self.assertEqual(
                            "detokenized_text_truncated" in table.column_names,
                            debug,
                        )
                        if stage == "av_sft":
                            self.assertEqual(
                                table.column("response").to_pylist()[0],
                                wrap_explanation("explanation-0"),
                            )
                            self.assertIn(
                                "<INJECT>",
                                table.column("prompt").to_pylist()[0][0]["content"],
                            )
                        elif stage == "ar_sft":
                            self.assertIn(
                                "explanation-0", table.column("prompt").to_pylist()[0]
                            )
                            self.assertEqual(len(tokenizer.prompts), 2)
                        else:
                            self.assertNotIn("response", table.column_names)

                        meta = read_sidecar(self.storage, str(output))
                        self.assertEqual(meta.stage, stage)
                        self.assertEqual(meta.extraction.checkpoints, [checkpoint_for_depth(depth)])
                        self.assertEqual(meta.extraction.layer_index, expected_layer)
                        self.assertEqual(meta.parent_datasets, ["base_multi"])
                        self.assertIn(expected_name, meta.dataset_id)

    def test_materializes_joint_actor_bundle_and_ordered_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "multi.parquet"
            output_path = Path(temp_dir) / "joint" / "av_sft.parquet"
            self._write_multi_input(input_path)

            self._run(input_path, output_path, "av_sft", debug=False)

            table = pq.read_table(output_path)
            activation_columns = [
                checkpoint.column_name for checkpoint in self.checkpoints
            ]
            self.assertTrue(all(column in table.column_names for column in activation_columns))
            self.assertNotIn("activation_vector", table.column_names)
            self.assertNotIn("activation_checkpoint", table.column_names)
            self.assertNotIn("activation_depth", table.column_names)
            self.assertEqual(
                table.column("activation_embedding")[0].as_py(),
                [0.0, 0.5],
            )
            self.assertEqual(
                table.column("activation_block_24")[0].as_py(),
                [24.0, 24.5],
            )
            prompt = table.column("prompt")[0].as_py()[0]["content"]
            self.assertEqual(prompt.count("<INJECT>"), len(self.checkpoints))
            self.assertLess(prompt.index("Embedding checkpoint"), prompt.index("Block 24 checkpoint"))

            meta = read_sidecar(self.storage, str(output_path))
            self.assertEqual(meta.extraction.checkpoints, self.checkpoints)
            self.assertIsNone(meta.extraction.layer_index)
            self.assertIn("joint", meta.dataset_id)
            self.assertEqual(
                meta.prompt_templates["actor"].count("{injection_char}"),
                len(self.checkpoints),
            )

    def test_materializes_joint_rl_bundle_without_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "multi.parquet"
            output_path = Path(temp_dir) / "joint" / "rl.parquet"
            self._write_multi_input(input_path, include_api=False)

            self._run(input_path, output_path, "rl", debug=False)

            table = pq.read_table(output_path)
            self.assertNotIn("response", table.column_names)
            self.assertNotIn("activation_vector", table.column_names)
            self.assertEqual(
                [
                    column
                    for column in table.column_names
                    if column.startswith("activation_")
                ],
                [checkpoint.column_name for checkpoint in self.checkpoints],
            )
            prompt = table.column("prompt")[0].as_py()[0]["content"]
            self.assertEqual(prompt.count("<INJECT>"), len(self.checkpoints))
            meta = read_sidecar(self.storage, str(output_path))
            self.assertEqual(meta.stage, "rl")
            self.assertEqual(meta.extraction.checkpoints, self.checkpoints)

    def test_legacy_input_preserves_legacy_provenance_and_accepts_matching_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "legacy.parquet"
            output_path = Path(temp_dir) / "av.parquet"
            self._write_legacy_input(input_path)
            self._run(input_path, output_path, "av_sft", depth=11, debug=True)

            table = pq.read_table(output_path)
            self.assertNotIn("activation_checkpoint", table.column_names)
            self.assertNotIn("activation_depth", table.column_names)
            self.assertEqual(table.column("activation_layer").to_pylist(), [10])
            self.assertEqual(table.column("activation_vector")[0].as_py(), [10.0, 11.0])
            meta = read_sidecar(self.storage, str(output_path))
            self.assertEqual(meta.dataset_id, "av_sft_legacy")

    def test_main_rejects_templates_and_input_contract_violations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "multi.parquet"
            self._write_multi_input(input_path)
            output = Path(temp_dir) / "out.parquet"

            cases = [
                ("rl", {"actor_template": "missing"}, "must contain exactly 7"),
                (
                    "ar_sft",
                    {"critic_template": "missing", "depth": 0},
                    "must contain '{explanation}'",
                ),
            ]
            for stage, kwargs, message in cases:
                with self.subTest(stage=stage, message=message):
                    depth = kwargs.pop("depth", "unset")
                    with self.assertRaisesRegex(AssertionError, message):
                        self._run(input_path, output, stage, depth=depth, **kwargs)

            wrong_stage = replace(
                read_sidecar(self.storage, str(input_path)), stage="rl"
            )
            write_sidecar(
                self.storage,
                str(input_path),
                replace(
                    wrong_stage,
                    tokens=_token_meta(),
                    prompt_templates={
                        "actor": " ".join(
                            "{injection_char}" for _ in self.checkpoints
                        )
                    },
                ),
            )
            with self.assertRaisesRegex(AssertionError, "expected stage=base"):
                self._run(input_path, output, "rl", depth=0)

            norm_meta = replace(
                wrong_stage,
                stage="base",
                extraction=replace(wrong_stage.extraction, norm="unit"),
                tokens=None,
                prompt_templates={},
            )
            write_sidecar(self.storage, str(input_path), norm_meta)
            with self.assertRaisesRegex(AssertionError, "expected raw vectors"):
                self._run(input_path, output, "rl", depth=0)

    def test_main_rejects_missing_selection_api_provenance_and_empty_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.parquet"

            input_path = Path(temp_dir) / "multi.parquet"
            self._write_multi_input(input_path)
            with self.assertRaisesRegex(AssertionError, "joint AR/reward"):
                self._run(input_path, output, "ar_sft")

            no_api = Path(temp_dir) / "no_api.parquet"
            self._write_multi_input(no_api, include_api=False)
            with self.assertRaisesRegex(AssertionError, "requires api_explanation"):
                self._run(no_api, output, "av_sft", depth=0)

            missing_doc = Path(temp_dir) / "missing_doc.parquet"
            self._write_multi_input(missing_doc, omit_column="doc_id")
            with self.assertRaisesRegex(AssertionError, "missing required provenance"):
                self._run(missing_doc, output, "rl", depth=0)

            missing_activation = Path(temp_dir) / "missing_activation.parquet"
            self._write_multi_input(missing_activation, omit_column="activation_block_24")
            with self.assertRaisesRegex(AssertionError, "input has no 'activation_block_24'"):
                self._run(missing_activation, output, "rl", depth=24)

            empty = Path(temp_dir) / "empty.parquet"
            self._write_multi_input(empty, row_count=0)
            with self.assertRaisesRegex(AssertionError, "input parquet is empty"):
                self._run(empty, output, "rl", depth=0)


if __name__ == "__main__":
    unittest.main()
