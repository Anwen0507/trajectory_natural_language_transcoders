import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pyarrow.parquet as pq
import torch
from datasets import Dataset

from nla.datagen import stage0_extract
from nla.datagen.extractors import MultiExtractionResult
from nla.datagen.sidecar import read_sidecar
from nla.datagen.stage0_extract import _MIN_POSITION, _sample_positions
from nla.datagen.storage import LocalStorage

DEPTHS = [0, 4, 8, 12, 16, 20, 24]


class _Tokenizer:
    all_special_ids: ClassVar[list[int]] = [0, 99]
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids, *, skip_special_tokens):
        self.decode_calls.append((list(token_ids), skip_special_tokens))
        return "tokens:" + ",".join(map(str, token_ids))


class _Extractor:
    d_model = 2
    num_layers = 24

    def __init__(self, token_ids_by_text):
        self.tokenizer = _Tokenizer()
        self.token_ids_by_text = token_ids_by_text
        self.calls = []
        self.result_count_delta = 0
        self.omit_depth = None
        self.bad_shape_depth = None

    def extract_many(self, texts, checkpoint_depths):
        self.calls.append((list(texts), list(checkpoint_depths)))
        results = []
        count = max(0, len(texts) + self.result_count_delta)
        for text in texts[:count]:
            token_ids = list(self.token_ids_by_text[text])
            activations = {}
            for depth in checkpoint_depths:
                if depth == self.omit_depth:
                    continue
                width = 3 if depth == self.bad_shape_depth else self.d_model
                values = [
                    [float(depth * 1000 + position + component / 10) for component in range(width)]
                    for position in range(len(token_ids))
                ]
                activations[depth] = torch.tensor(values, dtype=torch.float32)
            results.append(MultiExtractionResult(activations, token_ids))
        return results


class Stage0MultiCheckpointTest(unittest.TestCase):
    def _run(self, extractor, dataset, output_path, target_args, *extra_args):
        constructed_with = []

        def factory(**kwargs):
            constructed_with.append(kwargs)
            return extractor

        argv = [
            "stage0_extract",
            "--base-model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--corpus", "example/corpus",
            "--corpus-length", str(len(dataset) if isinstance(dataset, Dataset) else 1),
            "--positions-per-doc", "3",
            "--chunk-size", "2",
            "--seed", "7",
            "--output", str(output_path),
            *target_args,
            *extra_args,
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(stage0_extract, "make_storage", return_value=LocalStorage()),
            patch.object(stage0_extract, "load_class", return_value=factory),
            patch.object(stage0_extract, "load_dataset", return_value=dataset),
            patch.object(stage0_extract, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
        ):
            stage0_extract.main()
        return constructed_with

    def test_samples_aligned_vectors_for_all_qwen_boundaries(self):
        long_ids = [1] * 55
        long_ids[51] = 99  # special token is never a candidate
        token_ids = {
            "long": long_ids,
            "short": [2] * _MIN_POSITION,
            "one-candidate": [3] * (_MIN_POSITION + 1),
        }
        extractor = _Extractor(token_ids)
        dataset = Dataset.from_dict({"text": list(token_ids)})

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "base.parquet"
            kwargs = self._run(
                extractor,
                dataset,
                output,
                ["--checkpoint-depths", *map(str, DEPTHS)],
                "--extractor-kwargs", '{"batch_size": 1}',
            )

            self.assertEqual(
                kwargs,
                [{"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "batch_size": 1}],
            )
            self.assertEqual(
                extractor.calls,
                [(["long", "short"], DEPTHS), (["one-candidate"], DEPTHS)],
            )
            table = pq.read_table(output)
            self.assertEqual(table.num_rows, 4)
            self.assertEqual(
                table.column_names,
                [
                    "n_raw_tokens",
                    "detokenized_text_truncated",
                    "activation_embedding",
                    "activation_block_04",
                    "activation_block_08",
                    "activation_block_12",
                    "activation_block_16",
                    "activation_block_20",
                    "activation_block_24",
                    "doc_id",
                ],
            )
            self.assertEqual(
                table.column("doc_id").to_pylist().count("example/corpus:train:0"),
                3,
            )
            self.assertEqual(
                table.column("doc_id").to_pylist().count("example/corpus:train:2"),
                1,
            )
            for row in table.to_pylist():
                position = row["n_raw_tokens"] - 1
                for depth, name in zip(
                    DEPTHS,
                    [
                        "embedding",
                        "block_04",
                        "block_08",
                        "block_12",
                        "block_16",
                        "block_20",
                        "block_24",
                    ],
                    strict=True,
                ):
                    actual = row[f"activation_{name}"]
                    self.assertAlmostEqual(actual[0], float(depth * 1000 + position), places=5)
                    self.assertAlmostEqual(actual[1], float(depth * 1000 + position + 0.1), places=3)
            self.assertTrue(all(skip_special for _, skip_special in extractor.tokenizer.decode_calls))

            meta = read_sidecar(LocalStorage(), str(output))
            self.assertEqual(meta.row_count, 4)
            self.assertEqual([cp.depth for cp in meta.extraction.checkpoints], DEPTHS)
            self.assertIsNone(meta.extraction.layer_index)
            self.assertEqual(meta.extraction.norm, "none")
            self.assertEqual(meta.extraction.corpus_slice, {"start": 0, "length": 3})
            self.assertIn("_C0-4-8-12-16-20-24_", meta.dataset_id)

    def test_legacy_layer_cli_still_writes_legacy_schema(self):
        extractor = _Extractor({"doc": [1] * (_MIN_POSITION + 1)})
        dataset = Dataset.from_dict({"text": ["doc"]})
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "base.parquet"
            self._run(extractor, dataset, output, ["--layer-index", "3"])

            table = pq.read_table(output)
            self.assertEqual(
                table.column_names,
                [
                    "n_raw_tokens",
                    "detokenized_text_truncated",
                    "activation_vector",
                    "activation_layer",
                    "doc_id",
                ],
            )
            self.assertEqual(table.column("activation_layer").to_pylist(), [3])
            self.assertEqual(table.column("activation_vector")[0].as_py(), [4050.0, 4050.10009765625])
            self.assertEqual(extractor.calls, [(["doc"], [4])])
            meta = read_sidecar(LocalStorage(), str(output))
            self.assertEqual(meta.extraction.layer_index, 3)
            self.assertEqual(meta.extraction.checkpoints[0].depth, 4)
            self.assertIn("_L3_", meta.dataset_id)

    def test_sampling_is_keyed_by_doc_and_handles_candidate_edges(self):
        ids = [1] * 60
        ids[52] = 99
        first = _sample_positions(ids, 5, {99}, "doc-a", 42)
        self.assertEqual(first, _sample_positions(ids, 5, {99}, "doc-a", 42))
        self.assertNotEqual(first, _sample_positions(ids, 5, {99}, "doc-b", 42))
        self.assertTrue(all(position >= _MIN_POSITION and position != 52 for position in first))
        self.assertEqual(_sample_positions([1] * _MIN_POSITION, 2, set(), "short", 1), [])
        self.assertCountEqual(
            _sample_positions([1] * (_MIN_POSITION + 2), 99, set(), "few", 1),
            [_MIN_POSITION, _MIN_POSITION + 1],
        )
        self.assertEqual(_sample_positions(ids, 0, set(), "zero", 1), [])

    def test_rejects_bad_cli_numeric_boundaries_and_checkpoint_specs(self):
        dataset = Dataset.from_dict({"text": ["doc"]})
        token_ids = {"doc": [1] * (_MIN_POSITION + 1)}
        cases = [
            (["--layer-index", "-1"], ["--positions-per-doc", "1"], "layer_index"),
            (["--checkpoint-depths", "0", "4", "4"], [], "sorted and unique"),
            (["--checkpoint-depths", "0", "25"], [], "exceed model depth"),
            (["--checkpoint-depths", "0"], ["--positions-per-doc", "0"], "positions_per_doc"),
            (["--checkpoint-depths", "0"], ["--chunk-size", "0"], "chunk_size"),
            (["--checkpoint-depths", "0"], ["--corpus-length", "0"], "corpus_length"),
            (["--checkpoint-depths", "0"], ["--corpus-start", "-1"], "corpus_start"),
            (
                ["--checkpoint-depths", "0"],
                ["--corpus-start", "1"],
                "exceeds dataset length",
            ),
        ]
        for target, overrides, message in cases:
            with self.subTest(target=target, overrides=overrides):
                extractor = _Extractor(token_ids)
                with tempfile.TemporaryDirectory() as temp_dir:
                    base_args = [
                        "stage0_extract",
                        "--base-model", "model",
                        "--corpus", "corpus",
                        "--corpus-length", "1",
                        "--positions-per-doc", "1",
                        "--chunk-size", "1",
                        "--output", str(Path(temp_dir) / "base.parquet"),
                        *target,
                        *overrides,
                    ]
                    with (
                        patch.object(sys, "argv", base_args),
                        patch.object(stage0_extract, "make_storage", return_value=LocalStorage()),
                        patch.object(
                            stage0_extract,
                            "load_class",
                            return_value=lambda _extractor=extractor, **_: _extractor,
                        ),
                        patch.object(stage0_extract, "load_dataset", return_value=dataset),
                        self.assertRaisesRegex(AssertionError, message),
                    ):
                        stage0_extract.main()

    def test_rejects_model_name_override_dataset_type_and_pad_leak(self):
        extractor = _Extractor({"doc": [1] * (_MIN_POSITION + 1)})
        dataset = Dataset.from_dict({"text": ["doc"]})
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "base.parquet"
            with self.assertRaisesRegex(AssertionError, "pass --base-model"):
                self._run(
                    extractor,
                    dataset,
                    output,
                    ["--checkpoint-depths", "0"],
                    "--extractor-kwargs", '{"model_name": "wrong"}',
                )

            with self.assertRaisesRegex(AssertionError, "expected a concrete Dataset"):
                self._run(extractor, object(), output, ["--checkpoint-depths", "0"])

            extractor.token_ids_by_text["doc"][_MIN_POSITION] = 0
            with self.assertRaisesRegex(AssertionError, "pad_token_id 0 found"):
                self._run(extractor, dataset, output, ["--checkpoint-depths", "0"])

    def test_pad_leak_check_is_disabled_when_pad_and_eos_are_the_same_token(self):
        token_ids = [1] * (_MIN_POSITION + 2)
        token_ids[_MIN_POSITION] = 99
        extractor = _Extractor({"doc": token_ids})
        extractor.tokenizer.pad_token_id = 99
        extractor.tokenizer.eos_token_id = 99
        dataset = Dataset.from_dict({"text": ["doc"]})
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "base.parquet"
            self._run(extractor, dataset, output, ["--checkpoint-depths", "0"])
            self.assertEqual(pq.read_table(output).num_rows, 1)

    def test_rejects_custom_extractor_contract_violations(self):
        dataset = Dataset.from_dict({"text": ["doc"]})
        token_ids = {"doc": [1] * (_MIN_POSITION + 1)}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "base.parquet"
            extractor = _Extractor(token_ids)
            extractor.result_count_delta = -1
            with self.assertRaisesRegex(AssertionError, "returned 0 results for 1 texts"):
                self._run(extractor, dataset, output, ["--checkpoint-depths", "0", "4"])

            extractor = _Extractor(token_ids)
            extractor.omit_depth = 4
            with self.assertRaisesRegex(AssertionError, "missing checkpoint depth 4"):
                self._run(extractor, dataset, output, ["--checkpoint-depths", "0", "4"])

            extractor = _Extractor(token_ids)
            extractor.bad_shape_depth = 4
            with self.assertRaisesRegex(AssertionError, r"expected \[51, 2\]"):
                self._run(extractor, dataset, output, ["--checkpoint-depths", "0", "4"])

            extractor = _Extractor(token_ids)
            extractor.d_model = 0
            with self.assertRaisesRegex(AssertionError, "d_model must be positive"):
                self._run(extractor, dataset, output, ["--checkpoint-depths", "0"])


if __name__ == "__main__":
    unittest.main()
