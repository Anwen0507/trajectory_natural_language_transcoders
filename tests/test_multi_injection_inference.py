import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch import nn

import nla_inference


class _ConfigTokenizer:
    unk_token_id = -1

    def __init__(self, ids):
        self.ids = list(ids)

    def encode(self, _text, *, add_special_tokens):
        return [8]

    def apply_chat_template(self, *_args, **_kwargs):
        return list(self.ids)


def _sidecar(*, checkpoints=None, layer_index=None, template=None, scale=5.0):
    extraction = {"injection_scale": scale}
    if checkpoints is not None:
        extraction["checkpoints"] = checkpoints
    if layer_index is not None:
        extraction["layer_index"] = layer_index
    if template is None:
        count = len(checkpoints) if checkpoints else 1
        template = " ".join(["<concept>{injection_char}</concept>"] * count)
    return {
        "kind": "nla_model",
        "schema_version": 2,
        "d_model": 3,
        "extraction": extraction,
        "tokens": {
            "injection_char": "x",
            "injection_token_id": 8,
            "injection_left_neighbor_id": 7,
            "injection_right_neighbor_id": 9,
        },
        "prompt_templates": {"actor": template},
    }


def _write_sidecar(directory, contents):
    path = Path(directory) / "nla_meta.yaml"
    path.write_text(yaml.safe_dump(contents))


def _config(*, sites=2):
    names = ("embedding", "block_04") if sites == 2 else ("embedding",)
    depths = (0, 4) if sites == 2 else (0,)
    return nla_inference.NLAConfig(
        d_model=3,
        injection_char="x",
        injection_token_id=8,
        injection_left_neighbor_id=7,
        injection_right_neighbor_id=9,
        actor_prompt_template=" ".join(
            ["<concept>{injection_char}</concept>"] * sites
        ),
        activation_checkpoint_names=names,
        activation_checkpoint_depths=depths,
        injection_scale=5.0,
    )


class StandaloneConfigTest(unittest.TestCase):
    def test_loads_ordered_multi_checkpoint_contract(self):
        checkpoints = [
            {"name": "embedding", "depth": 0},
            {"name": "block_04", "depth": 4},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_sidecar(temp_dir, _sidecar(checkpoints=checkpoints))
            cfg = nla_inference.load_nla_config(
                temp_dir,
                _ConfigTokenizer([1, 7, 8, 9, 2, 7, 8, 9, 3]),
            )

        self.assertEqual(cfg.activation_checkpoint_names, ("embedding", "block_04"))
        self.assertEqual(cfg.activation_checkpoint_depths, (0, 4))
        self.assertEqual(cfg.num_injection_sites, 2)
        self.assertEqual(cfg.injection_scale, 5.0)

    def test_legacy_layer_and_unknown_layer_fallbacks(self):
        cases = [
            (3, ("block_04",), (4,)),
            (None, ("activation",), (None,)),
        ]
        for layer_index, names, depths in cases:
            with self.subTest(layer_index=layer_index):
                with tempfile.TemporaryDirectory() as temp_dir:
                    _write_sidecar(
                        temp_dir,
                        _sidecar(layer_index=layer_index),
                    )
                    cfg = nla_inference.load_nla_config(
                        temp_dir, _ConfigTokenizer([7, 8, 9])
                    )
                self.assertEqual(cfg.activation_checkpoint_names, names)
                self.assertEqual(cfg.activation_checkpoint_depths, depths)

    def test_rejects_duplicate_names_and_bad_depth_order(self):
        cases = [
            (
                [{"name": "same", "depth": 0}, {"name": "same", "depth": 4}],
                "duplicate activation checkpoint names",
            ),
            (
                [{"name": "late", "depth": 4}, {"name": "early", "depth": 0}],
                "depths must be sorted and unique",
            ),
            (
                [{"name": "a", "depth": 4}, {"name": "b", "depth": 4}],
                "depths must be sorted and unique",
            ),
        ]
        for checkpoints, message in cases:
            with self.subTest(checkpoints=checkpoints):
                with tempfile.TemporaryDirectory() as temp_dir:
                    _write_sidecar(temp_dir, _sidecar(checkpoints=checkpoints))
                    with self.assertRaisesRegex(AssertionError, message):
                        nla_inference.load_nla_config(
                            temp_dir, _ConfigTokenizer([7, 8, 9])
                        )

    def test_rejects_count_edge_and_neighbor_drift_at_any_site(self):
        checkpoints = [
            {"name": "embedding", "depth": 0},
            {"name": "block_04", "depth": 4},
        ]
        cases = [
            ([1, 7, 8, 9, 2], "appears 1×"),
            ([8, 9, 1, 7, 8, 9], None),
            ([1, 6, 8, 9, 2, 7, 8, 9, 3], "left neighbor drift"),
            ([1, 7, 8, 6, 2, 7, 8, 9, 3], "right neighbor drift"),
        ]
        for ids, message in cases:
            with self.subTest(ids=ids):
                with tempfile.TemporaryDirectory() as temp_dir:
                    _write_sidecar(temp_dir, _sidecar(checkpoints=checkpoints))
                    expectation = (
                        self.assertRaisesRegex(AssertionError, message)
                        if message is not None
                        else self.assertRaises(AssertionError)
                    )
                    with expectation:
                        nla_inference.load_nla_config(
                            temp_dir, _ConfigTokenizer(ids)
                        )


class StandaloneInjectionTest(unittest.TestCase):
    def test_injects_multiple_vectors_in_row_major_order_and_clones(self):
        ids = torch.tensor([
            [1, 7, 8, 9, 2, 7, 8, 9, 3],
            [4, 7, 8, 9, 5, 7, 8, 9, 6],
        ])
        embeddings = torch.full((2, 9, 2), -1.0, dtype=torch.float64)
        vectors = torch.arange(8, dtype=torch.float32).reshape(4, 2)

        actual = nla_inference.inject_at_marked_positions(
            ids, embeddings, vectors, 8, 7, 9
        )

        for vector_index, (batch, position) in enumerate(
            [(0, 2), (0, 6), (1, 2), (1, 6)]
        ):
            torch.testing.assert_close(
                actual[batch, position], vectors[vector_index].double()
            )
        self.assertTrue(torch.all(embeddings == -1))

    def test_ignores_edge_and_bad_neighbor_markers_then_fails_on_count(self):
        ids = torch.tensor([[8, 7, 8, 9, 8, 6, 9, 8]])
        embeddings = torch.zeros(1, 8, 2)
        vector = torch.ones(1, 2)
        actual = nla_inference.inject_at_marked_positions(
            ids, embeddings, vector, 8, 7, 9
        )
        torch.testing.assert_close(actual[0, 2], vector[0])
        with self.assertRaisesRegex(AssertionError, "found 1.*expected 2"):
            nla_inference.inject_at_marked_positions(
                ids, embeddings, torch.ones(2, 2), 8, 7, 9
            )


class _ClientTokenizer:
    def __init__(self, ids):
        self.ids = list(ids)
        self.messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages.append(messages)
        return list(self.ids)


def _client(*, sites=2):
    client = object.__new__(nla_inference.NLAClient)
    client.cfg = _config(sites=sites)
    ids = [1, 7, 8, 9, 2]
    if sites == 2:
        ids += [7, 8, 9, 3]
    client.tokenizer = _ClientTokenizer(ids)
    client.embed = nn.Embedding(16, 3)
    with torch.no_grad():
        for token_id in range(16):
            client.embed.weight[token_id] = torch.tensor(
                [token_id, token_id + 0.1, token_id + 0.2]
            )
    client.embed_scale = 2.0
    return client


class StandaloneClientTest(unittest.TestCase):
    def test_build_embeds_injects_default_bundle_independently(self):
        client = _client(sites=2)
        vectors = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])

        embeds, prompt_length = client._build_embeds(vectors, None)

        self.assertEqual(prompt_length, 9)
        self.assertEqual(embeds.shape, (9, 3))
        np.testing.assert_allclose(embeds[2], [3.0, 4.0, 0.0])
        np.testing.assert_allclose(embeds[6], [0.0, 0.0, 5.0])
        np.testing.assert_allclose(embeds[0], [2.0, 2.2, 2.4])
        content = client.tokenizer.messages[0][0]["content"]
        self.assertEqual(content.count("x"), 2)

    def test_build_embeds_custom_prompt_requires_exact_marker_count(self):
        client = _client(sites=2)
        vectors = torch.ones(2, 3)
        embeds, _ = client._build_embeds(
            vectors, "early <INJECT> late <INJECT>"
        )
        self.assertEqual(embeds.shape, (9, 3))
        content = client.tokenizer.messages[-1][0]["content"]
        self.assertEqual(content, "early x late x")

        for prompt in ("none", "one <INJECT>", "<INJECT> <INJECT> <INJECT>"):
            with self.subTest(prompt=prompt):
                with self.assertRaisesRegex(
                    AssertionError, "markers, expected 2"
                ):
                    client._build_embeds(vectors, prompt)

    def test_build_embeds_rejects_nonfinite_activation(self):
        client = _client(sites=2)
        vectors = torch.tensor([[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0]])
        with self.assertRaisesRegex(AssertionError, "NaN/Inf"):
            client._build_embeds(vectors, None)

    def test_generate_accepts_joint_shape_and_extracts_or_returns_raw_text(self):
        client = _client(sites=2)
        client._build_embeds = Mock(return_value=(np.zeros((2, 3)), 2))
        client._sglang_generate = Mock(
            return_value={"text": "<explanation> decoded </explanation>"}
        )
        activation = np.ones((2, 3), np.float32)

        self.assertEqual(client.generate(activation, temperature=0.2), "decoded")
        self.assertEqual(
            client.generate(activation, extract_explanation=False),
            "<explanation> decoded </explanation>",
        )
        self.assertEqual(tuple(client._build_embeds.call_args.args[0].shape), (2, 3))

    def test_generate_promotes_flat_legacy_vector(self):
        client = _client(sites=1)
        client._build_embeds = Mock(return_value=(np.zeros((2, 3)), 2))
        client._sglang_generate = Mock(return_value={"text": "untagged"})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = client.generate([1.0, 2.0, 3.0])
        self.assertEqual(result, "untagged")
        self.assertIn("WARNING: no <explanation> tags", output.getvalue())
        self.assertEqual(tuple(client._build_embeds.call_args.args[0].shape), (1, 3))

    def test_generate_rejects_every_wrong_bundle_shape(self):
        client = _client(sites=2)
        bad = [
            np.ones(3),
            np.ones((1, 3)),
            np.ones((2, 2)),
            np.ones((2, 3, 1)),
        ]
        for activation in bad:
            with self.subTest(shape=activation.shape):
                with self.assertRaisesRegex(
                    AssertionError, "activation bundle shape"
                ):
                    client.generate(activation)


class _CLIClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []

    def generate(self, activation, **kwargs):
        self.calls.append((np.asarray(activation).copy(), kwargs))
        return "decoded"


class StandaloneCLITest(unittest.TestCase):
    def test_cli_reads_named_columns_in_sidecar_order(self):
        fake = _CLIClient(_config(sites=2))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "joint.parquet"
            pq.write_table(
                pa.table({
                    "activation_block_04": pa.array(
                        [[4.0, 5.0, 6.0]], type=pa.list_(pa.float32(), 3)
                    ),
                    "activation_embedding": pa.array(
                        [[1.0, 2.0, 3.0]], type=pa.list_(pa.float32(), 3)
                    ),
                }),
                path,
            )
            argv = ["nla_inference", "model", "--parquet", str(path), "--n", "1"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(nla_inference, "NLAClient", return_value=fake),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                nla_inference._main()

        np.testing.assert_array_equal(
            fake.calls[0][0],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        self.assertIn("checkpoint ||v||=", output.getvalue())

    def test_cli_reads_legacy_column_for_single_site_actor(self):
        fake = _CLIClient(_config(sites=1))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.parquet"
            pq.write_table(
                pa.table({
                    "activation_vector": pa.array(
                        [[1.0, 2.0, 3.0]], type=pa.list_(pa.float32(), 3)
                    ),
                }),
                path,
            )
            with (
                patch.object(
                    sys,
                    "argv",
                    ["nla_inference", "model", "--parquet", str(path)],
                ),
                patch.object(nla_inference, "NLAClient", return_value=fake),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                nla_inference._main()
        self.assertEqual(fake.calls[0][0].shape, (1, 3))

    def test_cli_smoke_builds_full_random_bundle(self):
        fake = _CLIClient(_config(sites=2))
        with (
            patch.object(sys, "argv", ["nla_inference", "model"]),
            patch.object(nla_inference, "NLAClient", return_value=fake),
            patch.object(np.random, "randn", return_value=np.ones((2, 3))) as randn,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            nla_inference._main()
        randn.assert_called_once_with(2, 3)
        self.assertEqual(fake.calls[0][0].shape, (2, 3))

    def test_cli_rejects_legacy_column_for_multi_site_actor_and_missing_named_column(self):
        fake = _CLIClient(_config(sites=2))
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy = Path(temp_dir) / "legacy.parquet"
            pq.write_table(
                pa.table({
                    "activation_vector": pa.array(
                        [[1.0, 2.0, 3.0]], type=pa.list_(pa.float32(), 3)
                    )
                }),
                legacy,
            )
            missing = Path(temp_dir) / "missing.parquet"
            pq.write_table(
                pa.table({
                    "activation_embedding": pa.array(
                        [[1.0, 2.0, 3.0]], type=pa.list_(pa.float32(), 3)
                    )
                }),
                missing,
            )
            cases = [
                (legacy, "parquet has one activation_vector"),
                (missing, "missing actor checkpoint columns.*activation_block_04"),
            ]
            for path, message in cases:
                with self.subTest(path=path.name):
                    with (
                        patch.object(
                            sys,
                            "argv",
                            ["nla_inference", "model", "--parquet", str(path)],
                        ),
                        patch.object(
                            nla_inference, "NLAClient", return_value=fake
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                        self.assertRaisesRegex(AssertionError, message),
                    ):
                        nla_inference._main()


if __name__ == "__main__":
    unittest.main()
