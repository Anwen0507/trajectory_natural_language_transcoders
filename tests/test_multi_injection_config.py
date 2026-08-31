import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from nla.config import load_nla_config, write_model_sidecar


class _Tokenizer:
    unk_token_id = -1

    def encode(self, text, *, add_special_tokens):
        del text, add_special_tokens
        return [8]

    def apply_chat_template(self, *_args, **_kwargs):
        return [1, 7, 8, 9, 2, 7, 8, 9, 3]


class MultiInjectionConfigTest(unittest.TestCase):
    @staticmethod
    def _write_sidecar(path, *, checkpoints, layer_index=None):
        extraction = {
            "d_model": 4,
            "injection_scale": 10.0,
        }
        if checkpoints is not None:
            extraction["checkpoints"] = checkpoints
        if layer_index is not None:
            extraction["layer_index"] = layer_index
        path.write_text(yaml.safe_dump({
            "kind": "nla_dataset",
            "schema_version": 2,
            "extraction": extraction,
            "tokens": {
                "injection_char": "x",
                "injection_token_id": 8,
                "injection_left_neighbor_id": 7,
                "injection_right_neighbor_id": 9,
            },
            "prompt_templates": {
                "actor": "<concept>{injection_char}</concept>",
            },
        }))

    def test_dataset_contract_loads_and_model_sidecar_preserves_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet = Path(temp_dir) / "joint.parquet"
            sidecar = parquet.with_name(parquet.name + ".nla_meta.yaml")
            sidecar.write_text(yaml.safe_dump({
                "kind": "nla_dataset",
                "schema_version": 2,
                "extraction": {
                    "d_model": 4,
                    "checkpoints": [
                        {"name": "embedding", "depth": 0},
                        {"name": "block_24", "depth": 24},
                    ],
                },
                "tokens": {
                    "injection_char": "x",
                    "injection_token_id": 8,
                    "injection_left_neighbor_id": 7,
                    "injection_right_neighbor_id": 9,
                },
                "prompt_templates": {
                    "actor": (
                        "first <concept>{injection_char}</concept> "
                        "second <concept>{injection_char}</concept>"
                    ),
                },
            }))

            cfg = load_nla_config(str(parquet), _Tokenizer())
            self.assertEqual(cfg.activation_checkpoint_names, ("embedding", "block_24"))
            self.assertEqual(cfg.activation_checkpoint_depths, (0, 24))
            self.assertEqual(cfg.num_injection_sites, 2)

            model_dir = Path(temp_dir) / "model"
            write_model_sidecar(
                str(model_dir),
                cfg,
                role="actor",
                stage="sft",
                base_checkpoint="base",
                trained_on=[str(parquet)],
                parent_checkpoints=["base"],
                created_by="test",
            )
            written = yaml.safe_load((model_dir / "nla_meta.yaml").read_text())
            self.assertEqual(
                written["extraction"]["checkpoints"],
                [
                    {"name": "embedding", "depth": 0},
                    {"name": "block_24", "depth": 24},
                ],
            )

    def test_legacy_model_sidecar_implies_one_site(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            (model_dir / "nla_meta.yaml").write_text(yaml.safe_dump({
                "kind": "nla_model",
                "schema_version": 2,
                "d_model": 4,
                "extraction": {"injection_scale": 10.0},
                "tokens": {
                    "injection_char": "x",
                    "injection_token_id": 8,
                    "injection_left_neighbor_id": 7,
                    "injection_right_neighbor_id": 9,
                },
                "prompt_templates": {
                    "actor": "<concept>{injection_char}</concept>",
                },
            }))
            tokenizer = _Tokenizer()
            tokenizer.apply_chat_template = lambda *_args, **_kwargs: [7, 8, 9]
            cfg = load_nla_config(str(model_dir), tokenizer)
            self.assertEqual(cfg.num_injection_sites, 1)
            self.assertEqual(cfg.activation_checkpoint_names, ("activation",))

    def test_legacy_layer_index_maps_to_completed_block_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet = Path(temp_dir) / "legacy.parquet"
            sidecar = parquet.with_name(parquet.name + ".nla_meta.yaml")
            self._write_sidecar(sidecar, checkpoints=None, layer_index=3)
            tokenizer = _Tokenizer()
            tokenizer.apply_chat_template = lambda *_args, **_kwargs: [7, 8, 9]

            cfg = load_nla_config(str(parquet), tokenizer)

        self.assertEqual(cfg.activation_checkpoint_names, ("block_04",))
        self.assertEqual(cfg.activation_checkpoint_depths, (4,))

    def test_rejects_duplicate_names_and_unsorted_or_duplicate_depths(self):
        cases = [
            (
                [
                    {"name": "same", "depth": 0},
                    {"name": "same", "depth": 4},
                ],
                "duplicate activation checkpoint names",
            ),
            (
                [
                    {"name": "late", "depth": 8},
                    {"name": "early", "depth": 4},
                ],
                "depths must be sorted and unique",
            ),
            (
                [
                    {"name": "first", "depth": 4},
                    {"name": "second", "depth": 4},
                ],
                "depths must be sorted and unique",
            ),
        ]
        for checkpoints, message in cases:
            with self.subTest(checkpoints=checkpoints):
                with tempfile.TemporaryDirectory() as temp_dir:
                    parquet = Path(temp_dir) / "bad.parquet"
                    sidecar = parquet.with_name(parquet.name + ".nla_meta.yaml")
                    self._write_sidecar(sidecar, checkpoints=checkpoints)
                    with self.assertRaisesRegex(AssertionError, message):
                        load_nla_config(str(parquet), _Tokenizer())

    def test_model_sidecar_rejects_mismatched_checkpoint_tuple_lengths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet = Path(temp_dir) / "joint.parquet"
            sidecar = parquet.with_name(parquet.name + ".nla_meta.yaml")
            self._write_sidecar(
                sidecar,
                checkpoints=[
                    {"name": "embedding", "depth": 0},
                    {"name": "block_24", "depth": 24},
                ],
            )
            cfg = load_nla_config(str(parquet), _Tokenizer())
            bad_cfg = replace(cfg, activation_checkpoint_depths=(0,))
            with self.assertRaises(ValueError):
                write_model_sidecar(
                    str(Path(temp_dir) / "model"),
                    bad_cfg,
                    role="actor",
                    stage="sft",
                    base_checkpoint="base",
                    trained_on=[],
                    parent_checkpoints=[],
                    created_by="test",
                )


if __name__ == "__main__":
    unittest.main()
