import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from nla.datagen import run_pipeline

DEPTHS = [0, 4, 8, 12, 16, 20, 24]
TRAIN_STAGES = ("av_sft", "ar_sft", "rl")


def _config(*, multi=True):
    cfg = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_dir": "/tmp/output",
        "storage_cls": "nla.datagen.storage.LocalStorage",
        "corpus": {
            "name": "example/corpus",
            "config": "subset",
            "split": "train",
            "start": 5,
            "length": 10,
            "text_column": "body",
        },
        "stage0": {
            "positions_per_doc": 2,
            "chunk_size": 4,
            "seed": 7,
            "extractor_kwargs": {"batch_size": 1},
        },
        "stage1": {
            "av_sft_frac": 0.2,
            "ar_sft_frac": 0.3,
            "rl_frac": 0.5,
            "seed": 8,
        },
        "stage2": {
            "provider_cls": "example.Provider",
            "provider_kwargs": {"model": "test"},
            "chunk_size": 9,
            "cache_from": ["a.parquet", "b.parquet"],
            "cache_storage_cls": "example.CacheStorage",
        },
        "stage3": {"keep_debug_metadata": False},
        "shuffle": {"enabled": True, "seed": 10},
    }
    if multi:
        cfg["checkpoint_depths"] = list(DEPTHS)
    else:
        cfg["layer_index"] = 10
    return cfg


class PipelineCommandTest(unittest.TestCase):
    def test_paths_and_training_paths_cover_legacy_multi_and_shuffle(self):
        paths = run_pipeline._paths("/work/out")
        self.assertEqual(paths["base"], "/work/out/base.parquet")
        self.assertEqual(paths["av_sft_raw"], "/work/out/splits/av_sft_raw.parquet")
        self.assertEqual(paths["rl_shuf"], "/work/out/rl_shuf.parquet")
        cfg = {"output_dir": "/work/out"}
        self.assertEqual(
            run_pipeline._training_path(cfg, paths, "ar_sft", 0),
            "/work/out/checkpoints/embedding/ar_sft.parquet",
        )
        self.assertEqual(
            run_pipeline._training_path(cfg, paths, "ar_sft", 4, shuffled=True),
            "/work/out/checkpoints/block_04/ar_sft_shuf.parquet",
        )
        self.assertEqual(
            run_pipeline._training_path(cfg, paths, "ar_sft", None),
            "/work/out/ar_sft.parquet",
        )

    def test_storage_args_and_subprocess_runner(self):
        self.assertEqual(
            run_pipeline._storage_args({"storage_cls": "Storage"}),
            ["--storage-cls", "Storage"],
        )
        self.assertEqual(
            run_pipeline._storage_args({
                "storage_cls": "Storage",
                "storage_kwargs": {"bucket": "test"},
            }),
            ["--storage-cls", "Storage", "--storage-kwargs", '{"bucket": "test"}'],
        )
        with (
            patch.object(run_pipeline.subprocess, "run") as subprocess_run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_pipeline._run(["python", "command"])
        subprocess_run.assert_called_once_with(["python", "command"], check=True)

    def test_stage0_multi_serial_command(self):
        cfg = _config(multi=True)
        paths = run_pipeline._paths(cfg["output_dir"])
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage0(cfg, paths)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "nla.datagen.stage0_extract"])
        checkpoint_index = command.index("--checkpoint-depths")
        self.assertEqual(command[checkpoint_index + 1 : checkpoint_index + 8], list(map(str, DEPTHS)))
        self.assertNotIn("--layer-index", command)
        self.assertEqual(command[command.index("--corpus-config") + 1], "subset")
        self.assertEqual(
            json.loads(command[command.index("--extractor-kwargs") + 1]),
            {"batch_size": 1},
        )

    def test_stage0_legacy_multigpu_command_and_optional_defaults(self):
        cfg = _config(multi=False)
        cfg["stage0"]["multigpu"] = True
        cfg["stage0"].pop("extractor_kwargs")
        cfg["corpus"].pop("config")
        cfg["corpus"].pop("text_column")
        paths = run_pipeline._paths(cfg["output_dir"])
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage0(cfg, paths)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "bash")
        self.assertTrue(command[1].endswith("scripts/datagen/stage0_multigpu.sh"))
        self.assertEqual(command[command.index("--layer-index") + 1], "10")
        self.assertEqual(command[command.index("--text-column") + 1], "text")
        self.assertNotIn("--checkpoint-depths", command)
        self.assertNotIn("--corpus-config", command)
        self.assertNotIn("--extractor-kwargs", command)

    def test_stage1_and_stage2_commands(self):
        cfg = _config()
        paths = run_pipeline._paths(cfg["output_dir"])
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage1(cfg, paths)
        self.assertEqual(run.call_count, 1)
        stage1 = run.call_args.args[0]
        self.assertEqual(stage1[2], "nla.datagen.stage1_split")
        self.assertEqual(stage1[stage1.index("--output-dir") + 1], "/tmp/output/splits")

        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage2(cfg, paths)
        self.assertEqual(run.call_count, 2)
        for command, side in zip(
            [entry.args[0] for entry in run.call_args_list],
            ("av_sft", "ar_sft"),
            strict=True,
        ):
            self.assertEqual(command[2], "nla.datagen.stage2_api_explain")
            self.assertEqual(command[command.index("--input") + 1], paths[f"{side}_raw"])
            self.assertEqual(command.count("--cache-from"), 2)
            self.assertIn("--cache-storage-cls", command)
            self.assertEqual(
                json.loads(command[command.index("--provider-kwargs") + 1]),
                {"model": "test"},
            )

        minimal = _config()
        minimal["stage2"] = {"provider_cls": "Provider", "chunk_size": 1}
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage2(minimal, paths)
        self.assertNotIn("--provider-kwargs", run.call_args_list[0].args[0])
        self.assertNotIn("--cache-storage-cls", run.call_args_list[0].args[0])

    def test_stage3_builds_joint_actor_files_and_fans_out_ar(self):
        cfg = _config(multi=True)
        cfg["stage3"].update({
            "actor_template": "\n".join(
                f"checkpoint-{depth}: {{injection_char}}" for depth in DEPTHS
            ),
            "critic_template": "critic {explanation}",
        })
        paths = run_pipeline._paths(cfg["output_dir"])
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage3(cfg, paths)
        self.assertEqual(run.call_count, 2 + len(DEPTHS))
        commands = [entry.args[0] for entry in run.call_args_list]

        for stage in ("av_sft", "rl"):
            matches = [
                command for command in commands
                if command[command.index("--stage") + 1] == stage
            ]
            self.assertEqual(len(matches), 1)
            command = matches[0]
            self.assertNotIn("--checkpoint-depth", command)
            self.assertIn("--actor-template", command)
            self.assertIn("--critic-template", command)
            self.assertEqual(
                command[command.index("--output") + 1],
                run_pipeline._training_path(cfg, paths, stage, None),
            )

        for depth in DEPTHS:
            matches = [
                command
                for command in commands
                if command[command.index("--stage") + 1] == "ar_sft"
                and command[command.index("--checkpoint-depth") + 1] == str(depth)
            ]
            self.assertEqual(len(matches), 1)
            command = matches[0]
            self.assertNotIn("--actor-template", command)
            self.assertIn("--critic-template", command)
            self.assertEqual(
                command[command.index("--output") + 1],
                run_pipeline._training_path(cfg, paths, "ar_sft", depth),
            )
            self.assertIn("--no-keep-debug-metadata", command)

    def test_stage3_legacy_and_shuffle_fanout(self):
        legacy = _config(multi=False)
        legacy["stage3"]["keep_debug_metadata"] = True
        paths = run_pipeline._paths(legacy["output_dir"])
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._stage3(legacy, paths)
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(
            "--checkpoint-depth" not in entry.args[0]
            and "--keep-debug-metadata" in entry.args[0]
            for entry in run.call_args_list
        ))

        cfg = _config(multi=True)
        paths = run_pipeline._paths(cfg["output_dir"])
        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._shuffle(cfg, paths)
        self.assertEqual(run.call_count, 2 + len(DEPTHS))
        first = run.call_args_list[0].args[0]
        self.assertEqual(first[first.index("--input") + 1], "/tmp/output/av_sft.parquet")
        self.assertEqual(first[first.index("--output") + 1], "/tmp/output/av_sft_shuf.parquet")

        with patch.object(run_pipeline, "_run") as run:
            run_pipeline._shuffle(legacy, run_pipeline._paths(legacy["output_dir"]))
        self.assertEqual(run.call_count, 3)
        last = run.call_args_list[-1].args[0]
        self.assertEqual(
            last[last.index("--output") + 1],
            "/tmp/output/rl_shuf.parquet",
        )


class PipelineMainTest(unittest.TestCase):
    def _run_main(self, cfg, *args, stage_functions=None, import_module=None):
        stage_functions = stage_functions or {
            key: Mock(name=f"stage_{key}") for key in ("0", "1", "2", "3", "shuffle")
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            argv = ["run_pipeline", "--config", str(config_path), *args]
            output = io.StringIO()
            contexts = [
                patch.object(sys, "argv", argv),
                patch.dict(run_pipeline._STAGES, stage_functions, clear=True),
                contextlib.redirect_stdout(output),
            ]
            if import_module is not None:
                contexts.append(patch.object(run_pipeline.importlib, "import_module", import_module))
            with contextlib.ExitStack() as stack:
                for context in contexts:
                    stack.enter_context(context)
                run_pipeline.main()
        return stage_functions, output.getvalue()

    def test_default_multi_run_applies_preset_runs_shuffle_and_uploads_every_output(self):
        upload = Mock()
        module = types.SimpleNamespace(upload=upload)
        cfg = _config()
        cfg.pop("base_model")
        cfg["model"] = "qwen05b"
        cfg["upload"] = {"fn": "fake.module.upload", "subdir": "run-1"}
        stages, stdout = self._run_main(
            cfg,
            import_module=Mock(return_value=module),
        )

        for key in ("0", "1", "2", "3", "shuffle"):
            self.assertEqual(stages[key].call_count, 1)
            resolved = stages[key].call_args.args[0]
            self.assertEqual(resolved["base_model"], "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertNotIn("layer_index", resolved)
        self.assertIn("checkpoint_depths=[0, 4, 8, 12, 16, 20, 24]", stdout)
        self.assertIn("rl: /tmp/output/rl_shuf.parquet", stdout)

        upload.assert_called_once()
        files, subdir = upload.call_args.args
        self.assertEqual(subdir, "run-1")
        self.assertEqual(len(files), 22)
        self.assertIn("/tmp/output/av_sft_shuf.parquet", files)
        self.assertIn("/tmp/output/rl_shuf.parquet.nla_meta.yaml", files)
        self.assertIn("/tmp/output/splits/ar_sft_explained.parquet", files)

    def test_explicit_stage_overrides_are_applied_before_preset_resolution(self):
        cfg = _config()
        cfg.pop("base_model")
        cfg["model"] = "qwen05b"
        stages, stdout = self._run_main(
            cfg,
            "--stages", "3",
            "--override",
            "checkpoint_depths=[0, 4]",
            "output_dir=/tmp/override",
            "stage0.positions_per_doc=9",
        )
        self.assertEqual(stages["3"].call_count, 1)
        resolved = stages["3"].call_args.args[0]
        self.assertEqual(resolved["checkpoint_depths"], [0, 4])
        self.assertEqual(resolved["output_dir"], "/tmp/override")
        self.assertEqual(resolved["stage0"]["positions_per_doc"], 9)
        self.assertNotIn("_shuf.parquet", stdout)
        self.assertEqual(stages["shuffle"].call_count, 0)

    def test_default_stage_list_omits_disabled_shuffle(self):
        cfg = _config(multi=False)
        cfg["shuffle"]["enabled"] = False
        stages, stdout = self._run_main(cfg)
        for key in ("0", "1", "2", "3"):
            self.assertEqual(stages[key].call_count, 1)
        self.assertEqual(stages["shuffle"].call_count, 0)
        self.assertNotIn("_shuf.parquet", stdout)

    def test_legacy_main_target_and_validation_errors(self):
        stages, stdout = self._run_main(_config(multi=False), "--stages", "3")
        self.assertEqual(stages["3"].call_count, 1)
        self.assertIn("layer=10", stdout)
        self.assertIn("rl: /tmp/output/rl.parquet", stdout)

        cases = [
            (_config() | {"layer_index": 3}, ["--stages", "3"], "exactly one"),
            (
                {key: value for key, value in _config().items() if key != "checkpoint_depths"},
                ["--stages", "3"],
                "exactly one",
            ),
            (_config() | {"checkpoint_depths": [4, 4]}, ["--stages", "3"], "sorted and unique"),
            (_config(), ["--stages", "unknown"], "unknown stage"),
        ]
        preset_too_deep = _config()
        preset_too_deep.pop("base_model")
        preset_too_deep.update({"model": "qwen05b", "checkpoint_depths": [0, 25]})
        cases.append((preset_too_deep, ["--stages", "3"], "exceed model depth 24"))
        for cfg, args, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                self._run_main(cfg, *args)


if __name__ == "__main__":
    unittest.main()
