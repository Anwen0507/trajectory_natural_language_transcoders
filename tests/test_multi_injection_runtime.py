import contextlib
import importlib
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from torch import nn

from nla.schema import (
    ACTIVATION_COLUMN,
    ACTIVATIONS_KEY,
    MM_ACTIVATION_KEY,
    MM_ACTIVATIONS_KEY,
    MM_CRITIC_TOKENS_KEY,
    MM_MSE_SCALE_KEY,
)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _Box:
    def __init__(self, inner):
        self.inner = inner


class _StateDictOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DTensor:
    pass


class _FSDPParent:
    def _get_model_inputs_args(self, batch):
        self.parent_batch = batch
        return {"parent_called": True}


class _MegatronParent:
    def train_actor(self, rollout_id, rollout_data):
        self.parent_train_call = (rollout_id, rollout_data)


def _import_training_modules():
    mpu = types.SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0)
    stubs = {
        "ray": _module("ray", get=lambda value: value, put=lambda value: value),
        "torch.distributed.checkpoint.state_dict": _module(
            "torch.distributed.checkpoint.state_dict",
            StateDictOptions=_StateDictOptions,
            get_model_state_dict=lambda *_args, **_kwargs: {},
        ),
        "torch.distributed.tensor": _module(
            "torch.distributed.tensor", DTensor=_DTensor
        ),
        "miles": _module("miles"),
        "miles.backends": _module("miles.backends"),
        "miles.backends.fsdp_utils": _module("miles.backends.fsdp_utils"),
        "miles.backends.fsdp_utils.actor": _module(
            "miles.backends.fsdp_utils.actor",
            FSDPTrainRayActor=_FSDPParent,
            apply_fsdp2=lambda *_args, **_kwargs: None,
        ),
        "miles.backends.megatron_utils": _module(
            "miles.backends.megatron_utils"
        ),
        "miles.backends.megatron_utils.actor": _module(
            "miles.backends.megatron_utils.actor",
            MegatronTrainRayActor=_MegatronParent,
        ),
        "miles.backends.megatron_utils.model": _module(
            "miles.backends.megatron_utils.model",
            train=lambda *_args, **_kwargs: None,
        ),
        "miles.backends.training_utils": _module(
            "miles.backends.training_utils"
        ),
        "miles.backends.training_utils.data": _module(
            "miles.backends.training_utils.data",
            get_batch=lambda *_args, **_kwargs: None,
            get_data_iterator=lambda *_args, **_kwargs: None,
            get_rollout_data=lambda *_args, **_kwargs: None,
        ),
        "miles.backends.training_utils.log_utils": _module(
            "miles.backends.training_utils.log_utils",
            aggregate_forward_results=lambda *_args, **_kwargs: None,
            log_perf_data=lambda *_args, **_kwargs: None,
            log_rollout_data=lambda *_args, **_kwargs: None,
        ),
        "miles.backends.training_utils.loss": _module(
            "miles.backends.training_utils.loss",
            get_log_probs_and_entropy=lambda *_args, **_kwargs: None,
            loss_function=lambda *_args, **_kwargs: None,
        ),
        "miles.utils": _module("miles.utils"),
        "miles.utils.distributed_utils": _module(
            "miles.utils.distributed_utils", get_gloo_group=lambda: None
        ),
        "miles.utils.ray_utils": _module("miles.utils.ray_utils", Box=_Box),
        "miles.utils.timer": _module(
            "miles.utils.timer",
            timer=lambda *_args, **_kwargs: contextlib.nullcontext(),
        ),
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core", mpu=mpu),
        "megatron.core.pipeline_parallel": _module(
            "megatron.core.pipeline_parallel",
            get_forward_backward_func=lambda: None,
        ),
        "megatron.core.utils": _module(
            "megatron.core.utils", get_model_config=lambda *_args: None
        ),
        "megatron.training": _module("megatron.training"),
        "megatron.training.async_utils": _module(
            "megatron.training.async_utils",
            maybe_finalize_async_save=lambda **_kwargs: None,
        ),
        "nla.embed_store": _module(
            "nla.embed_store", get_embed_store=lambda: None
        ),
        "nla.megatron.checkpoint": _module(
            "nla.megatron.checkpoint",
            gather_embedding_for_dump=lambda *_args, **_kwargs: None,
        ),
    }
    sys.modules.pop("nla.train_actor", None)
    sys.modules.pop("nla.megatron.train_actor", None)
    with patch.dict(sys.modules, stubs):
        fsdp = importlib.import_module("nla.train_actor")
        megatron = importlib.import_module("nla.megatron.train_actor")
    return fsdp, megatron


class _RuntimeSample:
    class Status:
        COMPLETE = "complete"
        TRUNCATED = "truncated"
        FAILED = "failed"


async def _async_noop(*_args, **_kwargs):
    return None


def _import_generate_module():
    stubs = {
        "miles": _module("miles"),
        "miles.rollout": _module("miles.rollout"),
        "miles.rollout.generate_utils": _module("miles.rollout.generate_utils"),
        "miles.rollout.generate_utils.generate_endpoint_utils": _module(
            "miles.rollout.generate_utils.generate_endpoint_utils",
            compute_request_payload=lambda *_args, **_kwargs: ({}, None),
            update_sample_from_response=_async_noop,
        ),
        "miles.rollout.inference_rollout": _module(
            "miles.rollout.inference_rollout"
        ),
        "miles.rollout.inference_rollout.inference_rollout_train": _module(
            "miles.rollout.inference_rollout.inference_rollout_train",
            get_worker_urls=_async_noop,
        ),
        "miles.utils": _module("miles.utils"),
        "miles.utils.http_utils": _module(
            "miles.utils.http_utils", post=_async_noop
        ),
        "miles.utils.processing_utils": _module(
            "miles.utils.processing_utils",
            load_tokenizer=lambda *_args, **_kwargs: None,
        ),
        "miles.utils.types": _module(
            "miles.utils.types", Sample=_RuntimeSample
        ),
    }
    sys.modules.pop("nla.rollout.nla_generate", None)
    with patch.dict(sys.modules, stubs):
        return importlib.import_module("nla.rollout.nla_generate")


def _cfg(*, sites=2, d_model=3, scale=5.0):
    names = tuple(["embedding", "block_04"][:sites])
    if sites == 1:
        names = ("embedding",)
    return types.SimpleNamespace(
        d_model=d_model,
        num_injection_sites=sites,
        activation_checkpoint_names=names,
        injection_scale=scale,
        mse_scale=7.0,
        injection_token_id=8,
        injection_left_neighbor_id=7,
        injection_right_neighbor_id=9,
        critic_prompt_template="<text>{explanation}</text><summary>",
    )


class FSDPMultiInjectionRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.megatron_module = _import_training_modules()

    def _actor(self, *, critic=False, sites=2, d_model=3):
        actor = object.__new__(self.module.NLAFSDPActor)
        actor._is_critic_model = critic
        actor._nla_cfg = _cfg(sites=sites, d_model=d_model)
        actor._nla_vectors = None
        return actor

    def test_actor_consumes_and_normalizes_ordered_bundle(self):
        actor = self._actor()
        bundle = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]])
        legacy = torch.full((1, 3), 99.0)
        mm = {
            MM_ACTIVATIONS_KEY: bundle.clone(),
            MM_ACTIVATION_KEY: legacy,
            "preserved": torch.tensor(1),
        }

        result = actor._get_model_inputs_args({"multimodal_train_inputs": mm})

        self.assertEqual(result, {"parent_called": True, "use_cache": False})
        self.assertNotIn(MM_ACTIVATIONS_KEY, mm)
        self.assertNotIn(MM_ACTIVATION_KEY, mm)
        self.assertIn("preserved", mm)
        torch.testing.assert_close(
            actor._nla_vectors.norm(dim=-1), torch.tensor([[5.0, 5.0]])
        )
        torch.testing.assert_close(
            actor._nla_vectors[0, 0], torch.tensor([3.0, 4.0, 0.0])
        )

    def test_actor_promotes_legacy_vector_and_accepts_no_multimodal_data(self):
        actor = self._actor(sites=1)
        single = torch.tensor([[3.0, 4.0, 0.0]])
        actor._get_model_inputs_args({
            "multimodal_train_inputs": {MM_ACTIVATION_KEY: single}
        })
        self.assertEqual(tuple(actor._nla_vectors.shape), (1, 1, 3))
        torch.testing.assert_close(
            actor._nla_vectors.norm(dim=-1), torch.tensor([[5.0]])
        )

        result = actor._get_model_inputs_args({})
        self.assertFalse(result["use_cache"])

    def test_critic_keeps_only_singular_gold_and_mse_scale(self):
        actor = self._actor(critic=True)
        bundle = torch.ones(1, 2, 3)
        single = torch.tensor([[1.0, 2.0, 3.0]])
        batch = {
            "multimodal_train_inputs": {
                MM_ACTIVATIONS_KEY: bundle,
                MM_ACTIVATION_KEY: single,
            }
        }

        actor._get_model_inputs_args(batch)

        self.assertIs(batch[MM_ACTIVATION_KEY], single)
        self.assertEqual(batch[MM_MSE_SCALE_KEY], 7.0)
        self.assertIsNone(actor._nla_vectors)

    def test_critic_ignores_bundle_without_singular_gold(self):
        actor = self._actor(critic=True)
        batch = {
            "multimodal_train_inputs": {
                MM_ACTIVATIONS_KEY: torch.ones(1, 2, 3),
                "preserved": True,
            }
        }

        actor._get_model_inputs_args(batch)

        self.assertNotIn(MM_ACTIVATION_KEY, batch)
        self.assertNotIn(MM_MSE_SCALE_KEY, batch)
        self.assertEqual(
            batch["multimodal_train_inputs"], {"preserved": True}
        )

    def test_actor_accepts_nonempty_multimodal_dict_without_activation_keys(self):
        actor = self._actor()
        actor._get_model_inputs_args({
            "multimodal_train_inputs": {"preserved": torch.tensor(1)}
        })
        self.assertIsNone(actor._nla_vectors)

    def test_actor_rejects_bad_bundle_rank_checkpoint_count_and_width(self):
        cases = [
            (torch.zeros(2, 3), r"must be \[B, K, d_model\]"),
            (torch.zeros(1, 1, 3), r"expected \[B, 2, 3\]"),
            (torch.zeros(1, 2, 4), r"expected \[B, 2, 3\]"),
        ]
        for bundle, message in cases:
            with self.subTest(shape=tuple(bundle.shape)):
                actor = self._actor()
                with self.assertRaisesRegex(AssertionError, message):
                    actor._get_model_inputs_args({
                        "multimodal_train_inputs": {MM_ACTIVATIONS_KEY: bundle}
                    })

    def test_embedding_hook_flattens_sample_checkpoint_order_and_honors_skip(self):
        actor = self._actor(d_model=2)
        actor._nla_vectors = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        model = nn.Module()
        model.embed = nn.Embedding(16, 2)
        nn.init.constant_(model.embed.weight, -1.0)
        model.get_input_embeddings = lambda: model.embed
        actor._register_injection_hook(model)
        ids = torch.tensor([[1, 7, 8, 9, 2, 7, 8, 9, 3]])

        actual = model.embed(ids)

        torch.testing.assert_close(actual[0, 2], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(actual[0, 6], torch.tensor([3.0, 4.0]))
        with patch.dict(os.environ, {"NLA_SKIP_INJECTION": "1"}):
            skipped = model.embed(ids)
        torch.testing.assert_close(skipped, torch.full_like(skipped, -1.0))


class MegatronMultiInjectionRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fsdp_module, cls.module = _import_training_modules()

    def _actor(self, *, sites=2, d_model=3, cpu_ids=False, sequence_parallel=False):
        actor = object.__new__(self.module.NLAMegatronActor)
        actor._nla_cfg = _cfg(sites=sites, d_model=d_model)
        actor._nla_vectors_slot = [None]
        actor._nla_input_ids_slot = [None]
        actor._disable_train_offload = cpu_ids
        actor.args = types.SimpleNamespace(sequence_parallel=sequence_parallel)
        return actor

    def test_strip_hook_stashes_joint_bundle_and_removes_all_nla_kwargs(self):
        actor = self._actor(cpu_ids=True)
        bundle = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]])
        input_ids = torch.tensor([[1, 2, 3]])
        kwargs = {
            MM_ACTIVATIONS_KEY: bundle,
            MM_ACTIVATION_KEY: torch.full((1, 3), 99.0),
            MM_CRITIC_TOKENS_KEY: torch.tensor([1]),
            "input_ids": input_ids,
            "preserved": True,
        }

        args_out, kwargs_out = actor._make_strip_hook(stash=True)(
            None, ("arg",), kwargs
        )

        self.assertEqual(args_out, ("arg",))
        self.assertTrue(kwargs_out["preserved"])
        self.assertNotIn(MM_ACTIVATIONS_KEY, kwargs_out)
        self.assertNotIn(MM_ACTIVATION_KEY, kwargs_out)
        self.assertNotIn(MM_CRITIC_TOKENS_KEY, kwargs_out)
        torch.testing.assert_close(
            actor._nla_vectors_slot[0].norm(dim=-1),
            torch.tensor([[5.0, 5.0]]),
        )
        torch.testing.assert_close(actor._nla_input_ids_slot[0], input_ids)

    def test_strip_hook_promotes_legacy_vector_and_stash_false_only_strips(self):
        actor = self._actor(sites=1)
        single = torch.tensor([[3.0, 4.0, 0.0]])
        actor._make_strip_hook(stash=True)(
            None,
            (),
            {MM_ACTIVATION_KEY: single, "input_ids": torch.tensor([[1]])},
        )
        self.assertEqual(tuple(actor._nla_vectors_slot[0].shape), (1, 1, 3))

        actor._nla_vectors_slot[0] = None
        _, kwargs = actor._make_strip_hook(stash=False)(
            None,
            (),
            {
                MM_ACTIVATIONS_KEY: torch.ones(1, 1, 3),
                MM_ACTIVATION_KEY: single,
                MM_CRITIC_TOKENS_KEY: torch.tensor([1]),
                "keep": 1,
            },
        )
        self.assertEqual(kwargs, {"keep": 1})
        self.assertIsNone(actor._nla_vectors_slot[0])

    def test_strip_hook_rejects_bad_rank_count_and_width(self):
        cases = [
            (torch.zeros(2, 3), r"must be \[B, K, d_model\]"),
            (torch.zeros(1, 1, 3), r"expected \[B, 2, 3\]"),
            (torch.zeros(1, 2, 4), r"expected \[B, 2, 3\]"),
        ]
        for bundle, message in cases:
            with self.subTest(shape=tuple(bundle.shape)):
                actor = self._actor()
                hook = actor._make_strip_hook(stash=True)
                with self.assertRaisesRegex(AssertionError, message):
                    hook(
                        None,
                        (),
                        {
                            MM_ACTIVATIONS_KEY: bundle,
                            "input_ids": torch.tensor([[1]]),
                        },
                    )

    def test_injection_hook_is_one_shot_and_preserves_checkpoint_order(self):
        actor = self._actor(d_model=2)
        actor._nla_vectors_slot[0] = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]]
        )
        ids = torch.tensor([[1, 7, 8, 9, 2, 7, 8, 9, 3]])
        output = torch.full((9, 1, 2), -1.0)
        hook = actor._make_injection_hook()

        actual = hook(None, (), {"input_ids": ids}, output)

        torch.testing.assert_close(actual[2, 0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(actual[6, 0], torch.tensor([3.0, 4.0]))
        self.assertTrue(actual.is_contiguous())
        self.assertIsNone(actor._nla_vectors_slot[0])
        self.assertIs(hook(None, (), {"input_ids": ids}, output), output)

    def test_injection_hook_uses_global_vector_index_in_sequence_parallel_slice(self):
        actor = self._actor(
            d_model=2, cpu_ids=True, sequence_parallel=True
        )
        actor._nla_vectors_slot[0] = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]]
        )
        ids = torch.tensor([[1, 7, 8, 9, 0, 7, 8, 9]])
        actor._nla_input_ids_slot[0] = ids
        output = torch.full((4, 1, 2), -1.0)
        hook = actor._make_injection_hook()

        with patch.object(
            self.module.mpu, "get_tensor_model_parallel_rank", return_value=1
        ):
            actual = hook(
                None,
                (),
                {"input_ids": torch.tensor([[99]])},
                output,
            )

        torch.testing.assert_close(actual[2, 0], torch.tensor([3.0, 4.0]))
        self.assertTrue(torch.all(actual[[0, 1, 3], 0] == -1))
        self.assertIsNone(actor._nla_input_ids_slot[0])

    def test_injection_hook_honors_skip_environment_without_consuming_slot(self):
        actor = self._actor(d_model=2)
        bundle = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        actor._nla_vectors_slot[0] = bundle
        output = torch.zeros(4, 1, 2)
        with patch.dict(os.environ, {"NLA_SKIP_INJECTION": "1"}):
            actual = actor._make_injection_hook()(
                None, (), {"input_ids": torch.tensor([[1]])}, output
            )
        self.assertIs(actual, output)
        self.assertIs(actor._nla_vectors_slot[0], bundle)

    def test_train_actor_counts_joint_and_legacy_vectors_before_pipeline(self):
        actor = self._actor()
        actor.args = types.SimpleNamespace(
            use_dynamic_batch_size=True,
            micro_batch_size=1,
            use_critic=True,
        )
        actor.parallel_state = types.SimpleNamespace(dp_group=object())
        rollout_data = {
            "multimodal_train_inputs": [
                None,
                {MM_ACTIVATIONS_KEY: torch.zeros(2, 2, 3)},
                {MM_ACTIVATION_KEY: torch.zeros(3, 3)},
                {},
            ]
        }
        with self.assertRaisesRegex(
            RuntimeError, r"1/4 samples.*\(7 vectors survive\)"
        ):
            actor._train_nla_actor(4, rollout_data)

    def test_train_actor_success_strips_critic_tokens_and_restores_state(self):
        actor = self._actor()
        actor.args = types.SimpleNamespace(
            use_dynamic_batch_size=True,
            micro_batch_size=1,
            use_critic=True,
        )
        actor.parallel_state = types.SimpleNamespace(dp_group=object())
        actor._nla_vectors_slot[0] = torch.ones(1, 2, 3)
        mm = {
            MM_ACTIVATIONS_KEY: torch.ones(1, 2, 3),
            MM_CRITIC_TOKENS_KEY: torch.tensor([1, 2]),
        }
        rollout_data = {"multimodal_train_inputs": [mm]}
        truncated = {"multimodal_train_inputs": [mm], "truncated": True}

        with patch.object(
            self.module,
            "_truncate_to_cross_rank_min",
            return_value=truncated,
        ) as truncate:
            actor._train_nla_actor(5, rollout_data)

        truncate.assert_called_once_with(
            rollout_data, actor.parallel_state.dp_group, None
        )
        self.assertNotIn(MM_CRITIC_TOKENS_KEY, mm)
        self.assertEqual(actor.parent_train_call, (5, truncated))
        self.assertTrue(actor.args.use_critic)
        self.assertIsNone(actor._nla_vectors_slot[0])


class _PrepTokenizer:
    def __init__(self, ids):
        self.ids = list(ids)

    def apply_chat_template(self, *_args, tokenize=False, **_kwargs):
        return list(self.ids) if tokenize else "rendered prompt"

    def encode(self, _prompt, *, add_special_tokens):
        self.last_add_special_tokens = add_special_tokens
        return list(self.ids)

    def __call__(self, _text, *, add_special_tokens):
        return {"input_ids": [1, 2, 3]}


class RolloutMultiInjectionRuntimeTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _import_generate_module()

    def setUp(self):
        self.module._DEBUG_TIMING = False
        self.module._BF16_B64_EMBEDS = False
        self.module._PREFILL_LEAK_PINGED = False

    @staticmethod
    def _args(**overrides):
        values = {
            "sglang_disable_radix_cache": False,
            "rollout_max_context_len": 512,
            "save": None,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def _configure_prep(self, *, sites=2):
        ids = [1, 7, 8, 9, 2]
        if sites == 2:
            ids += [7, 8, 9, 3]
        self.module._CFG = _cfg(sites=sites)
        self.module._TOKENIZER = _PrepTokenizer(ids)
        self.module._EMBED = nn.Embedding(16, 3)
        with torch.no_grad():
            for token_id in range(16):
                self.module._EMBED.weight[token_id] = torch.tensor(
                    [token_id, token_id + 0.1, token_id + 0.2]
                )
        self.module._EMBED_SCALE = 1.0

    def test_prepare_payload_injects_and_normalizes_each_checkpoint(self):
        self._configure_prep(sites=2)
        vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], np.float32)
        payload = {"input_ids": [1], "sampling_params": {}}
        with patch.object(
            self.module,
            "compute_request_payload",
            return_value=(payload, None),
        ):
            input_ids, raw, embeds, actual_payload, halt = (
                self.module._prep_payload_sync(
                    self._args(), [], vectors, {"temperature": 1.0}, 17
                )
            )

        self.assertEqual(input_ids, [1, 7, 8, 9, 2, 7, 8, 9, 3])
        self.assertFalse(self.module._TOKENIZER.last_add_special_tokens)
        self.assertEqual(tuple(raw.shape), (1, 2, 3))
        self.assertIs(actual_payload, payload)
        self.assertIsNone(halt)
        self.assertIsInstance(embeds, np.ndarray)
        np.testing.assert_allclose(np.linalg.norm(embeds[[2, 6]], axis=-1), 5.0)
        np.testing.assert_allclose(embeds[2], vectors[0])
        np.testing.assert_allclose(embeds[6], [0.0, 0.0, 5.0])

    def test_prepare_payload_promotes_legacy_flat_vector(self):
        self._configure_prep(sites=1)
        with patch.object(
            self.module,
            "compute_request_payload",
            return_value=({"input_ids": [1], "sampling_params": {}}, None),
        ):
            _, raw, embeds, _, _ = self.module._prep_payload_sync(
                self._args(), [], [3.0, 4.0, 0.0], {}, 2
            )
        self.assertEqual(tuple(raw.shape), (1, 1, 3))
        np.testing.assert_allclose(embeds[2], [3.0, 4.0, 0.0])

    def test_prepare_payload_rejects_shape_and_nonfinite_values(self):
        self._configure_prep(sites=2)
        cases = [
            (np.zeros((3,), np.float32), r"shape \(1, 3\).*expected \(2, 3\)"),
            (np.zeros((2, 4), np.float32), r"shape \(2, 4\).*expected \(2, 3\)"),
            (
                np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], np.float32),
                "NaN/Inf",
            ),
        ]
        for vectors, message in cases:
            with self.subTest(shape=vectors.shape):
                with self.assertRaisesRegex(AssertionError, message):
                    self.module._prep_payload_sync(
                        self._args(), [], vectors, {}, 9
                    )

    async def _run_generate(self, *, sites, metadata):
        self.module._CFG = _cfg(sites=sites)
        self.module._TOKENIZER = _PrepTokenizer([1, 2, 3])
        sample = types.SimpleNamespace(
            prompt=[{"role": "user", "content": "prompt"}],
            metadata=metadata,
            index=12,
            response="",
            status=_RuntimeSample.Status.COMPLETE,
        )
        vectors = metadata.get(ACTIVATIONS_KEY, metadata.get(ACTIVATION_COLUMN))
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        raw = torch.from_numpy(array).unsqueeze(0)
        payload = {
            "input_ids": [1, 2],
            "sampling_params": {"max_new_tokens": 5},
        }
        prep = Mock(return_value=(
            [1, 2], raw, np.zeros((2, 3), np.float32), payload, None
        ))

        async def update(_args, target, *, payload, output):
            self.assertEqual(payload, {"input_ids": [1, 2]})
            self.assertIn("meta_info", output)
            target.response = "<explanation> decoded state </explanation>"
            target.status = _RuntimeSample.Status.COMPLETE

        with (
            patch.object(self.module, "_lazy_init"),
            patch.object(self.module, "_maybe_reload_embed"),
            patch.object(
                self.module, "_resolve_url", new=AsyncMock(return_value="engine")
            ),
            patch.object(self.module, "_prep_payload_sync", prep),
            patch.object(
                self.module,
                "post",
                new=AsyncMock(return_value={
                    "meta_info": {"output_token_logprobs": [0.1]},
                }),
            ),
            patch.object(
                self.module,
                "update_sample_from_response",
                new=AsyncMock(side_effect=update),
            ),
        ):
            result = await self.module.generate(self._args(), sample, {})
        return result, prep

    async def test_generate_prefers_joint_metadata_and_stashes_only_bundle(self):
        joint = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
        result, prep = await self._run_generate(
            sites=2,
            metadata={
                ACTIVATIONS_KEY: joint,
                ACTIVATION_COLUMN: np.array([99.0, 99.0, 99.0], np.float32),
            },
        )

        np.testing.assert_array_equal(prep.call_args.args[2], joint)
        self.assertEqual(
            set(result.multimodal_train_inputs),
            {MM_ACTIVATIONS_KEY, MM_CRITIC_TOKENS_KEY},
        )
        self.assertEqual(
            tuple(result.multimodal_train_inputs[MM_ACTIVATIONS_KEY].shape),
            (1, 2, 3),
        )

    async def test_generate_legacy_metadata_preserves_singular_training_key(self):
        legacy = np.array([1.0, 2.0, 3.0], np.float32)
        result, prep = await self._run_generate(
            sites=1, metadata={ACTIVATION_COLUMN: legacy}
        )

        np.testing.assert_array_equal(prep.call_args.args[2], legacy)
        self.assertEqual(
            set(result.multimodal_train_inputs),
            {MM_ACTIVATIONS_KEY, MM_ACTIVATION_KEY, MM_CRITIC_TOKENS_KEY},
        )
        torch.testing.assert_close(
            result.multimodal_train_inputs[MM_ACTIVATION_KEY],
            torch.tensor([[1.0, 2.0, 3.0]]),
        )

    async def test_generate_rejects_missing_activation_metadata(self):
        self.module._CFG = _cfg(sites=2)
        sample = types.SimpleNamespace(
            prompt=[], metadata={}, index=3
        )
        with (
            patch.object(self.module, "_lazy_init"),
            patch.object(self.module, "_maybe_reload_embed"),
            patch.object(
                self.module, "_resolve_url", new=AsyncMock(return_value="engine")
            ),
            self.assertRaisesRegex(
                AssertionError, "has neither 'activation_vectors' nor"
            ),
        ):
            await self.module.generate(self._args(), sample, {})


if __name__ == "__main__":
    unittest.main()
