import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nla.schema import (
    ACTIVATION_COLUMN,
    ACTIVATIONS_KEY,
    MM_ACTIVATIONS_KEY,
)


class _MaskGenerator:
    def get_loss_mask(self, messages):
        assert messages[-1]["role"] == "assistant"
        return [10, 11, 12, 13], [0, 0, 1, 1]

    def get_response_lengths(self, loss_masks):
        return [sum(loss_masks[0])]


def _import_sft_actor_with_miles_stubs():
    miles = types.ModuleType("miles")
    utils = types.ModuleType("miles.utils")
    mask_utils = types.ModuleType("miles.utils.mask_utils")
    processing_utils = types.ModuleType("miles.utils.processing_utils")
    mask_utils.MultiTurnLossMaskGenerator = lambda *_args, **_kwargs: _MaskGenerator()
    processing_utils.load_tokenizer = lambda *_args, **_kwargs: object()
    stubs = {
        "miles": miles,
        "miles.utils": utils,
        "miles.utils.mask_utils": mask_utils,
        "miles.utils.processing_utils": processing_utils,
    }
    sys.modules.pop("nla.rollout.sft_actor", None)
    with patch.dict(sys.modules, stubs):
        return importlib.import_module("nla.rollout.sft_actor")


class MultiInjectionSFTRolloutTest(unittest.TestCase):
    @staticmethod
    def _args():
        return types.SimpleNamespace(
            rollout_global_dataset=True,
            rollout_batch_size=1,
            hf_checkpoint="model",
            loss_mask_type="default",
        )

    def test_stashes_ordered_bundle_with_concat_dimension(self):
        module = _import_sft_actor_with_miles_stubs()
        sample = types.SimpleNamespace(
            prompt=[{"role": "user", "content": "early x late x"}],
            metadata={
                "response": "<explanation>state</explanation>",
                ACTIVATIONS_KEY: np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
                ),
            },
        )
        data_buffer = types.SimpleNamespace(get_samples=lambda _count: [[sample]])
        result = module.generate_rollout(self._args(), 0, data_buffer)

        self.assertEqual(result, [[sample]])
        actual = sample.multimodal_train_inputs[MM_ACTIVATIONS_KEY]
        self.assertEqual(tuple(actual.shape), (1, 2, 3))
        torch.testing.assert_close(
            actual,
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        )
        self.assertEqual(sample.response_length, 2)
        self.assertEqual(sample.loss_mask, [1, 1])

    def test_legacy_vector_is_promoted_to_one_checkpoint_bundle(self):
        module = _import_sft_actor_with_miles_stubs()
        sample = types.SimpleNamespace(
            prompt=[{"role": "user", "content": "single x"}],
            metadata={
                "response": "<explanation>state</explanation>",
                ACTIVATION_COLUMN: [1.0, 2.0, 3.0],
            },
        )
        data_buffer = types.SimpleNamespace(get_samples=lambda _count: [[sample]])

        module.generate_rollout(self._args(), 0, data_buffer)

        torch.testing.assert_close(
            sample.multimodal_train_inputs[MM_ACTIVATIONS_KEY],
            torch.tensor([[[1.0, 2.0, 3.0]]]),
        )

    def test_rejects_scalar_and_flat_joint_bundle_metadata(self):
        module = _import_sft_actor_with_miles_stubs()
        for raw in (1.0, [1.0, 2.0, 3.0]):
            with self.subTest(raw=raw):
                sample = types.SimpleNamespace(
                    prompt=[{"role": "user", "content": "bad x"}],
                    metadata={
                        "response": "<explanation>state</explanation>",
                        ACTIVATIONS_KEY: raw,
                    },
                )
                data_buffer = types.SimpleNamespace(
                    get_samples=lambda _count, sample=sample: [[sample]]
                )
                with self.assertRaisesRegex(
                    AssertionError, "actor activations must be"
                ):
                    module.generate_rollout(self._args(), 0, data_buffer)


if __name__ == "__main__":
    unittest.main()
