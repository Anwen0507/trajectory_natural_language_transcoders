import types
import unittest
from typing import ClassVar
from unittest.mock import patch

import torch
from torch import nn

from nla.datagen import extractors
from nla.datagen.extractors import (
    ActivationExtractor,
    ExtractionResult,
    HFExtractor,
)


class _AddBlock(nn.Module):
    def __init__(self, value: float, *, return_tuple: bool = False):
        super().__init__()
        self.value = value
        self.return_tuple = return_tuple

    def forward(self, hidden_states, **_kwargs):
        output = hidden_states + self.value
        return (output, "ignored-cache") if self.return_tuple else output


class _Scale(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states):
        return hidden_states * self.value


class _FakeCausalLM(nn.Module):
    def __init__(self, *, tuple_layer: int | None = None):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=3)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 3)
        with torch.no_grad():
            for token_id in range(32):
                self.model.embed_tokens.weight[token_id] = torch.tensor(
                    [token_id, token_id + 0.25, token_id + 0.5]
                )
        self.model.layers = nn.ModuleList(
            _AddBlock(i + 1, return_tuple=i == tuple_layer) for i in range(24)
        )
        self.model.norm = _Scale(2.0)
        self.forward_calls = 0
        self.pre_norm = None
        self.fail_after_embedding = False
        self.skip_layer = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        self.forward_calls += 1
        hidden = self.model.embed_tokens(input_ids)
        if self.fail_after_embedding:
            raise RuntimeError("intentional forward failure")
        for layer_index, layer in enumerate(self.model.layers):
            if layer_index == self.skip_layer:
                continue
            output = layer(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        self.pre_norm = hidden.detach().clone()
        return types.SimpleNamespace(last_hidden_state=self.model.norm(hidden))


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 31
    all_special_ids: ClassVar[list[int]] = [0, 31]

    def __call__(self, texts, **_kwargs):
        encoded = [[ord(char) % 20 + 1 for char in text] for text in texts]
        width = max(map(len, encoded))
        input_ids = [ids + [self.pad_token_id] * (width - len(ids)) for ids in encoded]
        attention_mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _InitTokenizer:
    def __init__(self, *, pad_token_id=None, pad_token=None):
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
        self.eos_token = "<eos>"
        self.padding_side = "left"
        self.truncation_side = "left"


class _LegacyExtractor(ActivationExtractor):
    def __init__(self):
        self.d_model = 2
        self.tokenizer = object()
        self.num_layers = 24
        self.calls = []
        self.result_count_delta = {}
        self.change_tokens_at_layer = None

    def extract(self, texts, layer_index):
        self.calls.append((list(texts), layer_index))
        count = len(texts) + self.result_count_delta.get(layer_index, 0)
        results = []
        for i in range(max(0, count)):
            token_ids = [i + 1, i + 2]
            if layer_index == self.change_tokens_at_layer:
                token_ids = [99]
            results.append(ExtractionResult(
                hidden_states=torch.full((len(token_ids), 2), float(layer_index)),
                token_ids=token_ids,
            ))
        return results


class MultiCheckpointExtractorTest(unittest.TestCase):
    def _extractor(self, *, model=None, batch_size=2):
        extractor = HFExtractor.__new__(HFExtractor)
        extractor.model = (model or _FakeCausalLM()).eval()
        extractor.tokenizer = _FakeTokenizer()
        extractor.d_model = 3
        extractor.num_layers = 24
        extractor.max_length = 32
        extractor.batch_size = batch_size
        extractor._captured = {}
        return extractor

    def test_constructor_configures_tokenizer_and_model(self):
        tokenizer = _InitTokenizer()
        model = _FakeCausalLM()
        with (
            patch.object(extractors, "load_tokenizer", return_value=tokenizer) as load,
            patch.object(
                extractors.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ) as load_model,
        ):
            extractor = HFExtractor(
                "Qwen/Qwen2.5-0.5B-Instruct",
                device_map="cpu",
                torch_dtype=torch.float16,
                max_length=77,
                batch_size=3,
            )

        load.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct")
        load_model.assert_called_once_with(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.truncation_side, "right")
        self.assertFalse(extractor.model.training)
        self.assertEqual(extractor.d_model, 3)
        self.assertEqual(extractor.num_layers, 24)
        self.assertEqual((extractor.max_length, extractor.batch_size), (77, 3))

    def test_constructor_preserves_an_existing_pad_token(self):
        tokenizer = _InitTokenizer(pad_token_id=0, pad_token="<pad>")
        with (
            patch.object(extractors, "load_tokenizer", return_value=tokenizer),
            patch.object(
                extractors.AutoModelForCausalLM,
                "from_pretrained",
                return_value=_FakeCausalLM(),
            ),
        ):
            HFExtractor("model")
        self.assertEqual(tokenizer.pad_token, "<pad>")

    def test_constructor_rejects_nonpositive_batch_and_length_before_loading(self):
        for kwargs, message in [
            ({"max_length": 0}, "max_length"),
            ({"max_length": -1}, "max_length"),
            ({"batch_size": 0}, "batch_size"),
            ({"batch_size": -1}, "batch_size"),
        ]:
            with (
                self.subTest(kwargs=kwargs),
                patch.object(extractors, "load_tokenizer") as load,
                self.assertRaisesRegex(AssertionError, message),
            ):
                HFExtractor("model", **kwargs)
            load.assert_not_called()

    def test_captures_requested_boundaries_in_one_forward(self):
        extractor = self._extractor()
        depths = [0, 4, 8, 12, 16, 20, 24]
        results = extractor.extract_many(["abc", "de"], depths)

        self.assertEqual(extractor.model.forward_calls, 1)
        self.assertEqual([list(result.activations) for result in results], [depths, depths])
        self.assertEqual(results[0].activations[0].shape, (3, 3))
        self.assertEqual(results[1].activations[0].shape, (2, 3))

        embedded = extractor.model.get_input_embeddings()(
            torch.tensor(results[0].token_ids)
        ).float()
        torch.testing.assert_close(results[0].activations[0], embedded)
        for depth in depths[1:]:
            expected = embedded + sum(range(1, depth + 1))
            torch.testing.assert_close(results[0].activations[depth], expected)

        # Block 24's hook is the residual stream entering final norm, not its output.
        torch.testing.assert_close(results[0].activations[24], extractor.model.pre_norm[0])
        self.assertFalse(torch.equal(
            results[0].activations[24], extractor.model.model.norm(extractor.model.pre_norm)[0]
        ))

        hooked_modules = [
            extractor.model.get_input_embeddings(),
            *(extractor.model.model.layers[depth - 1] for depth in depths[1:]),
        ]
        self.assertTrue(all(not module._forward_hooks for module in hooked_modules))

    def test_tuple_block_output_uses_hidden_state_element(self):
        extractor = self._extractor(model=_FakeCausalLM(tuple_layer=3))
        result = extractor.extract_many(["abc"], [4])[0]
        embedded = extractor.model.get_input_embeddings()(
            torch.tensor(result.token_ids)
        ).float()
        torch.testing.assert_close(result.activations[4], embedded + sum(range(1, 5)))

    def test_multiple_sub_batches_are_unpadded_cpu_float32_and_independent(self):
        extractor = self._extractor(batch_size=2)
        texts = ["a", "bc", "def", "ghij", "klmno"]
        results = extractor.extract_many(texts, [0, 24])

        self.assertEqual(extractor.model.forward_calls, 3)
        self.assertEqual([len(result.token_ids) for result in results], [1, 2, 3, 4, 5])
        for result in results:
            for activation in result.activations.values():
                self.assertEqual(activation.device.type, "cpu")
                self.assertEqual(activation.dtype, torch.float32)
                self.assertEqual(activation.shape, (len(result.token_ids), 3))
        before = results[1].activations[0].clone()
        results[0].activations[0].zero_()
        torch.testing.assert_close(results[1].activations[0], before)

    def test_empty_text_batch_returns_empty_without_forward(self):
        extractor = self._extractor()
        self.assertEqual(extractor.extract_many([], [0, 24]), [])
        self.assertEqual(extractor.model.forward_calls, 0)
        self.assertFalse(extractor.model.get_input_embeddings()._forward_hooks)
        self.assertFalse(extractor.model.model.layers[23]._forward_hooks)

    def test_invalid_checkpoint_depth_lists_fail_before_forward(self):
        cases = [
            ([], "non-negative"),
            ([4, 0], "sorted and unique"),
            ([4, 4], "sorted and unique"),
            ([-1, 4], "non-negative"),
            ([25], "out of range"),
        ]
        for depths, message in cases:
            with self.subTest(depths=depths):
                extractor = self._extractor()
                with self.assertRaisesRegex(AssertionError, message):
                    extractor.extract_many(["abc"], depths)
                self.assertEqual(extractor.model.forward_calls, 0)

    def test_legacy_layer_api_maps_to_completed_block_depth(self):
        extractor = self._extractor()
        result = extractor.extract(["abc"], layer_index=3)[0]
        embedded = extractor.model.get_input_embeddings()(
            torch.tensor(result.token_ids)
        ).float()
        torch.testing.assert_close(result.hidden_states, embedded + sum(range(1, 5)))

    def test_legacy_layer_api_rejects_negative_and_too_deep_indices(self):
        for layer_index in (-1, 24):
            with self.subTest(layer_index=layer_index), self.assertRaises(AssertionError):
                self._extractor().extract(["abc"], layer_index=layer_index)

    def test_hooks_are_removed_after_forward_failure(self):
        extractor = self._extractor()
        extractor.model.fail_after_embedding = True
        depths = [0, 4, 24]
        with self.assertRaisesRegex(RuntimeError, "intentional forward failure"):
            extractor.extract_many(["abc"], depths)
        hooked_modules = [
            extractor.model.get_input_embeddings(),
            extractor.model.model.layers[3],
            extractor.model.model.layers[23],
        ]
        self.assertTrue(all(not module._forward_hooks for module in hooked_modules))

    def test_missing_hook_and_wrong_width_fail_loudly_and_cleanup(self):
        extractor = self._extractor()
        extractor.model.skip_layer = 3
        with self.assertRaisesRegex(AssertionError, r"depths \[4\] did not fire"):
            extractor.extract_many(["abc"], [0, 4])
        self.assertFalse(extractor.model.get_input_embeddings()._forward_hooks)
        self.assertFalse(extractor.model.model.layers[3]._forward_hooks)

        extractor = self._extractor()
        extractor.d_model = 4
        with self.assertRaisesRegex(AssertionError, "tensor width 3 !="):
            extractor.extract_many(["abc"], [0])
        self.assertFalse(extractor.model.get_input_embeddings()._forward_hooks)


class LegacyExtractorCompatibilityTest(unittest.TestCase):
    def test_combines_legacy_results_and_maps_depth_to_layer(self):
        extractor = _LegacyExtractor()
        results = extractor.extract_many(["one", "two"], [1, 4, 24])
        self.assertEqual(
            extractor.calls,
            [(["one", "two"], 0), (["one", "two"], 3), (["one", "two"], 23)],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(list(results[0].activations), [1, 4, 24])
        torch.testing.assert_close(results[0].activations[4], torch.full((2, 2), 3.0))
        self.assertEqual(results[1].token_ids, [2, 3])

    def test_empty_texts_are_supported(self):
        extractor = _LegacyExtractor()
        self.assertEqual(extractor.extract_many([], [4, 8]), [])
        self.assertEqual([call[1] for call in extractor.calls], [3, 7])

    def test_custom_extractor_without_num_layers_owns_depth_validation(self):
        extractor = _LegacyExtractor()
        del extractor.num_layers
        result = extractor.extract_many(["one"], [25])[0]
        self.assertEqual(extractor.calls[0][1], 24)
        self.assertEqual(list(result.activations), [25])

    def test_rejects_embedding_and_invalid_checkpoint_lists(self):
        cases = [
            ([], "must not be empty"),
            ([0], "embedding checkpoint"),
            ([4, 4], "sorted and unique"),
            ([8, 4], "sorted and unique"),
            ([-1], "positive"),
            ([25], "model depth"),
        ]
        for depths, message in cases:
            with self.subTest(depths=depths), self.assertRaisesRegex(AssertionError, message):
                _LegacyExtractor().extract_many(["one"], depths)

    def test_rejects_result_count_mismatch(self):
        extractor = _LegacyExtractor()
        extractor.result_count_delta[7] = -1
        with self.assertRaisesRegex(AssertionError, "different number of results"):
            extractor.extract_many(["one"], [4, 8])

    def test_rejects_tokenization_mismatch(self):
        extractor = _LegacyExtractor()
        extractor.change_tokens_at_layer = 7
        with self.assertRaisesRegex(AssertionError, "token IDs changed"):
            extractor.extract_many(["one"], [4, 8])


class RealQwen2ArchitectureTest(unittest.TestCase):
    def test_24_layer_qwen2_boundaries_and_pre_final_rmsnorm(self):
        # This is the real Qwen2 module graph with tiny random dimensions: it
        # validates hook placement without downloading the 0.5B checkpoint.
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        model = Qwen2ForCausalLM(config).eval()
        extractor = HFExtractor.__new__(HFExtractor)
        extractor.model = model
        extractor.tokenizer = _FakeTokenizer()
        extractor.d_model = config.hidden_size
        extractor.num_layers = config.num_hidden_layers
        extractor.max_length = 32
        extractor.batch_size = 2
        extractor._captured = {}

        depths = [0, 4, 8, 12, 16, 20, 24]
        result = extractor.extract_many(["abc"], depths)[0]
        input_ids = torch.tensor([result.token_ids])
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            reference = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        for depth in depths[:-1]:
            torch.testing.assert_close(
                result.activations[depth], reference.hidden_states[depth][0].float()
            )
        pre_norm = result.activations[24]
        post_norm = reference.last_hidden_state[0].float()
        torch.testing.assert_close(model.model.norm(pre_norm), post_norm)
        self.assertFalse(torch.allclose(pre_norm, post_norm))


if __name__ == "__main__":
    unittest.main()
