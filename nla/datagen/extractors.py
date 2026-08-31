"""Activation extraction backends.

Stage 0 forwards a base model over a corpus and grabs hidden states at one or
more checkpoints. `ActivationExtractor` is the pluggable interface — stage 0
code calls `extract_many()` with a list of texts and checkpoint depths and gets
back per-text hidden states + token IDs. GPU placement, batching, model
parallelism, and choice of inference engine are all the extractor's problem.

Swap via `--extractor-cls my.module.MyExtractor` at stage0 invocation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config
from nla.datagen._common import load_tokenizer


@dataclass
class ExtractionResult:
    hidden_states: torch.Tensor  # [seq_len, d_model], float32, CPU, unpadded
    token_ids: list[int]


@dataclass
class MultiExtractionResult:
    activations: dict[int, torch.Tensor]  # depth -> [seq_len, d_model], float32, CPU
    token_ids: list[int]


class ActivationExtractor(ABC):
    """Submit a batch of texts, get back checkpoint activations + token IDs.

    Subclasses own all batching/device/parallelism decisions. Callers pass
    the full task chunk and wait for results.

    Constructor contract: stage0 always passes `model_name` as a kwarg (set
    from `--base-model`). Custom extractors MUST accept `model_name` in
    __init__ — this is the provenance key written to the sidecar. Everything
    else comes via `--extractor-kwargs`.
    """

    d_model: int
    tokenizer: Any

    @abstractmethod
    def extract(self, texts: list[str], layer_index: int) -> list[ExtractionResult]: ...

    def extract_many(
        self, texts: list[str], checkpoint_depths: list[int]
    ) -> list[MultiExtractionResult]:
        """Compatibility implementation for custom single-layer extractors.

        Backends can override this to capture every checkpoint in one forward.
        The default calls the legacy ``extract`` method once per block
        checkpoint. Embedding output requires a backend-specific implementation
        because the legacy API has no representation for it.
        """
        assert checkpoint_depths, "checkpoint_depths must not be empty"
        assert checkpoint_depths == sorted(set(checkpoint_depths)), (
            f"checkpoint depths must be sorted and unique, got {checkpoint_depths}"
        )
        assert 0 not in checkpoint_depths, (
            f"{type(self).__name__} only implements legacy block extraction; "
            "override extract_many() to support the embedding checkpoint"
        )
        assert checkpoint_depths[0] > 0, (
            f"legacy block checkpoint depths must be positive, got {checkpoint_depths}"
        )
        num_layers = getattr(self, "num_layers", None)
        if num_layers is not None:
            assert checkpoint_depths[-1] <= num_layers, (
                f"checkpoint depth {checkpoint_depths[-1]} exceeds model depth {num_layers}"
            )
        by_depth = {
            depth: self.extract(texts, layer_index=depth - 1)
            for depth in checkpoint_depths
        }
        expected_n = len(texts)
        assert all(len(results) == expected_n for results in by_depth.values()), (
            "custom extractor returned a different number of results across checkpoints"
        )
        combined: list[MultiExtractionResult] = []
        for i in range(expected_n):
            token_ids = by_depth[checkpoint_depths[0]][i].token_ids
            assert all(by_depth[depth][i].token_ids == token_ids for depth in checkpoint_depths), (
                f"custom extractor token IDs changed across checkpoints for result {i}"
            )
            combined.append(MultiExtractionResult(
                activations={depth: by_depth[depth][i].hidden_states for depth in checkpoint_depths},
                token_ids=token_ids,
            ))
        return combined


class HFExtractor(ActivationExtractor):
    """Default extractor: HuggingFace transformers with targeted forward hooks.

    `device_map="auto"` handles multi-GPU model parallelism via accelerate —
    layers get sharded across GPUs transparently, no FSDP needed for
    inference-only.

    Forward hooks on only the requested checkpoints avoid
    `output_hidden_states=True`, which stores every layer's activations and,
    for some Transformers versions, exposes the final post-norm state rather
    than the pre-final-norm residual stream. Hooks capture each requested
    output for a sub-batch; we then slice padding and return per-text CPU
    tensors.

    Assumes Llama-family architecture (`model.model.layers[K]` module path).
    Works for Qwen, Llama, Mistral, Gemma. GPT-2/NeoX/Falcon use different
    paths and will AttributeError loudly at hook registration.

    `layer_index=K` returns the output of the K-th decoder block (post-MLP,
    post-residual-add — the residual stream entering layer K+1). This matches
    HF's `hidden_states[K+1]` when `output_hidden_states=True` (their index 0
    is the embedding output).
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 2048,
        batch_size: int = 16,
    ):
        assert max_length > 0, f"max_length must be positive, got {max_length}"
        assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
        self.tokenizer = load_tokenizer(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # LlamaTokenizerFast, GemmaTokenizerFast etc. default to left-padding
        # for generation. We slice [:seq_len] below — MUST be right-padded or
        # we silently return pad-position activations. Same for truncation_side
        # — left-truncation would mean token_ids[0] is NOT the doc start.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype
        ).eval()
        self.d_model = resolve_text_config(self.model.config).hidden_size
        self.num_layers = len(resolve_decoder_layers(self.model))
        self.max_length = max_length
        self.batch_size = batch_size
        self._captured: dict[int, torch.Tensor] = {}

    def _register_hooks(
        self, checkpoint_depths: list[int]
    ) -> list[torch.utils.hooks.RemovableHandle]:
        layers = resolve_decoder_layers(self.model)
        assert checkpoint_depths == sorted(set(checkpoint_depths)), (
            f"checkpoint depths must be sorted and unique, got {checkpoint_depths}"
        )
        assert checkpoint_depths and 0 <= checkpoint_depths[0], (
            f"checkpoint depths must be non-negative, got {checkpoint_depths}"
        )
        assert checkpoint_depths[-1] <= len(layers), (
            f"checkpoint depth {checkpoint_depths[-1]} out of range for model "
            f"with {len(layers)} layers"
        )

        def make_hook(depth: int):
            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                # Transformer blocks may return tuples; embeddings return a tensor.
                # .clone() because .detach() alone shares storage — under torch.compile
                # the buffer could be reused before we .cpu() it post-forward.
                h = output[0] if isinstance(output, tuple) else output
                self._captured[depth] = h.detach().clone()

            return hook

        handles: list[torch.utils.hooks.RemovableHandle] = []
        for depth in checkpoint_depths:
            module = self.model.get_input_embeddings() if depth == 0 else layers[depth - 1]
            handles.append(module.register_forward_hook(make_hook(depth)))
        return handles

    @torch.no_grad()
    def extract(self, texts: list[str], layer_index: int) -> list[ExtractionResult]:
        assert 0 <= layer_index < self.num_layers, (
            f"layer_index={layer_index} out of range for model with "
            f"{self.num_layers} layers"
        )
        depth = layer_index + 1
        multi = self.extract_many(texts, [depth])
        return [
            ExtractionResult(hidden_states=result.activations[depth], token_ids=result.token_ids)
            for result in multi
        ]

    @torch.no_grad()
    def extract_many(
        self, texts: list[str], checkpoint_depths: list[int]
    ) -> list[MultiExtractionResult]:
        handles = self._register_hooks(checkpoint_depths)
        # try/finally for hook cleanup — an exception mid-extract would
        # otherwise leak the hook and double-register on the next call.
        # (arch doc §3 explicitly permits try/finally for this one purpose.)
        try:
            return self._extract_many_impl(texts, checkpoint_depths)
        finally:
            for handle in handles:
                handle.remove()

    def _extract_many_impl(
        self, texts: list[str], checkpoint_depths: list[int]
    ) -> list[MultiExtractionResult]:
        results: list[MultiExtractionResult] = []
        for start in range(0, len(texts), self.batch_size):
            sub = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                sub,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            device = self.model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            self._captured = {}
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            missing = sorted(set(checkpoint_depths) - self._captured.keys())
            assert not missing, (
                f"forward hooks for checkpoint depths {missing} did not fire. "
                f"This architecture may use a different module path "
                f"(e.g. .transformer.h, .decoder.layers). Check model.named_modules()."
            )
            for depth, captured in self._captured.items():
                assert captured.shape[-1] == self.d_model, (
                    f"checkpoint depth {depth} tensor width {captured.shape[-1]} != "
                    f"config.hidden_size {self.d_model}. Model config lies about itself."
                )
            hidden_by_depth = {
                depth: self._captured[depth].float().cpu()
                for depth in checkpoint_depths
            }

            lengths = attention_mask.sum(dim=1).cpu()
            for i, seq_len in enumerate(lengths.tolist()):
                results.append(
                    MultiExtractionResult(
                        activations={
                            depth: hidden_by_depth[depth][i, :seq_len].clone()
                            for depth in checkpoint_depths
                        },
                        token_ids=input_ids[i, :seq_len].cpu().tolist(),
                    )
                )
        return results
