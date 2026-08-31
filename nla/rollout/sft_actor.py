"""Actor-SFT rollout: no generation — tokenize prompt+response, stash activation.

Pattern follows miles/rollout/sft_rollout.py. The data_buffer yields Samples
whose .prompt is a list[dict] (from NLADataSource, <INJECT>→㊗ already substituted)
and whose .metadata["response"] is the <explanation>...</explanation> string.
"""

import torch

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer

from nla.schema import (
    ACTIVATION_COLUMN,
    ACTIVATIONS_KEY,
    MM_ACTIVATIONS_KEY,
)


_TOKENIZER = None
_MASK_GEN = None


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global _TOKENIZER, _MASK_GEN
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if _MASK_GEN is None:
        _MASK_GEN = MultiTurnLossMaskGenerator(_TOKENIZER, tokenizer_type=args.loss_mask_type)

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for group in samples:
        (sample,) = group
        messages = sample.prompt
        assert isinstance(messages, list), (
            f"actor SFT requires list[dict] prompt (got {type(messages).__name__}). "
            f"NLADataSource must use apply_chat_template=False."
        )
        response = sample.metadata["response"]
        messages = messages + [{"role": "assistant", "content": response}]

        token_ids, loss_mask = _MASK_GEN.get_loss_mask(messages)
        response_length = _MASK_GEN.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0.0
        sample.loss_mask = loss_mask[-response_length:]

        raw = sample.metadata.get(ACTIVATIONS_KEY)
        if raw is None:
            # Legacy NLADataSource/sample contract.
            raw = [sample.metadata[ACTIVATION_COLUMN]]
        activations = torch.as_tensor(raw, dtype=torch.float32)
        assert activations.ndim == 2, (
            f"actor activations must be [K, d_model], got {tuple(activations.shape)}"
        )
        sample.multimodal_train_inputs = {
            MM_ACTIVATIONS_KEY: activations.unsqueeze(0)
        }

    return samples
