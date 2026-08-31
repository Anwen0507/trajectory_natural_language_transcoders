"""Activation-checkpoint naming and validation.

A checkpoint depth is the number of completed decoder blocks:

* depth 0 is the embedding output;
* depth N > 0 is the output of decoder block ``N - 1`` in a zero-based
  ``ModuleList``.

Using depths at the CLI/metadata boundary avoids making users translate
"after block 4" into the implementation detail ``layers[3]``.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationCheckpoint:
    name: str
    depth: int
    kind: str  # embedding | block_output
    layer_index: int | None
    final_norm_applied: bool = False

    @property
    def column_name(self) -> str:
        return f"activation_{self.name}"


def checkpoint_for_depth(depth: int) -> ActivationCheckpoint:
    assert depth >= 0, f"checkpoint depth must be non-negative, got {depth}"
    if depth == 0:
        return ActivationCheckpoint(
            name="embedding",
            depth=0,
            kind="embedding",
            layer_index=None,
        )
    return ActivationCheckpoint(
        name=f"block_{depth:02d}",
        depth=depth,
        kind="block_output",
        layer_index=depth - 1,
    )


def checkpoints_for_depths(
    depths: list[int] | tuple[int, ...], num_layers: int | None = None
) -> list[ActivationCheckpoint]:
    depths = list(depths)
    assert depths, "at least one checkpoint depth is required"
    assert depths == sorted(set(depths)), (
        f"checkpoint depths must be sorted and unique, got {depths}"
    )
    if num_layers is not None:
        assert all(depth <= num_layers for depth in depths), (
            f"checkpoint depths {depths} exceed model depth {num_layers}"
        )
    return [checkpoint_for_depth(depth) for depth in depths]


def base_dataset_id(
    base_model: str,
    checkpoints: list[ActivationCheckpoint],
    corpus: str,
    corpus_slice: dict[str, int],
    *,
    legacy_layer_index: int | None = None,
) -> str:
    """Stable Stage-0 ID, preserving the historical single-layer format."""
    model_tag = base_model.split("/")[-1]
    if legacy_layer_index is not None:
        identity = f"{base_model}|{legacy_layer_index}|{corpus}|{corpus_slice}"
        tag = f"L{legacy_layer_index}"
    else:
        depths = tuple(checkpoint.depth for checkpoint in checkpoints)
        identity = f"{base_model}|checkpoint_depths={depths}|{corpus}|{corpus_slice}"
        tag = "C" + "-".join(str(depth) for depth in depths)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
    return f"base_{model_tag}_{tag}_{digest}"
