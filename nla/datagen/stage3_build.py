"""Stage 3: Build final training parquets — {av_sft,ar_sft,rl}.parquet + complete sidecars.

Transforms base-schema rows into the stage-specific training format:

  AV-SFT (actor SFT):
    prompt    list[dict]  [{"role":"user","content": actor_template with <INJECT> literal}]
    response  str         "<explanation>\n{api_explanation}\n</explanation>"
    activation_vector                      legacy/selected checkpoint
      OR
    activation_{checkpoint} (one per site) joint checkpoint injection

  AR-SFT (critic SFT):
    prompt    str         critic_template filled with api_explanation. Ends with
                          a known suffix (e.g. `</text> <summary>`).
    activation_vector
    EVERY ROW tokenized at build time — assert it ends with the expected
    suffix IDs. Training extracts at the last-token position (no scanning,
    just tokens[-1] — GPU-friendly).

  RL:
    Same as AV-SFT minus response. Joint checkpoint files carry all named
    activation columns for actor injection; multi-checkpoint AR/reward is a
    separate contract and is intentionally not implemented here.

Token IDs are NOT per-row columns — they're dataset constants, shipped once
in the sidecar, loaded once by training (v4 contract).

The parquet `prompt` column contains the literal `<INJECT>` placeholder —
training-side NLADataSource swaps it for ㊗ at load time. We compute and
write the REAL ㊗ token ID + neighbor IDs to the sidecar so training can
verify against its live tokenizer.

Activation vectors are passed through RAW (norm="none"). Normalization is a
training-time decision — NLADataSource or the training loop can normalize
as needed. Data-gen's job is just "get vectors out of the model".
"""

import argparse
from dataclasses import replace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nla.datagen._common import add_storage_args, load_tokenizer, make_storage
from nla.datagen.checkpoints import ActivationCheckpoint
from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.sidecar import read_sidecar, write_sidecar
from nla.schema import wrap_explanation

_INJECT_PLACEHOLDER = "<INJECT>"

# Matches the paper appendix prompt template, but 2-3 snippets
# (not 4-5) since that's what our stage2 API prompt produces. The
# {injection_char} slot is what becomes <INJECT> in the parquet (then the real
# injection char at training time). Neighbor-ID computation in build_token_meta
# looks at the bytes immediately around {injection_char}, so keep the
# <concept>...</concept> wrapping tight.
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""


def _default_actor_template(checkpoints: list[ActivationCheckpoint]) -> str:
    """Return the legacy prompt for K=1 and an ordered checkpoint prompt for K>1."""
    assert checkpoints, "at least one activation checkpoint is required"
    if len(checkpoints) == 1:
        return _DEFAULT_ACTOR_TEMPLATE
    rows = []
    for checkpoint in checkpoints:
        label = (
            "Embedding checkpoint"
            if checkpoint.depth == 0
            else f"Block {checkpoint.depth} checkpoint"
        )
        rows.append(f"{label}: <concept>{{injection_char}}</concept>")
    checkpoint_rows = "\n".join(rows)
    return f"""You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content represented jointly by those activation vectors.

We will pass vectors from ordered checkpoints in the model, each enclosed in <concept> tags, into your context. You must then produce one explanation for their joint semantic content, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing the represented state.

Here are the checkpoint vectors, ordered from earliest to latest:

{checkpoint_rows}

Please provide an explanation."""
# Critic template ends with a fixed suffix (no marker char). Training extracts
# at the last-token position of the tokenized prompt. The suffix token IDs are
# recorded in the sidecar so training can verify the tail matches.
_DEFAULT_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
_CHUNK_SIZE = 4096

# Provenance: always carried (cheap — a few bytes/row, always useful)
_PROVENANCE_COLS = ["n_raw_tokens", "activation_layer", "doc_id"]
_MULTI_INPUT_PROVENANCE_COLS = ["n_raw_tokens", "doc_id"]
# Heavy debug: gated on --keep-debug-metadata (detokenized text can dominate file size)
_HEAVY_DEBUG_COLS = ["detokenized_text_truncated"]

# Explicit schemas: activation_vector passes through as arrow arrays (never
# hits Python), and struct/list types aren't inferable from Python dicts anyway.
_PROMPT_STRUCT = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
_PROVENANCE_FIELDS = [
    ("n_raw_tokens", pa.int64()),
    ("activation_layer", pa.int64()),
    ("doc_id", pa.string()),
]
_MULTI_PROVENANCE_FIELDS = [
    ("n_raw_tokens", pa.int64()),
    ("activation_checkpoint", pa.string()),
    ("activation_depth", pa.int64()),
    # Null for the embedding checkpoint; otherwise zero-based decoder block.
    ("activation_layer", pa.int64()),
    ("doc_id", pa.string()),
]
_JOINT_PROVENANCE_FIELDS = [
    ("n_raw_tokens", pa.int64()),
    ("doc_id", pa.string()),
]
_HEAVY_DEBUG_FIELDS = [
    ("detokenized_text_truncated", pa.string()),
]


def _schema_for(
    stage: str,
    keep_heavy_debug: bool,
    d_model: int,
    *,
    multi_checkpoint_input: bool = False,
    joint_checkpoints: list[ActivationCheckpoint] | None = None,
) -> pa.Schema:
    # FixedSizeList for activation_vector — all vectors are exactly d_model
    # wide. Variable-length ListArray silently corrupts under take() when the
    # values buffer exceeds 4 GiB (hit on 100k RL: 500k × 3584 × 4 = 6.7 GiB).
    av = pa.list_(pa.float32(), d_model)
    activation_fields = (
        [(checkpoint.column_name, av) for checkpoint in joint_checkpoints]
        if joint_checkpoints is not None
        else [("activation_vector", av)]
    )
    match stage:
        case "av_sft":
            core = [
                ("prompt", _PROMPT_STRUCT),
                ("response", pa.string()),
                *activation_fields,
            ]
        case "rl":
            core = [
                ("prompt", _PROMPT_STRUCT),
                *activation_fields,
            ]
        case "ar_sft":
            assert joint_checkpoints is None, (
                "joint-checkpoint AR output is not implemented; materialize one "
                "checkpoint with --checkpoint-depth"
            )
            core = [
                ("prompt", pa.string()),
                ("activation_vector", av),
            ]
        case _:
            raise AssertionError(f"unreachable: stage={stage!r}")
    if joint_checkpoints is not None:
        provenance = _JOINT_PROVENANCE_FIELDS
    else:
        provenance = _MULTI_PROVENANCE_FIELDS if multi_checkpoint_input else _PROVENANCE_FIELDS
    fields = core + provenance
    if keep_heavy_debug:
        fields += _HEAVY_DEBUG_FIELDS
    return pa.schema(fields)


def _resolve_activation_source(
    in_meta, in_col_names: list[str], requested_depth: int | None
) -> tuple[ActivationCheckpoint, str, bool]:
    """Return selected checkpoint, source column, and whether input is multi-checkpoint."""
    checkpoints = in_meta.extraction.checkpoints
    assert checkpoints, "input sidecar has no extraction checkpoints"

    if "activation_vector" in in_col_names:
        assert len(checkpoints) == 1, (
            "input has one activation_vector column but sidecar declares multiple checkpoints"
        )
        checkpoint = checkpoints[0]
        if requested_depth is not None:
            assert requested_depth == checkpoint.depth, (
                f"--checkpoint-depth={requested_depth} does not match the input's "
                f"single checkpoint depth {checkpoint.depth}"
            )
        return checkpoint, "activation_vector", False

    assert requested_depth is not None, (
        "multi-checkpoint input requires --checkpoint-depth; available depths: "
        f"{[checkpoint.depth for checkpoint in checkpoints]}"
    )
    matches = [checkpoint for checkpoint in checkpoints if checkpoint.depth == requested_depth]
    assert len(matches) == 1, (
        f"--checkpoint-depth={requested_depth} not present exactly once; available depths: "
        f"{[checkpoint.depth for checkpoint in checkpoints]}"
    )
    checkpoint = matches[0]
    assert checkpoint.column_name in in_col_names, (
        f"sidecar declares checkpoint {checkpoint.name!r}, but input has no "
        f"{checkpoint.column_name!r} column. Available columns: {in_col_names}"
    )
    return checkpoint, checkpoint.column_name, True


def _resolve_activation_sources(
    in_meta,
    in_col_names: list[str],
    requested_depth: int | None,
    *,
    allow_joint: bool,
) -> tuple[list[ActivationCheckpoint], list[str], bool, bool]:
    """Resolve one selected source or an ordered joint-checkpoint bundle.

    Returns (checkpoints, source_columns, multi_checkpoint_input, joint_output).
    The selected/legacy path delegates to _resolve_activation_source so its
    compatibility checks remain the single source of truth.
    """
    if "activation_vector" in in_col_names or requested_depth is not None:
        checkpoint, column, multi_input = _resolve_activation_source(
            in_meta, in_col_names, requested_depth
        )
        return [checkpoint], [column], multi_input, False

    checkpoints = in_meta.extraction.checkpoints
    assert checkpoints, "input sidecar has no extraction checkpoints"
    assert allow_joint, (
        "multi-checkpoint AR input requires --checkpoint-depth; joint AR/reward "
        "is intentionally deferred"
    )
    source_columns = [checkpoint.column_name for checkpoint in checkpoints]
    missing = [column for column in source_columns if column not in in_col_names]
    assert not missing, (
        f"sidecar declares joint checkpoint columns {source_columns}, but input "
        f"is missing {missing}. Available columns: {in_col_names}"
    )
    return checkpoints, source_columns, True, True


def _build_av_sft_cols(
    batch: pa.RecordBatch, actor_prompt_content: str
) -> dict[str, pa.Array]:
    n = len(batch)
    api_expl = batch.column("api_explanation").to_pylist()
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    return {
        "prompt": pa.array([prompt_msg] * n, type=_PROMPT_STRUCT),
        "response": pa.array([wrap_explanation(e) for e in api_expl], type=pa.string()),
    }


def _build_rl_cols(batch: pa.RecordBatch, actor_prompt_content: str) -> dict[str, pa.Array]:
    n = len(batch)
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    return {
        "prompt": pa.array([prompt_msg] * n, type=_PROMPT_STRUCT),
    }


def _build_ar_sft_cols(
    batch: pa.RecordBatch, critic_template: str, suffix_ids: list[int], tokenizer: Any
) -> dict[str, pa.Array]:
    api_expl = batch.column("api_explanation").to_pylist()
    prompts: list[str] = []
    n_suf = len(suffix_ids)
    for expl in api_expl:
        prompt = critic_template.format(explanation=expl)
        # Verify the tokenized prompt ENDS with the expected suffix IDs.
        # Training extracts at tokens[-1], so this check guarantees that's the
        # right spot. If the explanation's final bytes merge with the suffix
        # start (BPE edge case), the tail won't match and we fail loud here,
        # not silently at training.
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        assert len(ids) >= n_suf and ids[-n_suf:] == suffix_ids, (
            f"critic prompt does not end with expected suffix IDs {suffix_ids}. "
            f"Got tail: {ids[-n_suf:] if len(ids) >= n_suf else ids}. "
            f"Prompt: {prompt[:200]!r}... "
            f"This means the explanation's final characters merged with the "
            f"template suffix at the BPE boundary. Either the explanation ends "
            f"with an unusual sequence, or the critic template needs a delimiter "
            f"before the suffix."
        )
        prompts.append(prompt)
    return {
        "prompt": pa.array(prompts, type=pa.string()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="stage2 output (for av_sft/ar_sft) or stage1 output (for rl)")
    p.add_argument("--stage", required=True, choices=["av_sft", "ar_sft", "rl"])
    p.add_argument("--output", required=True)
    p.add_argument(
        "--checkpoint-depth", type=int, default=None,
        help="checkpoint to materialize from a multi-checkpoint Stage-0 input "
             "(0=embedding, N=output after N decoder blocks)",
    )
    p.add_argument(
        "--actor-template", default=None,
        help="actor prompt template. Must contain one '{injection_char}' field "
             "per selected checkpoint. Default is generated from checkpoint metadata.",
    )
    p.add_argument("--critic-template", default=_DEFAULT_CRITIC_TEMPLATE)
    p.add_argument("--keep-debug-metadata", action=argparse.BooleanOptionalAction, default=True,
                   help="carry detokenized_text_truncated through "
                        "(heavy; off for prod). Provenance (n_raw_tokens/doc_id/activation_layer) "
                        "is always carried.")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)

    if args.stage == "ar_sft":
        assert "{explanation}" in args.critic_template, (
            f"--critic-template must contain '{{explanation}}' placeholder. "
            f"Got: {args.critic_template!r}"
        )

    in_meta = read_sidecar(storage, args.input)
    assert in_meta.stage == "base", (
        f"expected stage=base input (from stage1 or stage2), got stage={in_meta.stage!r}. "
        f"Re-feeding stage3 output into stage3 would produce double-wrapped garbage."
    )
    assert in_meta.extraction.norm == "none", (
        f"expected raw vectors (norm='none') from upstream, got norm={in_meta.extraction.norm!r}. "
        f"Data-gen never normalizes — if this sidecar claims otherwise, something "
        f"upstream transformed the data or the sidecar is stale. Training-side code "
        f"is the only place that should normalize."
    )

    in_pf = pq.ParquetFile(storage.open_read(args.input))
    in_col_names = in_pf.schema_arrow.names
    checkpoints, activation_source_cols, multi_checkpoint_input, joint_output = (
        _resolve_activation_sources(
            in_meta,
            in_col_names,
            args.checkpoint_depth,
            allow_joint=args.stage in ("av_sft", "rl"),
        )
    )

    actor_template = args.actor_template or _default_actor_template(checkpoints)
    expected_sites = len(checkpoints)
    actual_sites = actor_template.count("{injection_char}")
    assert actual_sites == expected_sites, (
        f"--actor-template must contain exactly {expected_sites} "
        f"'{{injection_char}}' placeholder(s), one per checkpoint in order; "
        f"found {actual_sites}. Got: {actor_template!r}"
    )

    tokenizer = load_tokenizer(in_meta.extraction.base_model)

    # ar_sft needs critic suffix IDs; av_sft/rl don't.
    critic_template_for_meta = args.critic_template if args.stage == "ar_sft" else None
    token_meta = build_token_meta(
        tokenizer,
        actor_template,
        critic_template=critic_template_for_meta,
        expected_injection_sites=expected_sites,
    )
    inj_id = token_meta.injection_token_id
    suffix_ids = token_meta.critic_suffix_ids  # None for av_sft/rl

    # Actor prompt is constant across all rows (template is fixed). Build once.
    actor_prompt_content = actor_template.format(injection_char=_INJECT_PLACEHOLDER)
    if args.stage in ("av_sft", "ar_sft"):
        assert "api_explanation" in in_col_names, (
            f"stage={args.stage} requires api_explanation column — run stage2 first. "
            f"Available columns: {in_col_names}"
        )
    assert in_pf.metadata.num_rows > 0, (
        "input parquet is empty — nothing to build. Check upstream split fractions."
    )

    out_schema = _schema_for(
        args.stage,
        args.keep_debug_metadata,
        in_meta.extraction.d_model,
        multi_checkpoint_input=multi_checkpoint_input,
        joint_checkpoints=checkpoints if joint_output else None,
    )
    input_provenance = (
        _MULTI_INPUT_PROVENANCE_COLS if multi_checkpoint_input else _PROVENANCE_COLS
    )
    carry_cols = input_provenance + (_HEAVY_DEBUG_COLS if args.keep_debug_metadata else [])
    storage.ensure_parent(args.output)
    row_count = 0

    # The selected activation column + carry_cols are never numerically
    # transformed, so pass them through as Arrow arrays instead of
    # round-tripping through to_pylist → from_pylist. At 4096 rows × 3584
    # floats that was 14.7M Python objects per batch, ~3.6B across the run.
    # Only api_explanation (small string col) gets materialized.
    missing_carry = [column for column in carry_cols if column not in in_col_names]
    assert not missing_carry, (
        f"input missing required provenance columns {missing_carry}; available: {in_col_names}"
    )
    with pq.ParquetWriter(storage.open_write(args.output), out_schema) as writer:
        for batch in tqdm(in_pf.iter_batches(batch_size=_CHUNK_SIZE), desc="chunks",
                          total=(in_pf.metadata.num_rows + _CHUNK_SIZE - 1) // _CHUNK_SIZE):
            match args.stage:
                case "av_sft":
                    built = _build_av_sft_cols(batch, actor_prompt_content)
                case "ar_sft":
                    assert suffix_ids is not None
                    built = _build_ar_sft_cols(batch, args.critic_template, suffix_ids, tokenizer)
                case "rl":
                    built = _build_rl_cols(batch, actor_prompt_content)
                case _:
                    raise AssertionError(f"unreachable: stage={args.stage!r}")

            if joint_output:
                for checkpoint, source_col in zip(
                    checkpoints, activation_source_cols, strict=True
                ):
                    built[checkpoint.column_name] = batch.column(source_col)
            else:
                built["activation_vector"] = batch.column(activation_source_cols[0])
            for col in carry_cols:
                built[col] = batch.column(col)
            if multi_checkpoint_input and not joint_output:
                checkpoint = checkpoints[0]
                n = len(batch)
                built["activation_checkpoint"] = pa.array(
                    [checkpoint.name] * n, type=pa.string()
                )
                built["activation_depth"] = pa.array(
                    [checkpoint.depth] * n, type=pa.int64()
                )
                built["activation_layer"] = pa.array(
                    [checkpoint.layer_index] * n, type=pa.int64()
                )

            writer.write_table(pa.table(built, schema=out_schema))
            row_count += len(batch)

    # Derive dataset_id from the input's — preserves stage0's corpus/slice hash
    # so two runs from different corpora on the same model/layer don't collide.
    dataset_suffix = in_meta.dataset_id.removeprefix("base_")
    if joint_output:
        dataset_suffix = f"joint_{dataset_suffix}"
    elif multi_checkpoint_input:
        dataset_suffix = f"{checkpoints[0].name}_{dataset_suffix}"
    output_extraction = (
        replace(in_meta.extraction, layer_index=None, checkpoints=checkpoints)
        if joint_output
        else replace(
            in_meta.extraction,
            layer_index=checkpoints[0].layer_index,
            checkpoints=[checkpoints[0]],
        )
    )
    out_meta = replace(
        in_meta,
        dataset_id=f"{args.stage}_{dataset_suffix}",
        stage=args.stage,
        row_count=row_count,
        extraction=output_extraction,
        keep_debug_metadata=args.keep_debug_metadata,
        tokens=token_meta,
        prompt_templates={"actor": actor_template, "critic": args.critic_template},
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage3_build",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"wrote {row_count} rows ({args.stage}) → {args.output}")
    if joint_output:
        print("checkpoints: " + ", ".join(
            f"{checkpoint.name}(depth={checkpoint.depth})"
            for checkpoint in checkpoints
        ))
    else:
        checkpoint = checkpoints[0]
        print(f"checkpoint: {checkpoint.name} (depth={checkpoint.depth}, "
              f"layer_index={checkpoint.layer_index}, pre-final-norm)")
    print(f"injection: {token_meta.injection_char!r} id={inj_id} "
          f"neighbors=({token_meta.injection_left_neighbor_id}, {token_meta.injection_right_neighbor_id})")
    if args.stage == "ar_sft":
        print(f"critic_suffix_ids: {suffix_ids} (extraction at tokens[-1])")


if __name__ == "__main__":
    main()
