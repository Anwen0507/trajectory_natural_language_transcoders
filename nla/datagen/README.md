# NLA Data Generation Pipeline

Generates the three training parquets (`av_sft`, `ar_sft`, `rl`) + sidecars that the NLA training side (`nla/config.py`, `nla/data_source.py`) reads.

Full design: [docs/design.md](../../docs/design.md) §0.

## Multi-checkpoint extraction

Stage 0 can capture several residual-stream checkpoints in one model forward.
Checkpoint depths count completed decoder blocks: depth `0` is the embedding
output and depth `N` is the output after `N` blocks. For Qwen2.5-0.5B's 24
blocks, this captures the embedding and every four-block boundary, including
the block-24 output before final RMSNorm:

```yaml
model: qwen05b
checkpoint_depths: [0, 4, 8, 12, 16, 20, 24]
```

The Stage-0 parquet keeps one row per sampled token and writes aligned columns
`activation_embedding`, `activation_block_04`, ..., `activation_block_24`.
Stages 1 and 2 retain those columns. Stage 3 keeps the ordered bundle for the
joint AV-SFT and RL actor files at `av_sft.parquet` and `rl.parquet`; their
prompts contain one `<INJECT>` position per checkpoint. AR remains
single-checkpoint for now, so Stage 3 writes one AR file under
`checkpoints/{checkpoint}/ar_sft.parquet`.

The joint RL parquet is an actor-side artifact only for now. Do not launch
joint-checkpoint RL until AR/reward/loss are generalized; legacy `K=1` RL is
unchanged.

Collection and parquet materialization support the embedding checkpoint. The
current critic bootstrap assumes its target follows at least one decoder block,
so training an embedding-target AR critic additionally requires a zero-block
critic mode; the block-4 through block-24 datasets use the existing critic
path.

## Config-driven run (recommended)

One YAML, all stages. See `configs/datagen/qwen7b_fineweb_1M.yaml` for a worked example (100k docs × 10 positions → ~1M vectors, split 25/25/50).

```bash
export PYTHONPATH=/path/to/natural_language_autoencoders:${PYTHONPATH:-}
python -m nla.datagen.run_pipeline --config configs/datagen/qwen7b_fineweb_1M.yaml

# Resume from a specific stage (e.g. after fixing an API error):
python -m nla.datagen.run_pipeline --config ... --stages 2,3
```

Output paths are derived from `config.output_dir`. Joint AV/RL and legacy
single-layer files land at the root; checkpoint-specific AR files land under
`checkpoints/{checkpoint}/`. Every subprocess command is printed before
running, so you can always re-run a single stage by hand.

## Quick start (manual, stage-by-stage)

```bash
export PYTHONPATH=/path/to/natural_language_autoencoders:${PYTHONPATH:-}

OUT=/tmp/nla_run
MODEL=Qwen/Qwen2.5-7B-Instruct

# Stage 0: extract activations from corpus (GPU required)
python -m nla.datagen.stage0_extract \
    --base-model $MODEL \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --layer-index 20 \
    --output $OUT/base.parquet

# Multi-checkpoint alternative (Qwen2.5-0.5B):
python -m nla.datagen.stage0_extract \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --checkpoint-depths 0 4 8 12 16 20 24 \
    --output $OUT/base.parquet

# Stage 1: three-way document-level split
python -m nla.datagen.stage1_split \
    --base $OUT/base.parquet \
    --av-sft-frac 0.3 --ar-sft-frac 0.3 --rl-frac 0.4 \
    --output-dir $OUT/splits

# Stage 2: API explanations (SL subsets only — RL doesn't need them)
export ANTHROPIC_API_KEY=sk-...
python -m nla.datagen.stage2_api_explain \
    --input $OUT/splits/av_sft_raw.parquet \
    --output $OUT/splits/av_sft_explained.parquet
python -m nla.datagen.stage2_api_explain \
    --input $OUT/splits/ar_sft_raw.parquet \
    --output $OUT/splits/ar_sft_explained.parquet

# Stage 3: build training-ready parquets. For multi-checkpoint inputs, omitting
# --checkpoint-depth builds the joint AV/RL bundle; AR still requires a selected
# depth, e.g. --checkpoint-depth 24.
python -m nla.datagen.stage3_build \
    --input $OUT/splits/av_sft_explained.parquet --stage av_sft --output $OUT/av_sft.parquet
# Legacy/single-checkpoint AR:
python -m nla.datagen.stage3_build \
    --input $OUT/splits/ar_sft_explained.parquet --stage ar_sft --output $OUT/ar_sft.parquet
# Multi-checkpoint AR alternative (repeat for each desired depth):
python -m nla.datagen.stage3_build \
    --input $OUT/splits/ar_sft_explained.parquet --stage ar_sft --checkpoint-depth 24 \
    --output $OUT/checkpoints/block_24/ar_sft.parquet
python -m nla.datagen.stage3_build \
    --input $OUT/splits/rl_raw.parquet --stage rl --output $OUT/rl.parquet

# Optional: shuffle rows before training (breaks position-within-doc clustering)
python -m nla.datagen.stage_shuffle \
    --input $OUT/av_sft.parquet --output $OUT/av_sft_shuf.parquet --seed 42
```

## Multi-GPU extraction

Stage 0 is the bottleneck (full forward pass per doc). The default `HFExtractor` uses `device_map="auto"` — model parallelism, not data parallelism — so it won't scale throughput across GPUs on its own.

`scripts/datagen/stage0_multigpu.sh` wraps stage 0 in data-parallel mode: one process per GPU, each bound via `CUDA_VISIBLE_DEVICES`, each processing a disjoint `--corpus-start` slice. Stage 0's per-doc keyed RNG means the merged output is row-for-row identical to a single serial run.

```bash
# Same args as stage0_extract. Auto-detects GPU count (override with NGPU env var).
scripts/datagen/stage0_multigpu.sh \
    --base-model $MODEL \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --layer-index 20 \
    --output $OUT/base.parquet
```

Shards are written to `$OUT/base.parquet.shards/shard_{i}.parquet` (with per-shard `.log` files) and merged via `nla.datagen.merge_base` into the final output. Shard files are left in place after merging for debugging.

`merge_base` is also usable standalone for merging shards produced across multiple nodes — it validates that all shards share the same extraction params and cover a contiguous corpus slice:

```bash
python -m nla.datagen.merge_base \
    --inputs node0/base.parquet node1/base.parquet node2/base.parquet \
    --output merged/base.parquet
```

## Stage-by-stage

| Stage | Input | Output | Notes |
|---|---|---|---|
| **0: extract** | HF corpus + model | `base.parquet` | Forward model, grab one or more hidden states at N positions/doc. RAW vectors (no normalization — that's training-time). Per-doc keyed RNG: same `(seed, doc_id)` → same positions. |
| **1: split** | `base.parquet` | 3 subset parquets | Document-level partition (all rows from same doc go to same bucket). Default 30:30:40. |
| **2: explain** | SL subset | +`api_explanation` col | Calls Anthropic API with the NLA instruction prompt (2-3 features, `<analysis>` tags). Strict extract: requires closing tag. Bullet cleanup (strip `- * 1.`). Drops rows with <2 features. |
| **3: build** | subsets | training parquets | av_sft: `prompt` (one ordered `<INJECT>` per checkpoint), `response` (`<explanation>...`). ar_sft: selected `activation_vector`, prompt ends with `<summary>`. rl: joint actor prompt/bundle only. Provenance always carried. |
| **shuffle** | any parquet | shuffled | Row permutation via `pyarrow.take()`. Keyed on `(seed, dataset_id)`. |
| **shuffle_activations** | any stage3 output | baseline | Permutes the singular vector or the entire joint checkpoint bundle with one shared row permutation — prompts/responses fixed. |

## Output schemas

**Training parquets** (`av_sft`/`ar_sft`/`rl`):

| Column | Type | Notes |
|---|---|---|
| `prompt` | `list[struct]` (av_sft/rl) or `str` (ar_sft) | `<INJECT>` literal for av_sft/rl — training-side `NLADataSource` swaps for the injection char |
| `response` | `str` | av_sft only, `<explanation>\n...\n</explanation>` wrapped |
| `activation_vector` | `list[float32]` | Legacy/selected-checkpoint RAW hidden state |
| `activation_{checkpoint}` | `list[float32]` | Joint AV/RL only: one RAW vector column per sidecar checkpoint, in the same order as prompt sites |
| `n_raw_tokens`, `activation_layer`, `doc_id` | provenance | always carried |
| `detokenized_text_truncated` | heavy debug | gated on `--keep-debug-metadata` (default on) |

Selected single-checkpoint files derived from multi-checkpoint extraction also
carry `activation_checkpoint`, `activation_depth`, and nullable
`activation_layer` provenance. Joint files encode checkpoint identity in their
named columns and ordered sidecar list instead of repeating it per row.

**Sidecar** (`{parquet}.nla_meta.yaml`):

| Field | Notes |
|---|---|
| `extraction.{base_model,d_model,checkpoints,norm}` | `checkpoints` records ordered names/depths and whether final norm was applied; `norm` is always `"none"` from datagen. Legacy single-layer sidecars retain `layer_index`. |
| `tokens.injection_{char,token_id,left_neighbor_id,right_neighbor_id}` | training hook scans for these |
| `tokens.critic_suffix_ids` | ar_sft only — expected tail token IDs, training verifies then extracts at `tokens[-1]` |
| `prompt_templates.{actor,critic}` | training MUST use these exact strings |

## Swapping backends

Every pluggable component is loaded via `--*-cls` import path. The shipped implementations are local-filesystem `LocalStorage` and the public-API `AnthropicProvider`; for cloud storage or alternative LLM APIs, subclass `nla.datagen.storage.Storage` / `nla.datagen.providers.CompletionProvider` and point at your class:

```bash
# Cloud storage (S3/GCS) — bring your own
--storage-cls my.module.GCSStorage

# Alternative completion provider — bring your own
--provider-cls my.module.OpenAIProvider \
--provider-kwargs '{"model": "gpt-4o", "concurrency": 50}'

# Custom extraction engine (e.g. vLLM server)
--extractor-cls my.module.VLLMExtractor \
--extractor-kwargs '{"url": "http://localhost:8000"}'
```

## Smoke test

`configs/datagen/quick_test_10docs.yaml` runs the multi-checkpoint pipeline
end-to-end on 10 docs. Run on a GPU box with `ANTHROPIC_API_KEY` set and
`transformers>=4.37.0`.
