# NLA-on-Miles: Design & Integration

**Goal:** Build NLA (Natural Language Autoencoder) training on top of Miles cleanly — using its extension points rather than monkey-patching, so we can pull upstream miles updates without constant merge hell.

---

## 0. Interface to Data-Gen Pipeline

The data-gen pipeline (activation extraction + API explanations) and NLA training are **fully decoupled**. The interface is parquet files + sidecar YAML. Data-gen does not need to know anything about miles or the `nla/` package.

### Parquet columns (per Stage 3a/b/c tables)

| Column | Type | Which stage | Notes |
|---|---|---|---|
| `prompt` | `list[dict]` (messages) or `str` | all | AV-SFT/RL: `[{"role":"user","content":"Explain: <concept><INJECT></concept>"}]`. AR-SFT: complete formatted critic string. |
| `response` | `str` | AV-SFT only | `f"<explanation>\n{api_explanation}\n</explanation>"` |
| `activation_vector` | `list[float]` len `d_model` | all | **RAW hidden states — no normalization.** Training normalizes at injection time per `extraction.injection_scale` / `mse_scale` in the sidecar. One dataset supports all norm experiments. |
| provenance: `n_raw_tokens`, `activation_layer`, `doc_id` | various | all | Always carried (cheap). |
| debug: `detokenized_text_truncated` | `str` | all (optional) | Gate on `keep_debug_metadata: bool` — default True, drop for prod. `skip_special_tokens=True` applied. |

**NOT in parquet (sidecar-only):** `injection_token_id`, neighbor IDs, `critic_suffix_ids` — these are dataset constants, shipped once in the sidecar, loaded once in `NLAFSDPActor.init`. Per-sample columns were dropped (v4).

### Sidecar YAML (`{parquet}.nla_meta.yaml`)

`nla/config.py`'s `load_nla_config` reads at startup, asserts against live tokenizer. Key fields:

| Field | Purpose |
|---|---|
| `kind` | `"nla_dataset"` (parquet sidecar) or `"nla_model"` (checkpoint sidecar, written by `write_model_sidecar`) |
| `extraction.d_model` | Asserted equal to `model.config.hidden_size` |
| `extraction.injection_scale` | L2-norm for injected vectors. `null` = inject raw, `"sqrt_d_model"` = default (ambient residual scale), float = custom. **Absent = `"sqrt_d_model"`**. |
| `extraction.mse_scale` | L2-norm BOTH pred+gold normalized to before MSE. `null` = raw MSE (critic learns magnitude too), `"sqrt_d_model"` = direction-only (default), float = custom. **Independent of `injection_scale`.** |
| `tokens.injection_token_id` + `injection_{left,right}_neighbor_id` | Hook scans for ID, verifies neighbors. Computed via `nla/schema.py:compute_canonical_neighbors`. |
| `tokens.critic_suffix_ids` | Sanity-check only — critic loss extracts at `tokens[-1]` (suffix-anchored, no scan) |
| `prompt_templates.{actor,critic}` | Training/inference MUST use these exact strings; drift = wrong injection position |
| `critic.num_hidden_layers` | Model sidecars only — extraction layer index K (config.json's `num_hidden_layers` is K+1) |

### Sidecar resolution precedence (`nla/config.py:resolve_sidecar_source`)

All load-sites (train_actor, data_source, and future nla_generate/nla_rm) MUST use this helper for consistent resolution:
1. `--nla-sidecar-source` (explicit)
2. `args.hf_checkpoint` **if** `{ckpt}/nla_meta.yaml` exists (model sidecar — authoritative for what THIS model was trained with)
3. `args.prompt_data` (dataset sidecar — fallback for fresh base-model runs)

**Model sidecar wins** when present — the baked `injection_scale`/`mse_scale` floats are what the model learned with; diverging = distribution shift.

### What data-gen does NOT need to know

- Miles' `Sample` dataclass, `multimodal_train_inputs` transport, anything in `miles/`
- `nla/data_source.py` (our code, not yet written) maps parquet columns → `Sample` — it adapts to whatever columns are present

### Useful for data-gen iteration

- Miles reads parquet natively (`miles/utils/data.py:48-58`, `pyarrow.ParquetFile.iter_batches()`). Emit standard parquet — no custom format.
- Miles supports `foo.parquet@[0:1000]` path-slicing for its stock data source, but `NLADataSource` does **not** (it bypasses Miles' loader to read `activation_vector` as zero-copy numpy). Slice the parquet upstream for smoke tests.

---

## 1. What Miles Gives Us (verified by code inspection)

### ✅ Ray-orchestrated, function-pointer extension system

Every customization point is a `--*-path` CLI arg that gets `load_function()`'d at runtime. This is how we plug in NLA without editing `miles/`:

| Arg | Purpose | Where loaded |
|---|---|---|
| `--rollout-function-path` | Replace the entire rollout loop | `ray/rollout.py:66-69` |
| `--custom-generate-function-path` | Replace per-sample generation (keep rollout orchestration) | `sglang_rollout.py:222` |
| `--custom-rm-path` | Custom reward function `async (args, samples) -> list[float]` | `rollout/rm_hub/__init__.py:31,86` |
| `--custom-loss-function-path` | Custom loss (when `--loss-type custom_loss`) | `training_utils/loss.py:890` |
| `--custom-convert-samples-to-train-data-path` | Convert `list[Sample] -> train_data dict` | `ray/rollout.py:75,359` |
| `--data-source-path` | Replace the `Dataset` loader | `ray/rollout.py:60` |

### ✅ SFT + RL in the same framework

| Mode | How | Evidence |
|---|---|---|
| **RL (GRPO/PPO)** | `--loss-type policy_loss`, rollout generates via SGLang | `loss.py:883`, default path |
| **SFT** | `--loss-type sft_loss`, rollout is a no-op that just tokenizes the dataset | `loss.py:887` + `rollout/sft_rollout.py` + `examples/formal_math/single_round/run_sft.py:32-45` (working example: `--debug-train-only` skips SGLang spinup) |
| **Custom** | `--loss-type custom_loss --custom-loss-function-path ...` | `loss.py:889` |

### ✅ SGLang with `input_embeds` (supported, not wired)

SGLang's HTTP `/generate` accepts `input_embeds` (stock SGLang feature, not miles-specific). Miles never sends it — all rollout code uses `input_ids` (generate_endpoint_utils.py:52). We build the embed-prep path ourselves via `--custom-generate-function-path`. The router and HTTP layer pass arbitrary payload keys through transparently (http_utils.py:273, router.py:149-157).

**Pitfall:** `update_sample_from_response` (generate_endpoint_utils.py:67-68) does `sample.tokens = payload["input_ids"]`. If payload only has `input_embeds` → KeyError. Our custom generate fn must populate `payload["input_ids"]` for bookkeeping even when sending embeds (SGLang accepts both; `input_embeds` overrides).

### ✅ Two training backends (we use FSDP)

| Backend | File | NLA fit |
|---|---|---|
| **FSDP2** | `miles/backends/fsdp_utils/actor.py` | **Use this.** Standard HF models, no PP, forward hooks just work. |
| **Megatron-LM** | `miles/backends/megatron_utils/actor.py` | Has `--custom-megatron-before-log-prob-hook-path` etc. but also all the PP complexity the arch doc says to leave behind. Skip. |

### ✅ Actor+critic on separate GPU pools (once unlocked)

`placement_group.py:132-166` creates separate `RayTrainGroup`s for actor and critic with independent GPU allocation (`--critic-num-nodes`, `--critic-num-gpus-per-node`), optimizers (`--critic-lr`), and checkpoints (`--critic-load`/`--critic-save`). `train.py:81-87` fires `critic_model.async_train(...)` in parallel with `actor_model.async_train(...)` — they run concurrently on their respective GPU pools, then join. Both receive the same `rollout_data_ref` (tokens + response + whatever we stashed in `multimodal_train_inputs`).

**The gate:** `args.use_critic = args.advantage_estimator == "ppo"` is a post-parse *assignment* (arguments.py:1713), not a CLI flag. GRPO forces `use_critic=False`. **We unlock this** (see §2). Once unlocked, miles' train loop orchestration is exactly what NLA needs — the critic slot's *contents* (PPO scalar-value head, GAE, `sync_actor_critic_data`) are irrelevant because `NLAFSDPActor.train()` dispatches on `self.role` and does its own thing.

**What GRPO doesn't need from the critic:** `values` / GAE returns. GRPO advantages come from group-normalized rewards (ray/rollout.py:333-349), computed at rollout time, independent of critic. The `sync_actor_critic_data` broadcast (training_utils/data.py:419) is a PPO-ism — we skip it. The critic just trains MSE on its own GPUs; actor never reads its outputs directly. Reward-scoring reads the critic's *saved checkpoint* (§4.3).

---

## 2. Required Upstream Changes (exactly two)

### 2.1 `--custom-actor-cls-path` — actor class is hardcoded

**`miles/ray/actor_group.py:75-84`**:
```python
if backend == "megatron":
    actor_impl = MegatronTrainRayActor
else:
    actor_impl = FSDPTrainRayActor
```

No extension point to inject `NLAFSDPActor`. Fix:

```python
# miles/utils/arguments.py — add:
parser.add_argument("--custom-actor-cls-path", type=str, default=None,
    help="Import path to a custom TrainRayActor subclass, e.g. 'nla.train_actor.NLAFSDPActor'")

# miles/ray/actor_group.py:75 — replace with:
if self.args.custom_actor_cls_path is not None:
    actor_impl = load_function(self.args.custom_actor_cls_path)
elif backend == "megatron":
    actor_impl = MegatronTrainRayActor
else:
    actor_impl = FSDPTrainRayActor
```

### 2.2 `--force-use-critic` — GRPO locks critic off

**`miles/utils/arguments.py:1713`**: `args.use_critic = args.advantage_estimator == "ppo"`. Not a CLI flag, post-parse assignment. GRPO → `use_critic=False` → no critic RayTrainGroup allocated.

Fix (keep the default, add an override):

```python
# miles/utils/arguments.py — add:
parser.add_argument("--force-use-critic", action="store_true",
    help="Enable critic training group regardless of advantage estimator. "
         "For custom critic training (e.g. NLA MSE critic) that doesn't feed GAE.")

# miles/utils/arguments.py:1713 — change to:
args.use_critic = args.advantage_estimator == "ppo" or args.force_use_critic
```

Both changes follow miles' existing patterns. Both are upstreamable. Alternative: monkey-patch at import time in our `train_nla.py` entrypoint — uglier but keeps miles/ pristine.

---

## 3. Data Transport: `Sample.multimodal_train_inputs`

### 3.1 Why not `train_metadata`

`Sample.train_metadata` → `train_data["metadata"]` (rollout.py:409-410) but **`_split_train_data_by_dp`'s key whitelist (rollout.py:444-457) doesn't include `"metadata"`**. It's silently dropped before reaching the training actor. Dead end.

### 3.2 `multimodal_train_inputs` flows end-to-end (verified)

`Sample.multimodal_train_inputs: dict[str, Tensor]` is plumbed through every stage:

| Stage | File:line | Behavior |
|---|---|---|
| Sample → train_data | `rollout.py:412-413` | `train_data["multimodal_train_inputs"] = [s.multimodal_train_inputs for s in samples]` |
| DP split | `rollout.py:446` | `"multimodal_train_inputs"` is in the whitelist |
| GPU move | `training_utils/data.py:37-46` | Every tensor value `.to(cuda)`. **All values must be Tensor** — no raw ints/lists. |
| Microbatch fetch | `actor.py:337,448` | `"multimodal_train_inputs"` is in `get_batch`'s key list for both `_compute_log_prob` and train |
| Concat | `training_utils/data.py:248-254` | Per-key `torch.cat(dim=0)` across samples in the microbatch |

**Shape constraint:** the concat at data.py:248-254 assumes `[num_items, ...]`-shaped tensors. Stash activation vectors as `tensor[1, d_model]` (not `[d_model]`) so concat produces `[B, d_model]` per microbatch.

**The one gotcha:** `_get_model_inputs_args` (actor.py:634-635) spreads the dict directly into `model(**kwargs)`:
```python
if batch.get("multimodal_train_inputs"):
    model_args.update(batch["multimodal_train_inputs"])
```
Unknown kwargs like `nla_activation` → `TypeError: forward() got unexpected keyword argument`. **We override `_get_model_inputs_args` anyway** (§5 — that's where injection state is set), and popping the NLA keys before `model_args.update()` is one line.

### 3.3 What we stash

**Actor modes (SFT + RL)** — fixed-shape only:
```python
sample.multimodal_train_inputs = {
    "nla_activation": torch.tensor(activation_vector, dtype=torch.float32).view(1, -1),  # [1, d_model]
}
```

After `get_batch`'s concat: `[B, d_model]`. This is the ONLY key the actor's hook needs.

**RL mode additionally stashes for the critic** — variable-length, so **1-D**:
```python
sample.multimodal_train_inputs["nla_critic_tokens"] = torch.tensor(critic_token_ids, dtype=torch.long)  # [seq_len_i]
```

1-D cat on dim=0 produces `[sum_seq_len]` — valid for arbitrary lengths. On the actor side this is useless junk (stripped before `get_batch` — see §5). On the critic side, `_train_critic` reads the pre-concat list-of-dicts from `rollout_data["multimodal_train_inputs"]` directly, so no unpacking needed.

**Token IDs (injection + neighbors) are NOT in multimodal transport.** They're dataset constants loaded once from the sidecar in `NLAFSDPActor.init()` via `nla/config.py`. Shipping them per-sample was the v2 design — dropped as gratuitous (two sources of truth for the same constant, the sidecar load path already exists).

### 3.4 `<explanation>` extraction failure path

Truncated generation → no closing `</explanation>` → regex miss. Asserting on this is wrong — in the rollout path (via `nla_generate`), `asyncio.gather` without `return_exceptions` means one assert kills the entire rollout batch.

**Handling:**
- `nla_generate`: on regex miss, set `sample.status = Sample.Status.FAILED`, skip the `nla_critic_tokens` stash. Sample still flows through (actor can still train on the tokens — GRPO doesn't care about `<explanation>` tags, only token-level log-probs + rewards).
- `nla_rm`: on regex miss, return `FAILED_EXTRACTION_REWARD` (the orthogonal-equivalent value: `-2.0` under `-mse_nrm`, `-log 2` under the log variant) so a truncated generation is never advantaged. Don't crash.
- `_train_critic`: skip samples where `"nla_critic_tokens" not in mm`. The critic just sees fewer samples that step.

The only assert-worthy case is at startup: first batch, 100% extraction failure → prompt template drift or wrong checkpoint. Check once, loud.

---

## 4. Training Modes (three separate `train.py` invocations)

### 4.1 Actor SFT (AV / decoder)

**Data:** AV-SFT parquet: `prompt`, `response`, `activation_vector`. Token IDs (injection + neighbors) come from the sidecar, not per-row.

```bash
python train.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --loss-type sft_loss \
    --debug-train-only \                       # skip SGLang — no generation for SFT
    --disable-compute-advantages-and-returns \ # skip advantage calc
    --rollout-function-path nla.rollout.sft_actor.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data $AV_SFT_PARQUET \
    --hf-checkpoint $INSTRUCT_MODEL \
    ...
```

| Step | Implementation |
|---|---|
| Dataset → `Sample` | `NLADataSource` with **`apply_chat_template=False`** — preserves `list[dict]` (AV-SFT's parquet `prompt` is messages). Substitute `<INJECT>` → `㊗` in the user message content HERE. `sample.prompt` = messages; `sample.metadata["response"]` = `<explanation>...</explanation>` string; `sample.metadata["activation_vector"]` = float list. |
| Rollout | `nla.rollout.sft_actor`: **append `{"role":"assistant","content":response}` to `sample.prompt`**, call `MultiTurnLossMaskGenerator.get_loss_mask(messages)` (same pattern as `sft_rollout.py:49` — requires `list[dict]`). Stash `{"nla_activation": tensor[1,d_model]}` in `sample.multimodal_train_inputs`. **No SGLang.** |
| Forward | Hook fires on embedding. `_get_model_inputs_args` pops `nla_activation` → `self._nla_vectors`. Hook scans `inputs[0]` for injection token (IDs from sidecar, loaded once in `init`). |
| Loss | Stock `sft_loss_function`. **Unchanged.** |

### 4.2 Critic SL (AR / encoder)

**Data:** AR-SFT parquet: `prompt` (complete formatted string ending with the fixed suffix `</text> <summary>`), `activation_vector`.

```bash
python train.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --loss-type custom_loss \
    --custom-loss-function-path nla.loss.nla_critic_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_critic.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data $AR_SFT_PARQUET \
    --hf-checkpoint $INSTRUCT_MODEL \
    --nla-model-is-critic \          # drives get_model_cls() → NLACriticModel
    --hf-checkpoint $CRITIC_INIT_CKPT \  # K+1 baked into config.json — no layer-count arg needed
    ...
```

**Prerequisite:** Run `nla/scripts/prepare_critic_checkpoint.py` once to produce `$CRITIC_INIT_CKPT` — a truncated checkpoint keeping layers 0..K inclusive, with `config.num_hidden_layers=K+1` baked into config.json + `nla_meta.yaml`. After this, `from_pretrained` loads K+1 layers naturally — no layer-count arg at train time.

| Step | Implementation |
|---|---|
| Dataset → `Sample` | `NLADataSource` with **`apply_chat_template=False`** — AR-SFT parquet's `prompt` is a complete formatted **string** (not messages), already ends with the fixed suffix. `sample.metadata["activation_vector"]` = RAW hidden state. |
| Rollout | `nla.rollout.sft_critic`: tokenize `prompt` string directly. `response_length=0`, `loss_mask=[]` (length-0). Stash `{"nla_activation": tensor[1,d_model]}` (raw) in `multimodal_train_inputs`. |
| Model | **`--nla-model-is-critic`** + `--hf-checkpoint $CRITIC_INIT_CKPT`. `get_model_cls()` → `NLACriticModel`; layer count (K+1) comes from `config.json` (self-describing checkpoint). Truncated layers, no final LN, `Linear(d, d)` head. |
| Loss | `nla.loss.nla_critic_loss`: extract `values` at `tokens[-1]` per sample in the packed stream (suffix-anchored — no scan) → `normalize_activation(pred, mse_scale)` + `normalize_activation(gold, mse_scale)` → MSE. Both raw from dataset; **both normalized here** (or neither if `mse_scale=null`). |

**No `--use-critic`.** The "actor" slot in miles-speak IS the critic model being trained. Single training group.

### 4.3 Actor RL (GRPO) + Online Critic

**Data:** RL parquet: `prompt`, `activation_vector`. No `response`. Token IDs come from the sidecar.

```bash
python train.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --loss-type policy_loss \
    --advantage-estimator grpo \
    --force-use-critic \                         # upstream change §2.2 — unlocks critic under GRPO
    --n-samples-per-prompt 8 \
    --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
    --custom-generate-function-path nla.rollout.nla_generate.generate \
    --custom-rm-path nla.reward.nla_rm \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data $RL_PARQUET \
    --hf-checkpoint $ACTOR_SFT_CKPT \
    --ref-load $ACTOR_SFT_CKPT \                 # SFT'd actor, NOT base model
    --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type k2 \  # k2, not Miles' default k1 (see configs/rl.sh)
    --critic-load $CRITIC_SL_CKPT \              # SL-trained critic checkpoint — critic RayTrainGroup starts here
    --critic-save $RUN_DIR/critic \
    --critic-lr 1.41e-5 \
    --critic-num-nodes $CRITIC_NODES --critic-num-gpus-per-node $CRITIC_GPUS \  # may differ from actor dp; _repartition_for_critic handles it
    --save-interval 10 \
    ...
```

**Two RayTrainGroups, both `NLAFSDPActor`, dispatched by `self.role`:**
- **Actor group** (`role="actor"`): `loss_type=policy_loss`, injection hook registered on model + ref_model, trains GRPO.
- **Critic group** (`role="critic"`): builds `NLACriticModel` (truncated + vector head), `loss_type=custom_loss` (MSE), no injection hook, no ref model.

The same `rollout_data_ref` goes to both (train.py:82-84). It contains `tokens` (prompt+response), `multimodal_train_inputs["nla_activation"]` (gold vectors). Actor uses tokens+activation for log-prob injection; critic uses response-tokens+activation for MSE target. One rollout, both consume it.

| Step | Actor group | Critic group |
|---|---|---|
| Rollout | `nla_generate`: embed → inject → SGLang `input_embeds` → response. Extract `<explanation>` payload (on failure: `sample.status = FAILED`, skip critic-tokens stash — see §3.4). Stash `{"nla_activation": tensor[1,d_model], "nla_critic_tokens": tensor[seq_len]}` in `multimodal_train_inputs`. **1-D** critic tokens — cat is valid. | (shares rollout output) |
| Reward | `nla.reward.nla_rm`: extract `<explanation>` (on failure: `FAILED_EXTRACTION_REWARD`, don't crash) → wrap in critic template → tokenize → forward the **live critic via Ray remote** (see below) → `-mse_nrm` (or `-log(MSE)` via `NLA_LOG_MSE_REWARD=1`). | runs `critic_fwd` on its idle GPUs |
| Train | `NLAFSDPActor.train()` when `role=="actor"`: runs `_train_core` (log-probs + GRPO). Injection hook fires per-microbatch via `_get_model_inputs_args`. | `NLAFSDPActor.train()` when `role=="critic"`: runs `_train_critic` — forward on `nla_critic_tokens`, extract at last-token position, MSE against `nla_activation`. Separate optimizer, separate LR. |
| Sync | `actor_model.update_weights()` → SGLang (train.py:97). | `critic_model.save_model(rollout_id)` every `save_interval` (train.py:58-61). **No direct critic→actor sync** — `sync_actor_critic_data` is PPO plumbing we skip. |

**Reward critic = the live critic, via Ray remote.** Reward is computed at
rollout time, when the critic-trainer GPUs are idle (generation runs on
SGLang's GPUs). `train.py` stashes the critic RayTrainGroup's actor handles on
`args._nla_critic_handles`; `nla_rm` calls `handle.critic_fwd.remote(ids, mask)`
on each, takes rank 0's result. No snapshot, no checkpoint reload, no
staleness — the reward always uses the current critic weights. An async
accumulator coalesces concurrent samples into one batched `critic_fwd` so the
collective fires once per `--nla-reward-batch-size` instead of once per sample.

- **Reward MSE uses the CRITIC'S `mse_scale`** (from its sidecar), NOT the
  dataset's. Use `normalize_activation(pred, cfg.mse_scale)` and
  `normalize_activation(gold, cfg.mse_scale)` — the same function the training
  loss uses.
- Gold activation comes RAW from `sample.metadata["activation_vector"]`;
  normalised inside `_mse_to_reward`.

**Skipped PPO critic machinery:**
- `sync_actor_critic_data` (training_utils/data.py:419) — broadcasts `values` for GAE. GRPO doesn't use values. `NLAFSDPActor.train()` skips the call.
- `compute_advantages_and_returns` reads `values` only when `advantage_estimator=="ppo"` (check loss.py:332-393). GRPO path computes from rewards alone. Already correct — no change needed.
- `actor_model.connect(critic_model)` (placement_group.py:160) creates an actor↔critic NCCL group for the broadcast. Harmless if unused; we can leave it or gate on `args.advantage_estimator == "ppo"`.

---

## 5. `NLAFSDPActor` — the core subclass

**Source of truth: `nla/train_actor.py`.** This section documents invariants and design rationale, not the full listing.

### Dispatch dimensions (orthogonal)

| Dimension | Values | Gates |
|---|---|---|
| `self._is_critic_model` (set before `super().init()`) | True / False | Model class, `train()` path (skip `_compute_log_prob` which needs `.logits`), `_train_step` branch |
| `self.role` | "actor" / "critic" | **Only** the token-swap inside the critic path (`_swap_to_critic_tokens`). Critic-SL standalone is `role=="actor"` but `_is_critic_model=True`. |

### Method overrides

| Override | Purpose |
|---|---|
| `init()` | Critic role: rewire hf_checkpoint/save/lr, set `nla_model_is_critic`. Set `_is_critic_model`. Assert `cp_size==1`. Detect asymmetric DP (actor_dp!=critic_dp → stash `_nla_actor_dp` for `_repartition_for_critic`). Load `_nla_cfg` via `resolve_sidecar_source(args)`. Assert `d_model == hf_config.hidden_size`. Register injection hook (LM-actor only, on model + ref). |
| `get_model_cls()` | `_is_critic_model` → `NLACriticModel` (K+1 layers from checkpoint's config.json). Else super. |
| `_get_model_inputs_args()` | Pop `nla_activation` from multimodal dict. **Actor**: `normalize_activation(popped, injection_scale)` → `self._nla_vectors` (hook reads). **Critic**: stash RAW to `batch["nla_activation"]` + `batch["nla_mse_scale"]` (loss normalizes both sides). Per-microbatch, single state-setting point for both `_compute_log_prob` and `_train_step`. |
| `_train_step()` | Critic branch: `values = self.model(...).values.float()` → `loss_function`. Else super. |
| `train()` | Critic-model: `_swap_to_critic_tokens` (if `role=="critic"`) → `_train_critic_loop`. LM-actor: strip `nla_critic_tokens` from MM → `_train_core`. Wrapped in `timer` + `inverse_timer("train_wait")` + `log_perf_data_raw`. |
| `save_model()` | Super (DCP). Then **all ranks** `get_model_state_dict` (COLLECTIVE — rank-0-only would deadlock). Rank-0: critic → `save_pretrained` to `iter_{rollout_id+1}/hf/` (matches checkpoint.py:199's +1 convention), both → `_write_sidecar`. Barrier. |

### `_swap_to_critic_tokens` + `_train_critic_loop`

The only in-the-guts code. Miles' `_train_core` unconditionally calls `_compute_log_prob` (needs `.logits`); `_train_critic_loop` is the minimal replacement (iterator + get_batch + train_step, with logging). `_swap_to_critic_tokens` rewires `rollout_data` — swaps `tokens`/`total_lengths`/`response_lengths`/`loss_masks` to critic-token versions, filters `nla_critic_tokens`-missing samples, **drops** parallel-but-unfiltered keys (`rewards`, `max_seq_lens`, etc.) so `log_rollout_data` / `DataIterator` don't hit length-mismatch.

**Why scan inside the hook** instead of precomputing positions: miles reorders samples twice — once at `_split_train_data_by_dp` via `get_seqlen_balanced_partitions` (rollout.py:434), again at `get_data_iterator` for micro-batch balancing (training_utils/data.py:403). Then `get_batch` packs them into a fresh `cu_seqlens` stream per microbatch (data.py:163-167). Any position computed before these reorders is garbage. The hook's `inputs[0]` IS the exact `input_ids` the model sees — scan-by-token-ID there is correct by construction and padding-agnostic.

**Vector→position mapping inside the hook:** after `get_batch`'s concat, `self._nla_vectors[i]` corresponds to the i-th sample *in microbatch order*. The packed token stream `inputs[0]` is also in microbatch order (same `torch.cat` iteration, data.py:167 + data.py:251). So iterating `matches` in seq-position order (`.nonzero()` returns sorted) and consuming `_nla_vectors` in order `[0, 1, 2, ...]` is correct. The count assertion catches any drift.

**Why `_get_model_inputs_args`** for state setting: it's called exactly once per microbatch, immediately before `model(**args)`, in *both* `_compute_log_prob` (actor.py:351) and `_train_step` (actor.py:524). Overriding here covers both paths with one function. The v1 design's `_train_step` override was wrong for `_compute_log_prob` (which has its own internal micro-batch loop at actor.py:326-370 that v1's "set before loop, clear after" wrapping misses).

**State clearing:** the hook is a no-op when `self._nla_vectors is None`. State is overwritten on the next `_get_model_inputs_args` call. If paranoid: add `self._nla_vectors = None` at the end of `_train_core`.

**FSDP2 safety:** `fully_shard` (actor.py:702) modifies modules in-place — `model.get_input_embeddings()` returns the original `nn.Embedding`. Forward hooks see the full post-gather output tensor. For tied-embedding models the embedding isn't wrapped as its own unit (actor.py:679 gate) but lives inside the top-level unit; hook still fires correctly — the parent gathers params before children's forwards.

**Sequence packing preserves neighbors (cp_size=1 only):** In `thd` mode with `cp_size==1`:
- `slice_with_cp` is a no-op (cp_utils.py:202-206 returns tokens unchanged)
- `torch.cat(tokens, dim=0)` (data.py:167) concatenates full samples contiguously — no intra-sample splitting
- Padding added only at the end of the packed stream (data.py:169-173), not between samples
So within the packed stream, each sample's `[..., left_id, inj_id, right_id, ...]` triple is intact. The boundary between sample N's last token and sample N+1's first token is a *different* pair than `(left_id, inj_id)` — sample boundaries are at end-of-response/start-of-prompt, neither of which is `>㊗` or `㊗</`. False positives at boundaries are theoretically possible if the last response token of sample N happens to equal `left_id` AND the first prompt token of sample N+1 equals `inj_id` — but `inj_id` is specifically chosen to be rare-in-corpus, and prompts all start with the same chat-template prefix (BOS or `<|im_start|>`), not `㊗`. The count assertion catches this if it ever happens.

**With `cp_size>1`**: `slice_with_cp` splits each sample into 2 non-contiguous chunks per rank (ring-attention layout, cp_utils.py:32-33: `chunk_0 = [rank*sz, (rank+1)*sz)`, `chunk_1 = [(2*cp-rank-1)*sz, (2*cp-rank)*sz)`). The injection token and its neighbors can land on different ranks. The hook on rank 0 wouldn't see all three adjacent → neighbor check fails or worse, finds spurious matches. **NLA sequences are short** (prompt ~50 tokens + response ~300 tokens); CP is for multi-K-token sequences. Assert `cp_size==1` in `init` and move on.

**Monitoring signal:** if injection breaks (hook didn't fire, wrong position, vector not set), the model sees literal `㊗` and produces Chinese/CJK text. Every `save_interval`, run a 3-sample eval and grep for CJK characters in the output — loud smoke test that catches the entire class of injection bugs, regardless of root cause.

---

## 6. `nla/` Package Layout

Miles is installed as a separate package (see `docs/setup.md`), not vendored here. The two upstream patches (§2) are applied to the installed `miles` package.

```
natural_language_autoencoders/
├── nla/
│   ├── __init__.py
│   ├── schema.py                   # Shared: NLATokenMeta, sidecar_path_for, normalize_activation,
│   │                               #   compute_canonical_neighbors, resolve_target_scale, SCALE_SQRT_D
│   ├── config.py                   # NLAConfig dataclass + load_nla_config (reads sidecar, asserts
│   │                               #   token IDs/neighbors against live tokenizer) + write_model_sidecar
│   │                               #   + resolve_sidecar_source (precedence: explicit > ckpt > dataset)
│   ├── models.py                   # NLACriticModel: truncated transformer (K+1 layers from config.json),
│   │                               #   no final LN (Identity swap), Linear(d,d) head, HF-compatible save/load
│   ├── data_source.py              # NLADataSource: parquet → Sample, <INJECT>→㊗, raw activation_vector
│   ├── loss.py                     # nla_critic_loss — MSE at last-token pos, normalize BOTH to mse_scale (or raw)
│   ├── train_actor.py              # NLAFSDPActor (§5) — hook, dispatch, critic-loop, save w/ sidecar
│   ├── rollout/
│   │   ├── sft_actor.py            # Actor-SFT rollout (no generation)
│   │   ├── sft_critic.py           # Critic-SL rollout (no generation)
│   │   └── nla_generate.py         # RL: embed → inject → SGLang input_embeds
│   │                               #   Reloads fresh embedding from actor dump each rollout
│   ├── reward.py                   # nla_rm: extract <explanation> → wrap → critic fwd → -mse_nrm
│   │                               #   Load critic + sidecar from iter_*/hf/ (fallback: SL ckpt)
│   ├── scripts/
│   │   └── prepare_critic_checkpoint.py  # One-time: base → truncated ckpt w/ config.json(K+1) + sidecar
│   └── datagen/                    # Activation extraction + API explanations — shared schema in nla/schema.py
└── configs/
    ├── actor_sft.sh
    ├── critic_sft.sh               # Requires prepare_critic_checkpoint.py first
    └── rl.sh
```

---

## 7. Build Order


1. ✅ `nla/schema.py` + `nla/config.py` + `nla/models.py` — pure, no miles dependency.
2. ✅ Upstream patches — `--custom-actor-cls-path` + `--force-use-critic` + `--nla-*` args.
3. ✅ `nla/scripts/prepare_critic_checkpoint.py` — one-time base→truncated prep.
4. ✅ `nla/data_source.py` + `nla/rollout/sft_critic.py` + `nla/loss.py` — Critic-SL end-to-end.
5. ✅ `nla/train_actor.py` + `nla/rollout/sft_actor.py` — Actor-SFT end-to-end. Hook + state management. **Debug signal: if injection fails, output is Chinese.**

6. ✅ `nla/rollout/nla_generate.py` — SGLang `input_embeds`. CLI `--nla-injection-scale` resolved (mirrors train_actor). Embedding refreshes via actor dump (`update_weights` override → `{save}/_nla_rollout_embed.pt`).
7. ✅ `nla/reward.py` — `-mse_nrm (default; -log(MSE) via NLA_LOG_MSE_REWARD=1)`. `FAILED_EXTRACTION_REWARD` (orthogonal-equivalent).
8. ✅ **Online critic** — `--force-use-critic` ON. Reward calls the live critic via Ray remote (`critic_fwd`), so it always uses current weights. This is the ONLY supported RL mode — frozen-critic was never implemented.

---

