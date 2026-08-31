"""NLADataSource: parquet → Sample with activation vectors in metadata.

Handles both prompt formats:
  - AV-SFT / RL parquets: prompt is list[dict] (messages), response is separate column
  - AR-SFT parquets: prompt is a complete formatted string

Sets apply_chat_template=False (loss_mask generator needs list[dict] preserved).
Does the <INJECT> → ㊗ substitution here so downstream never sees the placeholder.
"""

import copy
import gc
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from miles.rollout.data_source import RolloutDataSource
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

from nla.config import load_nla_config, resolve_sidecar_source
from nla.schema import (
    ACTIVATION_COLUMN,
    ACTIVATIONS_KEY,
    INJECT_PLACEHOLDER,
)
from nla.storage import fetch_to_local_cache, is_remote


class NLADataSource(RolloutDataSource):
    def __init__(self, args):
        # Auto-fetch remote parquets. NLADataSource runs once in RolloutManager
        # (single process, CPU) so this is a one-time download, not per-worker.
        # After fetch, args.prompt_data points at the local cache — downstream
        # (sidecar resolution, miles read_file) sees a normal local path.
        if is_remote(args.prompt_data):
            storage_cls = getattr(args, "nla_storage_cls", None)
            assert storage_cls is not None, (
                f"prompt_data {args.prompt_data!r} is remote but --nla-storage-cls "
                f"not set. Either fetch manually (nla.scripts.fetch_parquet) or "
                f"set --nla-storage-cls to a Storage backend that handles the scheme."
            )
            args.prompt_data = fetch_to_local_cache(
                args.prompt_data,
                storage_cls=storage_cls,
                cache_dir=getattr(args, "nla_fetch_cache_dir", "/tmp/nla_fetch_cache"),
            )

        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        nla_cfg = load_nla_config(
            resolve_sidecar_source(
                explicit=getattr(args, "nla_sidecar_source", None),
                hf_checkpoint=args.hf_checkpoint,
                prompt_data=args.prompt_data,
            ),
            tokenizer,
        )
        inj_char = nla_cfg.injection_char
        checkpoint_names = tuple(
            getattr(nla_cfg, "activation_checkpoint_names", ("activation",))
        )
        expected_sites = len(checkpoint_names)

        # CRITICAL: miles' read_file does batch.to_pylist() which creates a Python
        # list[float] per activation column. 500k × 3584 = 1.8B PyFloat objects
        # per checkpoint → hangs the load for 10+ minutes + pollutes GC. Read
        # activation columns as pure numpy (flatten→reshape, zero Python-object
        # intermediate); read other columns via to_pylist (small strings/ints).
        t0 = time.perf_counter()
        assert "@[" not in args.prompt_data, (
            f"NLADataSource does not honor the @[start:end] slice syntax "
            f"({args.prompt_data!r}) — it would silently load the full file. "
            f"Slice the parquet upstream or pass the full path."
        )
        parquet_path = args.prompt_data
        pf = pq.ParquetFile(parquet_path)
        cols = pf.schema_arrow.names
        if ACTIVATION_COLUMN in cols:
            assert expected_sites == 1, (
                f"parquet {parquet_path!r} has legacy {ACTIVATION_COLUMN!r}, but "
                f"sidecar declares {expected_sites} checkpoints: {checkpoint_names}"
            )
            activation_cols = [ACTIVATION_COLUMN]
        else:
            activation_cols = [f"activation_{name}" for name in checkpoint_names]
            missing = [column for column in activation_cols if column not in cols]
            assert not missing, (
                f"parquet {parquet_path!r} missing joint activation columns {missing}; "
                f"sidecar order is {checkpoint_names}, available columns: {cols}"
            )
        activation_col_set = set(activation_cols)
        other_cols = [c for c in cols if c not in activation_col_set]

        samples = []
        for batch in pf.iter_batches(batch_size=16384):
            # FixedSizeList columns → [rows, K, d] numpy.  No Python floats;
            # checkpoint axis follows the sidecar's canonical order.
            activation_matrices = []
            for column in activation_cols:
                av_col = batch.column(column)
                av_flat = av_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
                activation_matrices.append(av_flat.reshape(len(av_col), -1))
            av = np.stack(activation_matrices, axis=1)
            assert av.shape[1] == expected_sites and av.shape[2] == nla_cfg.d_model, (
                f"activation bundle has shape {av.shape}, expected "
                f"[rows, {expected_sites}, {nla_cfg.d_model}]"
            )
            # other columns via to_pylist — small (prompts, strings, ints)
            rest = batch.select(other_cols).to_pylist()

            for i, (row, vec) in enumerate(zip(rest, av, strict=True)):
                prompt = row[args.input_key]
                if isinstance(prompt, list):
                    marker_count = sum(
                        m.get("content", "").count(INJECT_PLACEHOLDER)
                        for m in prompt
                    )
                    assert marker_count == expected_sites, (
                        f"row {len(samples)+i}: prompt contains {marker_count} "
                        f"{INJECT_PLACEHOLDER!r} markers, expected {expected_sites} "
                        f"for checkpoints {checkpoint_names}."
                    )
                    prompt = [
                        {**msg, "content": msg["content"].replace(INJECT_PLACEHOLDER, inj_char)}
                        for msg in prompt
                    ]
                else:
                    assert isinstance(prompt, str), (
                        f"row {len(samples)+i}: prompt must be list[dict] or str, got {type(prompt).__name__}"
                    )

                sample_meta: dict[str, object] = {ACTIVATIONS_KEY: vec}
                # Legacy critic/reward code consumes a single [d] vector. Keep
                # that contract only when K=1; never silently select one vector
                # from a joint bundle.
                if expected_sites == 1:
                    sample_meta[ACTIVATION_COLUMN] = vec[0]
                if "response" in row:
                    sample_meta["response"] = row["response"]
                for k in ("n_raw_tokens", "detokenized_text_truncated",
                          "activation_checkpoint", "activation_depth",
                          "activation_layer", "doc_id", "sample_uuid"):
                    if k in row:
                        sample_meta[k] = row[k]

                samples.append(Sample(prompt=prompt, metadata=sample_meta))

        print(f"[NLA] NLADataSource loaded {len(samples)} samples in {time.perf_counter()-t0:.1f}s")

        self.dataset = _ListDataset(samples, seed=args.rollout_seed)
        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)
        # Freeze the loaded dataset so future GC collections never scan it.
        # Belt-and-suspenders on top of the numpy fix above.
        gc.freeze()

    def get_samples(self, num_samples):
        if self.sample_offset + num_samples <= len(self.dataset):
            prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            prompt_samples = self.dataset.samples[self.sample_offset :]
            remaining = num_samples - len(prompt_samples)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            prompt_samples += self.dataset.samples[:remaining]
            self.sample_offset = remaining

        out = []
        for ps in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                s = copy.deepcopy(ps)
                s.group_index = self.sample_group_index
                s.index = self.sample_index
                self.sample_index += 1
                group.append(s)
            self.sample_group_index += 1
            out.append(group)
        return out

    def add_samples(self, samples: list[list[Sample]]):
        # sglang_rollout over-samples and aborts the excess once rollout_batch_size
        # is hit, then tries to re-queue the aborted prompts. Parent raises
        # RuntimeError (read-only). Roll back sample_offset so these prompts are
        # re-drawn next rollout — get_samples deepcopies, so originals are clean.
        self.sample_offset = max(0, self.sample_offset - len(samples))

    def load(self, rollout_id=None):
        # Miles' base load() looks at {args.load}/rollout/global_dataset_state_dict_{N}.pt.
        # Our GCS backup pushes iter_dir/ only (sibling rollout/ is missed), so on a
        # fresh-pod resume the state file lands INSIDE iter_dir (via _maybe_background_push's
        # snapshot). Copy it back to where base load() expects before delegating.
        if self.args.load is not None and rollout_id is not None:
            sibling = Path(self.args.load) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
            if not sibling.exists():
                in_iter = Path(self.args.load) / f"iter_{rollout_id + 1:07d}" / f"global_dataset_state_dict_{rollout_id}.pt"
                if in_iter.exists():
                    sibling.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(in_iter, sibling)
                    print(f"[NLA] restored sample_offset state from {in_iter} → {sibling}")
        super().load(rollout_id)

    def save(self, rollout_id):
        super().save(rollout_id)
        self._maybe_push_rollout_dumps()

    def _maybe_push_rollout_dumps(self):
        """Fire-and-forget GCS sync of rollout_dumps/ — same env-var gating
        as NLAFSDPActor._maybe_background_push. Runs at checkpoint intervals
        (RolloutManager.save → data_source.save). Storage backend is pluggable
        via NLA_BACKUP_STORAGE_CLS."""
        remote = os.environ.get("NLA_BACKUP_REMOTE")
        storage_cls = os.environ.get("NLA_BACKUP_STORAGE_CLS")
        dump_template = self.args.save_debug_rollout_data
        if not remote or not storage_cls or not dump_template:
            return
        dump_dir = Path(dump_template.format(rollout_id=0)).parent
        if not dump_dir.is_dir():
            return
        subprocess.Popen(
            [sys.executable, "-m", "nla.scripts.push_checkpoint",
             "--local", str(dump_dir),
             "--remote", f"{remote}/rollout_dumps",
             "--storage-cls", storage_cls,
             "--flat"],
            stdout=open("/tmp/push_rollout_dumps.log", "w"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"[NLA] rollout_dumps push fired: {dump_dir} → {remote}/rollout_dumps")


class _ListDataset:
    def __init__(self, samples: list[Sample], seed: int):
        self.origin_samples = samples
        self.samples = samples
        self.seed = seed
        self.epoch_id = -1

    def shuffle(self, new_epoch_id: int):
        if self.epoch_id == new_epoch_id:
            return
        r = random.Random(self.seed + new_epoch_id)
        perm = list(range(len(self.origin_samples)))
        r.shuffle(perm)
        self.samples = [self.origin_samples[i] for i in perm]
        self.epoch_id = new_epoch_id

    def __len__(self):
        return len(self.samples)
