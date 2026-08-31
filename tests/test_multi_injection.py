import unittest
from unittest.mock import patch

import torch

from nla.datagen.injection_tokens import build_token_meta
from nla.injection import inject_at_marked_positions
from nla.schema import compute_canonical_neighbors, normalize_activation


class _Tokenizer:
    def __init__(self, ids):
        self.ids = ids

    def apply_chat_template(self, *_args, **_kwargs):
        return list(self.ids)


class MultiInjectionTest(unittest.TestCase):
    def setUp(self):
        self.left = 7
        self.marker = 8
        self.right = 9
        # Two samples, three checkpoint markers per sample.
        self.ids = torch.tensor([
            [1, 7, 8, 9, 2, 7, 8, 9, 3, 7, 8, 9, 4],
            [5, 7, 8, 9, 6, 7, 8, 9, 1, 7, 8, 9, 2],
        ])
        self.vectors = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)

    def test_injects_flattened_sample_checkpoint_order(self):
        embeddings = torch.full((*self.ids.shape, 4), -1.0)
        actual = inject_at_marked_positions(
            self.ids,
            embeddings,
            self.vectors,
            self.marker,
            self.left,
            self.right,
        )

        marker_positions = [2, 6, 10]
        for batch_index in range(2):
            for checkpoint_index, position in enumerate(marker_positions):
                vector_index = batch_index * 3 + checkpoint_index
                torch.testing.assert_close(
                    actual[batch_index, position], self.vectors[vector_index]
                )
        # The helper clones and changes marker rows only.
        self.assertTrue(torch.all(embeddings == -1))
        self.assertTrue(torch.all(actual[:, 0] == -1))

    def test_sequence_parallel_slice_keeps_global_vector_indexing(self):
        # Slice [4:9) contains only the middle marker from each logical row.
        embeddings = torch.full((2, 5, 4), -1.0)
        actual = inject_at_marked_positions(
            self.ids,
            embeddings,
            self.vectors,
            self.marker,
            self.left,
            self.right,
            seq_slice=(4, 9),
        )
        torch.testing.assert_close(actual[0, 2], self.vectors[1])
        torch.testing.assert_close(actual[1, 2], self.vectors[4])

    def test_sequence_parallel_slice_with_no_local_markers_is_unchanged(self):
        embeddings = torch.full((2, 1, 4), -1.0)
        actual = inject_at_marked_positions(
            self.ids,
            embeddings,
            self.vectors,
            self.marker,
            self.left,
            self.right,
            seq_slice=(0, 1),
        )
        torch.testing.assert_close(actual, embeddings)

    def test_invalid_tensor_and_sequence_parallel_shapes_fail_before_write(self):
        embeddings = torch.zeros((*self.ids.shape, 4))
        cases = [
            (
                {"embeddings": embeddings[:, :-1]},
                "batch dims must match",
            ),
            (
                {"embeddings": embeddings[:1, :5], "seq_slice": (4, 9)},
                "batch dim mismatch",
            ),
            (
                {"embeddings": embeddings[:, :4], "seq_slice": (4, 9)},
                "spans 5 positions",
            ),
            (
                {"vectors": torch.zeros(6, 3)},
                "vectors must be",
            ),
            (
                {"vectors": torch.zeros(2, 3, 4)},
                "vectors must be",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                kwargs = {
                    "input_ids": self.ids,
                    "embeddings": embeddings,
                    "vectors": self.vectors,
                    "inj_id": self.marker,
                    "left_id": self.left,
                    "right_id": self.right,
                }
                kwargs.update(overrides)
                with self.assertRaisesRegex(AssertionError, message):
                    inject_at_marked_positions(**kwargs)

    def test_count_and_neighbor_mismatches_fail_loudly(self):
        embeddings = torch.zeros((*self.ids.shape, 4))
        with self.assertRaisesRegex(RuntimeError, "found 6 injection sites.*expected 5"):
            inject_at_marked_positions(
                self.ids,
                embeddings,
                self.vectors[:5],
                self.marker,
                self.left,
                self.right,
            )
        with self.assertRaisesRegex(RuntimeError, "found 0 injection sites"):
            inject_at_marked_positions(
                self.ids,
                embeddings,
                self.vectors,
                self.marker,
                left_id=99,
                right_id=self.right,
            )

    def test_edge_and_false_positive_markers_are_ignored(self):
        ids = torch.tensor([[8, 7, 8, 9, 8, 6, 9, 8]])
        embeddings = torch.full((1, 8, 2), -1.0, dtype=torch.float64)
        vector = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        actual = inject_at_marked_positions(
            ids,
            embeddings,
            vector,
            self.marker,
            self.left,
            self.right,
        )
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual[0, 2], vector[0].double())
        self.assertTrue(torch.all(actual[0, [0, 4, 7]] == -1))

    def test_distributed_count_failure_destroys_process_group(self):
        embeddings = torch.zeros((*self.ids.shape, 4))
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy,
            self.assertRaisesRegex(RuntimeError, "expected 5"),
        ):
            inject_at_marked_positions(
                self.ids,
                embeddings,
                self.vectors[:5],
                self.marker,
                self.left,
                self.right,
            )
        destroy.assert_called_once_with()

    def test_normalization_is_independent_per_checkpoint(self):
        bundle = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        normalized = normalize_activation(bundle, 10.0)
        torch.testing.assert_close(
            normalized.norm(dim=-1), torch.tensor([[10.0, 10.0]])
        )
        zero = normalize_activation(torch.zeros(1, 2, 3), 10.0)
        torch.testing.assert_close(zero, torch.zeros_like(zero))

    def test_canonical_prompt_validates_count_and_shared_neighbors(self):
        tokenizer = _Tokenizer([1, 7, 8, 9, 2, 7, 8, 9, 3])
        self.assertEqual(
            compute_canonical_neighbors(
                tokenizer,
                "{injection_char} {injection_char}",
                "x",
                8,
                expected_count=2,
            ),
            (7, 9),
        )
        with self.assertRaisesRegex(AssertionError, "expected 3"):
            compute_canonical_neighbors(
                tokenizer,
                "{injection_char}",
                "x",
                8,
                expected_count=3,
            )
        with self.assertRaisesRegex(AssertionError, "identical immediate token neighbors"):
            compute_canonical_neighbors(
                _Tokenizer([1, 7, 8, 9, 2, 6, 8, 9, 3]),
                "{injection_char} {injection_char}",
                "x",
                8,
                expected_count=2,
            )
        with self.assertRaisesRegex(AssertionError, "must be positive"):
            compute_canonical_neighbors(tokenizer, "", "x", 8, expected_count=0)
        for ids in ([8, 9], [7, 8]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(AssertionError, "at edge of sequence"):
                    compute_canonical_neighbors(
                        _Tokenizer(ids), "{injection_char}", "x", 8
                    )

    def test_token_meta_forwards_expected_multi_site_count(self):
        tokenizer = object()
        with (
            patch(
                "nla.datagen.injection_tokens.find_injection_token",
                return_value=("x", 8),
            ),
            patch(
                "nla.datagen.injection_tokens.compute_canonical_neighbors",
                return_value=(7, 9),
            ) as neighbors,
        ):
            meta = build_token_meta(
                tokenizer,
                "{injection_char} {injection_char}",
                expected_injection_sites=2,
            )
        neighbors.assert_called_once_with(
            tokenizer,
            "{injection_char} {injection_char}",
            "x",
            8,
            expected_count=2,
        )
        self.assertEqual(meta.injection_left_neighbor_id, 7)
        self.assertIsNone(meta.critic_suffix_ids)


if __name__ == "__main__":
    unittest.main()
