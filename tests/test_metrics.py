import json
import math
import unittest

import numpy as np

from looplab.metrics import (
    aggregate_hidden_proximity,
    aggregate_language_model_metrics,
    aggregate_logits_proximity,
    hidden_proximity,
    language_model_metrics,
    log_softmax,
    logits_proximity,
    paired_accuracy,
    paired_bootstrap,
    routing_proximity,
)


class LanguageModelMetricTests(unittest.TestCase):
    def test_log_softmax_is_translation_invariant_and_stable(self):
        result = log_softmax([[1000.0, 1000.0], [-1000.0, -1000.0]])
        np.testing.assert_allclose(result, -math.log(2.0), atol=1e-12)
        np.testing.assert_allclose(np.exp(result).sum(axis=-1), 1.0)

    def test_teacher_forcing_uses_next_token_not_current(self):
        logits = np.log([[0.1, 0.9], [0.2, 0.8], [0.99, 0.01]])
        result = language_model_metrics(logits, [0, 1, 0])
        self.assertEqual(result["token_count"], 2)
        self.assertAlmostEqual(result["nll_sum"], -math.log(0.9) - math.log(0.2))
        self.assertAlmostEqual(result["perplexity"], 1.0 / math.sqrt(0.9 * 0.2))

    def test_mask_avoids_rescoring_overlapping_context(self):
        logits = np.log([[0.1, 0.9], [0.2, 0.8], [0.99, 0.01]])
        result = language_model_metrics(logits, [0, 1, 0], np.array([False, True]))
        self.assertEqual(result["token_count"], 1)
        self.assertAlmostEqual(result["perplexity"], 5.0)

    def test_corpus_result_is_token_weighted(self):
        result = aggregate_language_model_metrics([
            {"nll_sum": math.log(2), "token_count": 1},
            {"nll_sum": 3 * math.log(8), "token_count": 3},
        ])
        self.assertAlmostEqual(result["perplexity"], math.exp((math.log(2) + 3 * math.log(8)) / 4))
        self.assertNotAlmostEqual(result["perplexity"], 5.0)

    def test_overflow_is_explicit_and_json_safe(self):
        result = language_model_metrics([[0, -1000], [0, 0]], [0, 1])
        self.assertEqual(result["mean_nll"], 1000.0)
        self.assertIsNone(result["perplexity"])
        self.assertTrue(result["perplexity_overflow"])
        json.dumps(result, allow_nan=False)

    def test_invalid_lm_inputs_are_rejected(self):
        for logits, ids, mask in [
            ([[0, 0]], [0], None),
            ([[0, 0], [0, 0]], [0, 2], None),
            ([[0, 0], [0, 0]], [0.0, 1.0], None),
            ([[0, 0], [0, 0]], [0, 1], [0]),
            ([[0, 0], [0, 0]], [0, 1], [False]),
            ([[math.inf, 0], [0, 0]], [0, 1], None),
        ]:
            with self.subTest(logits=logits, ids=ids, mask=mask), self.assertRaises(ValueError):
                language_model_metrics(logits, ids, mask)
        with self.assertRaises(ValueError):
            aggregate_language_model_metrics([])


class ProximityMetricTests(unittest.TestCase):
    def test_analytic_probability_distance(self):
        result = logits_proximity(np.log([[0.8, 0.2]]), np.log([[0.2, 0.8]]))
        self.assertAlmostEqual(result["kl_baseline_to_candidate"], 0.6 * math.log(4))
        self.assertAlmostEqual(result["js_divergence"], 0.8 * math.log(1.6) + 0.2 * math.log(0.4))
        self.assertEqual(result["top1_agreement"], 0.0)

    def test_identical_distribution_including_logit_offset(self):
        base = np.array([[1, 2, 5], [-1000, 2, -3000]], dtype=float)
        result = logits_proximity(base, base + 12345)
        self.assertAlmostEqual(result["kl_baseline_to_candidate"], 0.0)
        self.assertAlmostEqual(result["js_divergence"], 0.0)
        self.assertEqual(result["top1_agreement"], 1.0)

    def test_chunking_and_masks_preserve_exact_distribution_comparison(self):
        random = np.random.default_rng(8)
        base, cand = random.normal(size=(31, 257)), random.normal(size=(31, 257))
        a = logits_proximity(base, cand, chunk_size=1)
        b = logits_proximity(base, cand, chunk_size=17)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=12)
        mask = np.arange(31) % 2 == 0
        a, b = logits_proximity(base, cand, mask), logits_proximity(base[mask], cand[mask])
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=12)

    def test_proximity_aggregation_is_token_weighted(self):
        baseline = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
        candidate = baseline[::-1]
        a = aggregate_logits_proximity([
            logits_proximity(np.log(baseline[:1]), np.log(candidate[:1])),
            logits_proximity(np.log(baseline[1:]), np.log(candidate[1:])),
        ])
        b = logits_proximity(np.log(baseline), np.log(candidate))
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=12)

    def test_hidden_zero_vector_semantics_are_explicit(self):
        zero = np.zeros((2, 3))
        result = hidden_proximity(zero, zero)
        self.assertEqual(result["cosine_similarity"], 1)
        self.assertEqual(result["relative_l2"], 0)
        result = hidden_proximity(zero, np.ones((2, 3)))
        self.assertEqual(result["cosine_similarity"], 0)
        self.assertIsNone(result["relative_l2"])
        self.assertFalse(result["relative_l2_defined"])
        json.dumps(result, allow_nan=False)

    def test_hidden_scaling_and_aggregation(self):
        base = np.array([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]])
        result = hidden_proximity(base, 2 * base)
        self.assertAlmostEqual(result["cosine_similarity"], 1)
        self.assertAlmostEqual(result["relative_l2"], 1)
        aggregate = aggregate_hidden_proximity([
            hidden_proximity(base[:1], 2 * base[:1]),
            hidden_proximity(base[1:], 2 * base[1:]),
        ])
        for key in result:
            self.assertAlmostEqual(result[key], aggregate[key])

    def test_misalignment_nonfinite_and_empty_are_rejected(self):
        with self.assertRaises(ValueError):
            logits_proximity([[0, 0]], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            hidden_proximity([[0, math.nan]], [[0, 0]])
        with self.assertRaises(ValueError):
            logits_proximity(np.empty((0, 2)), np.empty((0, 2)))


class RoutingMetricTests(unittest.TestCase):
    @staticmethod
    def event(layer, visit, ids):
        return {"layer": layer, "visit": visit, "expert_ids": np.array(ids)}

    def test_each_candidate_visit_compares_to_physical_layer_baseline(self):
        base = [self.event(3, 0, [[0, 1], [2, 3]])]
        cand = [self.event(3, 0, [[0, 1], [2, 3]]), self.event(3, 1, [[1, 2], [0, 3]])]
        result = routing_proximity(base, cand, num_experts=4)
        self.assertTrue(result["available"])
        self.assertEqual(result["token_count"], 4)
        self.assertAlmostEqual(result["mean_topk_jaccard"], (1.0 + 1.0 / 3.0) / 2.0)
        visits = result["per_layer_visit"]
        self.assertEqual([v["candidate_visit"] for v in visits], [0, 1])
        self.assertAlmostEqual(visits[0]["baseline_load"]["normalized_load_entropy"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_unavailable_routing_is_not_reported_as_zero_distance(self):
        result = routing_proximity([], [])
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean_topk_jaccard"])

    def test_inferred_expert_count_does_not_claim_normalized_load_entropy(self):
        event = self.event(0, 0, [[0], [1]])
        result = routing_proximity([event], [event])
        self.assertEqual(result["num_experts_source"], "inferred_from_observed_ids")
        self.assertIsNone(result["per_layer_visit"][0]["baseline_load"]["normalized_load_entropy"])

    def test_partial_missing_candidate_layer_is_explicit(self):
        a, b = self.event(0, 0, [[0, 1]]), self.event(1, 0, [[0, 1]])
        result = routing_proximity([a, b], [a], num_experts=2)
        self.assertTrue(result["available"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["unmatched_baseline_layers"], [1])

    def test_unsigned_router_indices_are_supported(self):
        event = {"layer": 0, "visit": 0, "expert_ids": np.array([[0, 1]], dtype=np.uint64)}
        result = routing_proximity([event], [event], num_experts=2)
        self.assertEqual(result["mean_topk_jaccard"], 1)

    def test_invalid_routing_is_rejected(self):
        event = self.event(0, 0, [[0, 1], [2, 3]])
        for cand in [
            [self.event(0, 0, [[0, 1]])],
            [self.event(0, 0, [[0, 0], [1, 2]])],
            [event, event],
        ]:
            with self.subTest(candidate=cand), self.assertRaises(ValueError):
                routing_proximity([event], cand, num_experts=4)
        with self.assertRaises(ValueError):
            routing_proximity([event], [event], num_experts=3)


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_resampling_preserves_constant_differences(self):
        result = paired_bootstrap([1, 100, -4], [3, 102, -2], samples=100)
        self.assertEqual(result["delta"], 2)
        self.assertEqual(result["ci_low"], 2)
        self.assertEqual(result["ci_high"], 2)

    def test_seed_and_weighted_document_means(self):
        args = dict(samples=100, seed=91, weights=[1, 3])
        first = paired_bootstrap([1, 1], [1, 5], **args)
        second = paired_bootstrap([1, 1], [1, 5], **args)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta"], 3)
        json.dumps(first, allow_nan=False)

    def test_paired_accuracy_tracks_discordant_examples(self):
        result = paired_accuracy([True, False, True, False], [True, True, False, False], samples=100)
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["degraded"], 1)
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["both_incorrect"], 1)

    def test_single_pair_is_labeled_insufficient(self):
        result = paired_bootstrap([1], [2], samples=10)
        self.assertTrue(result["insufficient_pairs"])
        self.assertEqual(result["ci_low"], result["ci_high"])

    def test_empty_misaligned_and_invalid_samples_rejected(self):
        for base, cand, kwargs in [
            ([], [], {}), ([1], [1, 2], {}), ([1], [1], {"samples": 0}),
            ([1], [1], {"confidence": 1}), ([1], [1], {"weights": [0]}),
            ([math.nan], [1], {}),
        ]:
            with self.subTest(base=base, cand=cand, kwargs=kwargs), self.assertRaises(ValueError):
                paired_bootstrap(base, cand, **kwargs)


if __name__ == "__main__":
    unittest.main()
