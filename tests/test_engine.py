"""Offline semantic tests, including real Transformer classes when HF is installed."""
from dataclasses import replace
import importlib.util
import unittest

import numpy as np

from looped_transformers.backends import create_backend
from looped_transformers.backends.huggingface import require_adapter
from looped_transformers.config import LoopConfig, ModelConfig
from looped_transformers.looping import assert_cache_free, refine


class ConfigurationTests(unittest.TestCase):
    def test_invalid_configuration_fails_early(self):
        for data in ({"start": -1, "end": 1}, {"start": 2, "end": 1},
                     {"start": 0, "end": 1, "steps": 0},
                     {"start": 0, "end": 1, "steps": True},
                     {"start": 0, "end": 1, "anchor": float("nan")},
                     {"start": 0, "end": 1, "mode": "arbitrary"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                LoopConfig(**data)
        with self.assertRaises(ValueError):
            LoopConfig(1, 10).validate(6)
        for data in ({"top_k": 8}, {"dtype": "int8"}, {"backend": "remote"},
                     {"device": "cuda"}, {"max_context": 1}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                ModelConfig(**data)

    def test_unsupported_hybrids_and_native_recurrent_models_fail_closed(self):
        for architecture in ("inkling2_loop", "qwen3_5", "qwen3_8_flash_next", "custom"):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                require_adapter(architecture)

    def test_caches_are_rejected(self):
        for key in ("cache", "past_key_value", "past_key_values", "cache_params", "inference_params"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                assert_cache_free({key: object()})
        with self.assertRaises(RuntimeError):
            assert_cache_free({"use_cache": True})
        with self.assertRaises(RuntimeError):
            assert_cache_free({}, require_explicit=True)
        assert_cache_free({"use_cache": False}, require_explicit=True)


class ResidualEquationTests(unittest.TestCase):
    def test_euler_affine_field_has_known_closed_form(self):
        # g(x)=3*x, residual field=2*x. K substeps yield (1+2/K)^K*x.
        x = np.asarray([1.0, -2.0, 0.5])
        config = LoopConfig(0, 0, steps=4)
        calls = []
        def fn(value):
            calls.append(value.copy())
            return 3 * value
        actual = refine(x, 3 * x, fn, config)
        np.testing.assert_allclose(actual, (1 + 2 / 4) ** 4 * x)
        self.assertEqual(len(calls), 3)

    def test_naive_anchor_and_k1_semantics(self):
        x = np.asarray([1.0, 2.0])
        natural = 3 * x
        naive = LoopConfig(0, 0, steps=3, strategy="naive", anchor=0.25)
        np.testing.assert_allclose(refine(x, natural, lambda h: 3 * h, naive), 0.25 * natural + 0.75 * 27 * x)
        def forbidden(_):
            raise AssertionError("K=1 must not replay")
        self.assertIs(refine(x, natural, forbidden, LoopConfig(0, 0, steps=1)), natural)


class ToyMoETests(unittest.TestCase):
    def setUp(self):
        self.backend = create_backend(ModelConfig(max_context=64, width=16, num_layers=4))
        self.ids = self.backend.tokenize("hello MoE")

    def test_determinism_and_frozen_weights(self):
        a = self.backend.forward(self.ids)
        b = create_backend(self.backend.config).forward(self.ids)
        np.testing.assert_array_equal(a.logits, b.logits)
        for array in self.backend._weights():
            self.assertFalse(array.flags.writeable)
        changed = create_backend(replace(self.backend.config, seed=43)).forward(self.ids)
        self.assertFalse(np.array_equal(a.logits, changed.logits))

    def test_byte_tokenization_unicode_and_empty(self):
        text = "Caffè — MoE"
        self.assertEqual(self.backend.decode(self.backend.tokenize(text)), text)
        self.assertEqual(self.backend.tokenize(""), [])
        with self.assertRaises(ValueError):
            self.backend.forward([])

    def test_k1_and_post_intervention_baseline_are_exact(self):
        baseline = self.backend.forward(self.ids)
        fingerprint = self.backend.info()["weights_sha256"]
        for mode in ("layer", "block"):
            k1 = self.backend.forward(self.ids, LoopConfig(1, 2, steps=1, mode=mode))
            np.testing.assert_array_equal(baseline.logits, k1.logits)
            for strategy in ("euler", "naive"):
                loop = LoopConfig(1, 2, steps=3, mode=mode, strategy=strategy, anchor=0.3)
                result = self.backend.forward(self.ids, loop)
                self.assertEqual(result.metadata["layer_visits"], 8)
                self.assertEqual(result.metadata["visits_per_layer"], [1, 3, 3, 1])
                self.assertFalse(np.array_equal(baseline.logits, result.logits))
                np.testing.assert_array_equal(baseline.logits, self.backend.forward(self.ids).logits)
        self.assertEqual(self.backend.info()["weights_sha256"], fingerprint)

    def test_causal_prefix_independence_with_and_without_loops(self):
        extended = self.ids + self.backend.tokenize(" future")
        for loop in (None, LoopConfig(1, 2, steps=3), LoopConfig(1, 2, steps=2, mode="layer")):
            a = self.backend.forward(self.ids, loop)
            b = self.backend.forward(extended, loop)
            np.testing.assert_allclose(a.logits, b.logits[:len(self.ids)], atol=2e-6, rtol=2e-6)

    def test_routes_are_sparse_normalized_and_visit_specific(self):
        output = self.backend.forward(self.ids, LoopConfig(1, 2, steps=3))
        self.assertEqual(len(output.routing), 8)
        for route in output.routing:
            ids = route["expert_ids"]
            self.assertEqual(ids.shape, (len(self.ids), 2))
            self.assertTrue(np.all(ids[:, 0] != ids[:, 1]))
            self.assertTrue(np.all((ids >= 0) & (ids < 4)))
            np.testing.assert_allclose(route["weights"].sum(-1), 1.0, atol=1e-6)
        self.assertEqual([r["visit"] for r in output.routing if r["layer"] == 1], [1, 2, 3])
        self.assertEqual(len(output.hidden_states), 4)
        no_trace = self.backend.forward(self.ids, trace=False)
        self.assertEqual(no_trace.routing, [])
        self.assertEqual(no_trace.hidden_states, {})
        np.testing.assert_array_equal(no_trace.logits, self.backend.forward(self.ids).logits)

    def test_anchor_one_restores_natural_output(self):
        baseline = self.backend.forward(self.ids).logits
        for mode in ("block", "layer"):
            result = self.backend.forward(self.ids, LoopConfig(1, 2, steps=3, mode=mode, anchor=1)).logits
            np.testing.assert_array_equal(result, baseline)

    def test_context_and_token_bounds(self):
        for ids in ([1] * 65, [-1], [258], [1.2], [[1, 2]], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.backend.forward(ids)


_HAS_HF = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


class _TokenizerFixture:
    bos_token_id = 1
    eos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [2 + x % 35 for x in text.encode()]
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


@unittest.skipUnless(_HAS_HF, "optional torch/transformers dependencies are not installed")
class HuggingFaceIntegrationTests(unittest.TestCase):
    """Use actual random-weight HF architectures; no network or checkpoint is needed."""
    def make_backend(self, family="qwen3"):
        import torch
        from transformers import (
            Qwen3Config, Qwen3ForCausalLM, OlmoeConfig, OlmoeForCausalLM,
            Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen2MoeConfig, Qwen2MoeForCausalLM,
            MixtralConfig, MixtralForCausalLM,
        )
        from looped_transformers.backends.huggingface import HuggingFaceBackend
        torch.manual_seed(7)
        common = dict(vocab_size=37, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=4, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64,
                      attention_dropout=0.0, pad_token_id=0, bos_token_id=1, eos_token_id=0)
        if family == "qwen3":
            model = Qwen3ForCausalLM(Qwen3Config(**common, head_dim=8))
        elif family == "olmoe":
            model = OlmoeForCausalLM(OlmoeConfig(**common, num_experts=4, num_experts_per_tok=2))
        elif family == "qwen3_moe":
            model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
                **common, head_dim=8, num_experts=4, num_experts_per_tok=2,
                moe_intermediate_size=16, decoder_sparse_step=1))
        elif family == "qwen2_moe":
            model = Qwen2MoeForCausalLM(Qwen2MoeConfig(
                **common, num_experts=4, num_experts_per_tok=2,
                moe_intermediate_size=16, shared_expert_intermediate_size=32))
        elif family == "mixtral":
            model = MixtralForCausalLM(MixtralConfig(
                **common, num_local_experts=4, num_experts_per_tok=2))
        else:
            raise ValueError(family)
        return HuggingFaceBackend(ModelConfig(backend="hf", model_id="random-test-fixture", max_context=64),
                                  model=model, tokenizer=_TokenizerFixture())

    def test_real_dense_class_loop_restoration_and_causality(self):
        import torch
        backend = self.make_backend()
        ids = [1, 8, 13, 9]
        original_weights = {name: value.clone() for name, value in backend.model.state_dict().items()}
        original_hooks = [(len(layer._forward_pre_hooks), len(layer._forward_hooks)) for layer in backend.layers]
        base = backend.forward(ids)
        for mode in ("block", "layer"):
            one = backend.forward(ids, LoopConfig(1, 2, steps=1, mode=mode))
            np.testing.assert_array_equal(one.logits, base.logits)
            loop = LoopConfig(1, 2, steps=3, mode=mode, anchor=0.25)
            output = backend.forward(ids, loop)
            self.assertEqual(output.metadata["visits_per_layer"], [1, 3, 3, 1])
            longer = backend.forward(ids + [4, 5], loop)
            np.testing.assert_allclose(output.logits, longer.logits[:len(ids)], atol=1e-6, rtol=1e-5)
            np.testing.assert_array_equal(backend.forward(ids).logits, base.logits)
        self.assertEqual(original_hooks, [(len(layer._forward_pre_hooks), len(layer._forward_hooks)) for layer in backend.layers])
        for name, value in backend.model.state_dict().items():
            self.assertTrue(torch.equal(value, original_weights[name]))

    def test_real_sparse_class_routes_and_k1(self):
        backend = self.make_backend("olmoe")
        ids = [1, 8, 13, 9]
        baseline = backend.forward(ids)
        self.assertEqual(len(baseline.routing), 4)
        output = backend.forward(ids, LoopConfig(1, 2, steps=3, mode="layer"))
        self.assertEqual(len(output.routing), 8)
        self.assertEqual(output.routing[0]["expert_ids"].shape, (4, 2))
        np.testing.assert_array_equal(baseline.logits, backend.forward(ids, LoopConfig(1, 2, steps=1)).logits)
        np.testing.assert_array_equal(baseline.logits, backend.forward(ids).logits)

    def test_remaining_moe_adapters_with_real_random_hf_classes(self):
        for family in ("qwen3_moe", "qwen2_moe", "mixtral"):
            with self.subTest(family=family):
                backend = self.make_backend(family)
                ids = [1, 8, 13, 9]
                baseline = backend.forward(ids)
                self.assertEqual(len(baseline.routing), 4)
                for mode in ("block", "layer"):
                    config = LoopConfig(1, 2, steps=2, mode=mode)
                    output = backend.forward(ids, config)
                    self.assertEqual(len(output.routing), 6)
                    self.assertEqual(output.metadata["layer_visits"], 6)
                    self.assertTrue(np.isfinite(output.logits).all())
                np.testing.assert_array_equal(baseline.logits, backend.forward(ids).logits)

    def test_cache_rejection_and_exception_rollback(self):
        import torch
        from looped_transformers.looping import TorchLoopPatch
        backend = self.make_backend()
        ids = [1, 8, 13, 9]
        base = backend.forward(ids).logits
        with self.assertRaises(RuntimeError):
            with TorchLoopPatch(backend.model, backend.layers, LoopConfig(1, 2)):
                backend.model(input_ids=torch.tensor([ids]), use_cache=True)
        self.assertFalse(hasattr(backend.model, "_looped_transformers_active_patch"))
        np.testing.assert_array_equal(base, backend.forward(ids).logits)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with TorchLoopPatch(backend.model, backend.layers, LoopConfig(1, 2)):
                raise RuntimeError("injected failure")
        np.testing.assert_array_equal(base, backend.forward(ids).logits)

    def test_concurrent_calls_and_duplicate_patch_fail(self):
        from looped_transformers.looping import TorchLoopPatch
        backend = self.make_backend()
        backend._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "concurrent"):
                backend.forward([1, 2])
        finally:
            backend._lock.release()
        with TorchLoopPatch(backend.model, backend.layers, LoopConfig(1, 2)):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with TorchLoopPatch(backend.model, backend.layers, LoopConfig(1, 2)):
                    pass


if __name__ == "__main__":
    unittest.main()
