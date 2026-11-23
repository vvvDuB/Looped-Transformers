# Research protocol: frozen MoE to looped MoE

Source review date: **2026-11-23**.

This repository is an experiment platform. A successful run establishes that the implementation executed; it does not establish that the transformation improves a pretrained model. Offline synthetic MoE examples validate mechanics and numerical accounting. They cannot establish language-model quality.

## Research basis and implementation scope

[*Training-Free Looped Transformers*, arXiv:2605.23872](https://arxiv.org/abs/2605.23872), studies recurrence added to frozen checkpoints. The authors report task-specific improvements as well as serious failures from naive repetition. The paper distinguishes repetition of a whole layer window from repetition of each individual layer and discusses damped updates, MoE routing, and cache handling. [Paper HTML](https://arxiv.org/html/2605.23872v1).

Looped Transformers Lab is an independent implementation with explicitly named transformations, not a claim to reproduce every experiment in that paper. Its first supported execution mode evaluates the entire supplied token prefix without incremental caches. Autoregressive generation can recompute that complete prefix at every step. This is a correctness-oriented reference, and its cost must not be reported as that of an optimized cached decoder.

## Specify the transformation mathematically

Let `g` be the normal computation of one selected layer, or a selected contiguous window. Let `h0` be its input. A damped Euler experiment with `K` visits computes:

```text
x = h0
for i in 0..K-1:
    x = x + (g(x) - x) / K
```

An optional natural-path anchor combines the resulting state with `g(h0)`:

```text
result = beta * g(h0) + (1 - beta) * x
```

The `g(h0)` result can be reused from the first visit. Record whether the implementation reuses it; otherwise compute accounting changes. The naive comparison applies `x = g(x)` repeatedly. It is a distinct experiment, not an equivalent algebraic form.

Block mode iterates the composed window. Layer mode completes the visits of each layer before advancing. Count the original visit as part of `K`; `K=1` must recover the original forward pass. With `L` layers and a window of width `W`, an implementation without additional cache-stash passes executes `L + W*(K-1)` decoder layer visits. This is a visit count, not a FLOP estimate: attention, shared experts, and selected experts can have different costs.

## Routing is an independent experimental variable

Calling the same MoE layer several times does not freeze its router. The hidden state changes, so a router can select different experts. Expose dynamic routing and frozen routing as separate policies. If an adapter cannot implement a requested policy faithfully, reject that configuration.

For frozen routing, define what is frozen: expert identities, mixture weights, or the entire router output. These choices differ. Record the first-pass selection for each token, then verify what subsequent visits actually used. Measure per-visit expert-set overlap, expert usage distribution, and gate changes. A metric labeled “routing stability” must state its denominator and its comparison pair.

The paper discusses routing stabilization as part of its MoE rationale. This project does not infer frozen routing merely from selecting layer mode, nor use that label without an explicit mechanism and tests. [Source: sections 2.2 and 3.3](https://arxiv.org/html/2605.23872v1).

## Perplexity and proximity answer different questions

Perplexity measures how much probability a model assigns to observed next tokens:

```text
mean_nll = sum(negative_log_probability_of_each_scored_token) / scored_token_count
perplexity = exp(mean_nll)
```

Aggregate negative log-likelihood and token counts across samples before exponentiating. Averaging sample perplexities gives a different result. The first token needs an explicit context policy; padding, context-only positions, and masked labels must not enter the denominator. For long documents, use overlapping windows and score each target token once. Keep the tokenizer, document boundaries, stride, and context length fixed when comparing transformations. [Source: Hugging Face perplexity guide](https://huggingface.co/docs/transformers/en/perplexity).

Proximity measures how a transformation changes outputs relative to the baseline. Useful comparisons on the same aligned tokens include:

| Metric | Interpretation | Limitation |
| --- | --- | --- |
| `KL(p_baseline || p_modified)` | Change in the predicted token distribution | A smaller value establishes closeness, not improvement |
| Jensen–Shannon divergence | Symmetric distribution difference | State whether values use natural logarithms |
| Top-1 agreement | Fraction of positions retaining the predicted token | Hides shifts elsewhere in the distribution |
| Logit RMSE | Difference between raw output vectors | Sensitive to additive shifts that leave softmax unchanged |
| Hidden-state cosine similarity | Change in an aligned representation | Valid only for comparable dimensions and layer visits |
| Expert-set overlap | Stability of selected experts | Requires actual router traces, not guessed selections |

Do not compare raw token-level KL or perplexity between unrelated tokenizers as though their vocabularies were aligned. Inkling is an architectural reference; a Qwen-to-Inkling raw-logit distance is not a meaningful baseline comparison.

## Evaluation discipline

1. **Mechanical control:** verify baseline, `K=1`, patch removal, unchanged parameters, and deterministic repeated evaluation before a quality claim.
2. **Screening:** use a designated development split to choose windows, `K`, damping, and routing policy. Log unsuccessful configurations too.
3. **Confirmation:** freeze the chosen recipe and evaluate on held-out data. Do not optimize using the reported test set.
4. **Task comparison:** use paired examples and identical prompts. Report absolute baseline score, modified score, difference, and uncertainty.
5. **Cost comparison:** measure warm runs, synchronization policy, context length, output length, and actual wall time. Separate prefill from decode when the backend exposes both.
6. **Generalization:** try multiple task families and context lengths. A gain on one multiple-choice dataset does not establish general intelligence.

A real benchmark integration should preserve the task's prompting, answer-scoring, few-shot count, dataset version, and split. EleutherAI's harness defines separate interfaces for conditional log-likelihood, rolling log-likelihood, and generation. These should remain distinct in backend capability checks. [Source: evaluation-harness model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md).

## Cache policy is part of the model definition

The training-free paper's decode procedure temporarily restores cache lengths during refinement and then commits a canonical state once per original layer and token. Repeating cache-mutating calls directly can manufacture extra positions. That policy is different from native Inkling, which maintains state for each logical layer visit. Neither policy is universally correct for all models. [Sources: paper appendix C](https://arxiv.org/html/2605.23872v1), [Inkling Transformers notes](https://huggingface.co/docs/transformers/main/model_doc/inkling2_loop).

A future cached adapter must identify every mutable state, preserve token positions, and validate cached logits against a reference. Hybrid models may add recurrent, convolution, and sparse-index state. `use_cache=False` is a request to the backend, not proof that an arbitrary model is stateless.

## Interpreting results

Keep three labels distinct in reports: **mechanics verified**, **checkpoint execution verified**, and **quality measured**. A toy test supports the first. A real model loading and producing outputs supports the second. A matched, documented benchmark supports the third. Report unsupported architectures as unsupported; a successful import is not a model compatibility result.
