# Scaling toward Qwen3.8-Flash-Next

Source review date: **2026-11-23**.

**Current status: architecture reference and future adapter target. Generic looping is not enabled for this hybrid model.** The existing conventional-decoder transformation must not be assumed compatible with it.

## Verified architecture boundary

The official [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) card reports a 125B language-model component with 6B activated parameters, plus 51B n-gram embedding parameters and a 4B multi-token prediction component. The decoder contains 48 layers arranged as twelve cycles of three Gated DeltaNet layers and one Qwen Sparse Attention layer, each followed by MoE. Its routing uses 10 of 512 routed experts plus a shared expert. The gated residual mechanism has four branches. Native context is 262,144 tokens. [Source: model overview](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

The published configuration identifies the wrapper as `Qwen4ExpForConditionalGeneration`, `model_type="qwen4_exp"`; its text configuration is `qwen4_exp_text`. Relevant fields include `hc_count=4`, `hc_lowrank=320`, `hidden_size=2560`, `full_attention_interval=4`, `indexer_compress_ratio=4`, and `indexer_budget=2048`. N-gram injection is configured at `ple_layer_ids=[2]`. These fields are useful adapter checks; they should not replace inspecting the exact installed implementation. [Source: official config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json).

## Why an adapter is required

The conventional transformation takes a hidden tensor and replays a pure layer computation. For this architecture, the adapter must first establish what constitutes a complete layer state: residual branches, attention and recurrent states, convolution history, sparse-attention indexing, position information, and n-gram-related inputs. That is an implementation requirement inferred from the published components, not a claim that every named item mutates on every call.

Treat the internal residual representation as opaque until its layout and read/write gates are verified. Flattening, averaging, or duplicating streams to make an existing wrapper accept the tensor changes the architecture and would need its own experiment.

## Implementation sequence

### 1. Establish the stock reference

Pin the checkpoint revision, engine revision, tokenizer, precision, and kernel configuration. Store short-prefix logits and normal generation outputs. Record which components are loaded or offloaded, rather than estimating memory from active parameters.

### 2. Map the state contract

For every candidate layer, document inputs, outputs, mutable buffers, cache ownership, and logical token positions. Verify whether whole-prefix forwards are independent of preceding calls. Check the n-gram injection path separately; a repeated depth visit must not be interpreted as a newly consumed sequence token.

### 3. Add cache-free experiments

Start with a short context and a small window, testing one change at a time. Cycle-aligned windows may be useful hypotheses because they contain both layer types, but their quality is unknown. Preserve a conventional no-loop baseline and test a no-op `K=1` path before increasing visits.

Add instrumentation for physical layer, visit, token position, routing decisions, hidden norms, and nonfinite values. If the adapter cannot safely replay a selected layer, reject the configuration before running the experiment.

### 4. Validate generation by recomputation

Complete-prefix recomputation provides an initial generation reference. Measure it honestly as such; its latency will not predict an optimized streaming engine. Confirm that disabling and removing the transformation restores baseline behavior without reloading altered weights.

### 5. Design cached decoding

Specify the desired semantics before optimizing: which past states refinement reads, what is committed, and how often. Implement snapshot/restore or a functional equivalent for every affected state. Validate that one generated token advances each intended sequence state exactly once. Compare short cached generation with recomputation and inspect divergence position by position.

This step must not copy Inkling's independent-per-visit cache layout by analogy. Inkling was trained with its native recurrent computation. A frozen Qwen transformation needs a justified state policy and measured consequences.

### 6. Optimize only after quality is measured

Profile expert routing, selected expert loading, activation movement, and cache handling separately. Weight reuse can reduce repeated transfers only when the backend retains the required weights. More layer visits can increase offload traffic, so fewer stored parameters or small active parameter counts are insufficient performance predictions.

Record prefill latency, decode latency, peak VRAM and RAM, context length, output length, quality metrics, and the exact quantization. Measure power only when a reliable sampler is available. Keep throughput and quality claims tied to the tested hardware and backend.

## Promotion criteria

Promote the target from reference to experimental support only after a real-checkpoint run and state-contract tests pass. Promote it further only after held-out quality results and cost measurements are available. Until then, the project can prepare configurations and reject unsafe execution, but cannot claim a working Qwen3.8-Flash-Next-to-looped conversion.
