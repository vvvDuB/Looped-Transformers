# Inkling 2 Mini as a research reference

Source review date: **2026-11-23**. This document describes the upstream architecture. It does not claim that this repository implements an Inkling runtime or has evaluated its checkpoint.

## Two different research questions

Inkling 2 Mini is useful evidence that a **model trained to use recurrence** can make effective use of repeated computation. This project's initial question is whether a **frozen conventional MoE checkpoint** can benefit from an inference-time transformation. Those questions require separate experiments: adding a loop does not import Inkling's trained message gates, training distribution, or capabilities.

The initial LoopLab implementation rejects native looped architectures for generic layer patching. Inkling belongs in the reference registry and in a future dedicated integration, where its native forward pass remains the baseline.

## Verified checkpoint properties

The official checkpoint is [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini). It was released on November 20, 2026 under Apache 2.0. It accepts text and image inputs and emits text. Its decoder stores 24 layers; the middle 12 run three times, producing 48 layer visits. Each MoE layer selects 8 of 128 experts. The card reports 26.4B total parameters, 3.4B active parameters at one pass, and 5.4B effective parameter activations at three passes. These are different accounting quantities, not interchangeable memory estimates. [Source: model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

| Stage | Physical layer indices | Visits at the released setting |
| --- | --- | --- |
| Prelude | `0..5` | Once |
| Recurrent core | `6..17` | Three passes |
| Coda | `18..23` | Once |

In Transformers configuration, the core is the half-open range `[6, 18)`; `loop_layer_index_end=18` is exclusive. The published defaults include hidden size 2304 and `loop_num_passes=3`. Attention is eager-only in the documented implementation. Each visit keeps separate key, value, and convolution state, so the released checkpoint has 48 logical cache layers. Hidden-state and attention outputs likewise count visits rather than physical modules. [Source: Transformers integration](https://huggingface.co/docs/transformers/main/model_doc/inkling2_loop).

## How information crosses passes

The launch article describes attention residuals and a learned message-passing mechanism. Every core pass receives the prelude representation; later passes also receive information from the preceding pass. A learned gate controls that message. Its initialization suppresses the message, and training learns how much to use. Thus, copying an ordinary decoder's output back to its input is not an implementation of this architecture. [Source: architecture and training description](https://thinkingmachines.ai/news/inkling-2-mini/).

The distinction affects observability. A future adapter should label traces with `(physical_layer, loop_pass, logical_visit, token_position)`. Recording only `layer_idx` would conflate different recurrent computations. The same tuple should identify cache entries, attention outputs, routing decisions, and activation comparisons. In vLLM's implementation account, each pass has its own KV cache and receives gated messages from the preceding pass. [Source: vLLM implementation discussion](https://vllm-project.github.io/2026/11/20/inkling2-mini.html).

## Use the research checkpoint for loop-count sweeps

The separate checkpoint [`thinkingmachines/TML-d24-varloop-research`](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) was designed for varying loop count. Its model card reports training with one through five passes in a `1:2:6:2:1` sampling ratio, followed by post-training at three passes. Its later training differs from Mini's, so a direct comparison between the two checkpoints does not isolate looping. Prefer within-checkpoint comparisons at fixed tokenizer, prompts, generation settings, and task data. [Source: variable-loop model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/).

The documented Transformers loading API exposes `text_config={"loop_num_passes": K}` through `AutoModelForMultimodalLM.from_pretrained`. This is an upstream API reference, not a supported LoopLab command. A dedicated adapter must validate that its installed Transformers version provides this class, preserve all other checkpoint settings, and reconstruct caches when `K` changes. Do not change `K` midway through a cached sequence. [Source: variable-pass example](https://huggingface.co/docs/transformers/main/model_doc/inkling2_loop).

## Memory and reproducibility

The official Mini checkpoint occupies approximately 20 GB in mixed NVFP4 and MXFP8; its card recommends 32 GB or more VRAM for other GPU platforms. The documented Transformers path dequantizes these layers on load. The download size therefore does not establish runtime memory usage, and a 12 GB GPU cannot be assumed to hold this checkpoint. Measure the exact backend and precision before selecting a deployment configuration. [Sources: checkpoint hardware guidance](https://huggingface.co/thinkingmachines/Inkling-2-Mini), [Transformers numerics note](https://huggingface.co/docs/transformers/main/model_doc/inkling2_loop).

For a future integration, save the checkpoint revision, full text configuration, precision and dequantization policy, loop count, engine version, chat template, and tokenizer revision. Measure both quality and cost: total generated tokens, prefill latency, decode latency, peak memory, and visits per generated token. A shorter chain of thought alone does not establish either better reasoning or lower computation.

## Dedicated adapter acceptance criteria

1. Match the stock native forward pass at Mini's released three-pass setting.
2. Preserve independent state for every logical visit, including convolution state.
3. Verify cache-based generation against complete-prefix recomputation on short sequences.
4. Confirm that reported hidden states and router traces identify the correct visit.
5. Permit loop-count sweeps on the variable-loop checkpoint with new state for every run.
6. Report unsupported precision, multimodal, and engine combinations explicitly.

Until those conditions are met, the status is **documented research reference; dedicated adapter pending**.
