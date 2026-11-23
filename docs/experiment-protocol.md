# Running a reproducible experiment

This guide covers the implemented local workflow. Begin with an offline pipeline check, then validate a real checkpoint, then measure a frozen recipe on held-out data. A completed run with random weights or synthetic smoke data is not evidence of improved model quality.

## 1. Install and run an offline experiment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,server]'
python -m pytest -q
```

Save the following as `configs/local-text.yaml`. All paths resolve relative to that configuration file, so `../data/smoke_text.jsonl` refers to the repository's sample data.

```yaml
name: local-text
model:
  backend: toy
  model_id: toy-moe
  max_context: 128
  width: 32
  num_layers: 6
  num_experts: 4
  top_k: 2
  seed: 42
datasets:
  - path: ../data/smoke_text.jsonl
    kind: text
    split: synthetic
    purpose: smoke
    limit: 3
variants:
  - name: identity
    loop: {start: 2, end: 3, steps: 1, mode: layer, strategy: euler, anchor: 0.0}
  - name: euler
    loop: {start: 2, end: 3, steps: 2, mode: layer, strategy: euler, anchor: 0.0}
output_dir: ../runs
stride: 64
bootstrap_samples: 1000
seed: 42
trace: true
```

```bash
looplab run configs/local-text.yaml
looplab runs --root runs
looplab serve --configs configs --runs runs
```

The runner checks exact `K=1` identity and baseline restoration before measuring, then checks restoration again at the end. Inspect `summary.json`, the manifest, and per-dataset artifacts. Identity results should have equal NLL, identical top-1 predictions, and zero distribution distance within numerical precision. The two-pass variant can change scores without changing weights.

## 2. Validate a real small checkpoint

Install a PyTorch build appropriate for the local device and driver, then install the HF extra:

```bash
python -m pip install -e '.[hf]'
```

For an initial Qwen dense experiment, copy the configuration and replace the model section:

```yaml
model:
  backend: hf
  model_id: Qwen/Qwen3-0.6B-Base
  device: cuda:0
  dtype: bfloat16
  max_context: 256
  local_files_only: false
```

The small `[2,3]` window from the offline example remains a mechanics test, not a recommended quality optimum. Change the window only as an explicit experimental choice. Use `trace: false` for the first execution if limiting instrumentation memory is important, then enable tracing on a short corpus to inspect hidden states and routing. Dense models correctly have no routing measurements.

An initial `inspect` or `run` can download the checkpoint. Set `revision` to an immutable model commit for repeatable later runs. `local_files_only: true` requires the required files already be cached. The loader does not execute remote model code.

On a workstation with **12 GB VRAM and 64 GB system RAM**, start with the small checkpoint, short contexts, and one variant. This backend uses one device and does not combine RAM with VRAM through automatic offloading. Check actual allocation for the selected dtype, context, and tracing settings; the configuration examples are not memory-fit or throughput guarantees.

## 3. Move to a conventional MoE

The registered OLMoE adapter is a useful next engineering target. An optional configuration can use:

```yaml
model:
  backend: hf
  model_id: allenai/OLMoE-1B-7B-0924-Instruct
  device: cuda:0
  dtype: bfloat16
  max_context: 256
  load_in_4bit: true
  local_files_only: false
```

Install `.[hf,quant]` for the bitsandbytes path. Quantized execution is an experiment in its own right: compare the original and looped computation using the same quantization and exact software environment. Support for a model layout does not prove that a particular quantized package/version combination works. The engine rejects dispatch/offload configurations that violate its replay assumptions.

Do not infer storage requirements from a model's active parameter count. If loading fails because of available memory, reduce context/instrumentation or use an appropriate smaller checkpoint/device. CPU offloading and large hybrid MoE support are future integration work, not implicit fallbacks. Inkling 2 Mini and Qwen3.8-Flash-Next are documented references, not accepted generic HF transformations.

## 4. Prepare evaluation data

Input is local UTF-8 JSONL. Optional metadata must be the first record:

```jsonl
{"_meta":{"split":"validation","purpose":"screening","source":"Describe the dataset source and version here"}}
{"id":"document-001","text":"The complete first held-out document."}
{"id":"document-002","text":"The complete second document."}
```

For multiple-choice examples, set `kind: qa` and use:

```jsonl
{"id":"item-001","question":"Which number is even?","choices":["Three","Four","Five"],"answer":1}
```

Answers are zero-based indices. All rows are validated, including those beyond an optional positive `limit`; zero does not mean unlimited. Duplicate IDs, invalid answer indices, repeated choices, unknown fields, duplicate JSON keys, and nonstandard numeric values are rejected. Split and purpose metadata must agree with the experiment configuration.

Use `split: validation, purpose: screening` while choosing a transformation, and `split: test, purpose: confirmation` for a frozen choice. Use `split: synthetic, purpose: smoke` for pipeline fixtures. The runner detects identical task content shared between screening and confirmation datasets in the same experiment, ignoring IDs and incidental metadata. This check does not establish freedom from near duplicates, pretraining contamination, or previous manual tuning.

The local QA protocol prompts with the question, lettered choices, and `Answer:`. It scores the summed log probability of a space-plus-letter continuation; prompt and continuation are tokenized separately. It is a documented custom protocol, not an official ARC/MMLU leaderboard reproduction. It does not apply a model-specific chat template. Oversized questions are rejected rather than silently truncated.

## 5. Run a controlled search

Keep the checkpoint, tokenizer, precision, corpus, and prompts fixed. Start with one window and two visits, and compare one change at a time. The CLI can expand a small search:

```bash
looplab sweep configs/local-text.yaml --steps 1,2 --anchors 0,0.5 --modes block,layer
```

`steps` includes the natural visit. `anchor: 0` uses the refined path; `anchor: 1` returns the normal path while still exercising extra computation. Dynamic routing can change on each visit. No current recipe promises frozen expert selections.

Do not select the largest observed test improvement after trying many configurations. Freeze the chosen recipe using development results, then run independent confirmation. Keep failed and negative experiments in the run registry. The bootstrap intervals do not correct for searching across many variants.

## 6. Read quality, distance, and cost separately

Text evaluation uses overlapping windows with `0 < stride < max_context`. Each document's first token is excluded because there is no preceding token; every later target is scored exactly once. Documents remain separate and do not receive artificial context from neighboring documents. Corpus perplexity is computed from summed NLL divided by total scored tokens, then exponentiated.

The paired NLL interval resamples documents together with their token counts and estimates the change in token-weighted corpus mean NLL. It is not an interval in perplexity units. QA intervals resample paired item correctness. Very small smoke sets cannot provide reliable generalization evidence.

Proximity includes full-vocabulary `KL(baseline || candidate)`, Jensen–Shannon divergence, top-1 agreement, hidden cosine and relative L2 where tracing is available, and expert-set overlap per physical-layer visit. Distances use aligned outputs from the same tokenizer. A smaller distance establishes closeness to the baseline, not better answers.

Recorded forward timings include instrumentation and host copies; first-call warmup uses a short probe. They are diagnostic measurements. For latency work, keep trace settings and input lengths fixed and use repeated measurements. Complete-prefix generation reports its own mode and per-token timings; do not call that optimized decode throughput. There is no performance promise for the 12 GB workstation or for the large target models.

## 7. Preserve the recipe and results

```bash
looplab show RUN_ID --root runs
looplab export RUN_ID euler recipe.json --root runs
```

The recipe preserves model configuration, resolved model information, the loop transformation, and source run. It requires the original weights and LoopLab's runtime. Keep the configuration, manifest, detailed dataset results, and exact code revision together when reporting findings. Distinguish **mechanics verified**, **real-checkpoint execution verified**, and **quality measured**.

See [architecture and API](architecture.md) for job management, [research notes](research-notes.md) for scientific boundaries, and the [Qwen integration path](scaling-qwen.md) before attempting a hybrid model.
