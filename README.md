# Looped Transformers Lab

An experiment infrastructure for studying **frozen Mixture-of-Experts models and recurrent depth**: change the execution of a model, run inference, measure perplexity and proximity to its baseline, inspect expert routing, and retain reproducible benchmark results.

The initial research question is whether a conventional pretrained MoE benefits from reusing some of its layers **without updating its weights**. The code is a working research foundation, with explicit boundaries between tested mechanics and unverified model-quality hypotheses.

## What works

- Reversible **block** and **layer** loops; damped Euler substeps, natural-path anchoring, and naive repetition as a control.
- A deterministic, genuinely sparse **tiny causal MoE** that runs offline on CPU. Its random weights are an engineering fixture, not a useful language model.
- Hugging Face adapters for **Qwen3, Qwen3-MoE, Qwen2-MoE, OLMoE, and Mixtral**. Single-device loading, optional CUDA NF4, frozen weights, explicit unsupported-architecture rejection.
- Autoregressive generation that recomputes the **complete prefix** at each token. A model manager keeps one resident checkpoint and serializes experiments/inference.
- Token-weighted **NLL/perplexity**, full-vocabulary **KL and Jensen–Shannon divergence**, top-1 agreement, hidden-state cosine and relative L2, expert-set Jaccard and expert-load entropy.
- Local text and multiple-choice benchmarks; overlapping context windows score every target once; paired bootstrap intervals and explicit screening/confirmation splits.
- Strict YAML/JSON experiments, parameter sweeps, SQLite run registry, atomic JSON artifacts, SHA-256 dataset provenance, checkpoint revision/environment capture, and portable transformation recipes.
- A minimal local dashboard and HTTP API with a bounded serial job queue, cancellation, model unload, and generation.
- Tests and GitHub Actions for the offline pipeline and real small, random-initialized Hugging Face architectures.

**Current limits:** no training, no production KV/SSM cache implementation, no multi-device offload, no vision processing, and no automatic claim that a transformed model is better. Inkling 2 Mini and Qwen3.8-Flash-Next are documented reference/target architectures; their dedicated adapters are not implemented.

## Start in a minute

Python 3.11 or newer is required. No model download or GPU is needed for the demo.

```bash
git clone https://github.com/vvvDuB/Looped-Transformers.git
cd Looped-Transformers
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,server]'

looped-transformers run configs/demo.yaml
looped-transformers serve --configs configs --runs runs
```

Open **http://127.0.0.1:8765**. Select a configuration, start a run, then select its result to compare baseline and loop variants. `demo.yaml` evaluates four synthetic text documents and four QA examples. Its numbers validate the pipeline; they are not evidence of improved reasoning.

The dashboard also exposes generation. Output from the untrained toy model is expected to be nonsense. A real checkpoint is needed for meaningful text.

## Real checkpoints

Install a PyTorch build suitable for your GPU and driver using the [official installation instructions](https://pytorch.org/get-started/locally/), then install the optional backend:

```bash
python -m pip install -e '.[hf,server,dev]'
looped-transformers inspect configs/qwen3-06b.yaml
looped-transformers run configs/qwen3-06b.yaml
looped-transformers generate configs/qwen3-06b.yaml \
  --variant euler-block --prompt 'Explain why the sky appears blue.' \
  --max-new-tokens 64
```

The preset uses `Qwen/Qwen3-0.6B-Base` on CUDA with BF16. Set `device: cpu` and `dtype: float32` for CPU experiments. Set `revision` to an immutable checkpoint commit for confirmation experiments; the resolved revision is captured when available.

For a conventional MoE experiment:

```bash
python -m pip install -e '.[hf,quant,server,dev]'
looped-transformers run configs/olmoe-4bit.yaml
```

This preset uses `allenai/OLMoE-1B-7B-0924-Instruct`, one CUDA device, and bitsandbytes NF4. Its actual peak memory must be measured on the target machine. GPU kernels, context, tracing, and quantization overhead all matter; active parameter count is not total weight memory. CPU offload and automatic multi-device dispatch are intentionally rejected by the current loop engine.

The adapters are tested with **PyTorch 2.15.0 CPU / Transformers 5.24.0** on tiny random models from all five registered families. This verifies execution, routing, identity and restoration, not the quality or GPU memory of the full pretrained checkpoints. Four-bit loading is available but has not been validated on a GPU in this initial implementation.

## Describe a transformation

Each experiment always includes an unmodified baseline. Add named variants:

```yaml
variants:
  - name: identity
    loop: {start: 6, end: 9, steps: 1, mode: layer}
  - name: euler-layer
    loop: {start: 6, end: 9, steps: 2, mode: layer, anchor: 0.5}
  - name: naive-control
    loop: {start: 6, end: 9, steps: 2, mode: block, strategy: naive}
```

Layer bounds are **zero-based and inclusive**. `steps` is the total number of evaluations, including the first. With a four-layer window and `steps: 3`, there are eight additional layer evaluations. This count is a structural cost measure, not a FLOP, energy, or latency estimate.

For a chosen layer or block `g`, the Euler variant computes:

```python
natural = g(h0)
x = h0
for j in range(steps):
    y = natural if j == 0 else g(x)
    x = x + (y - x) / steps
result = anchor * natural + (1 - anchor) * x
```

`strategy: naive` instead sets `x = g(x)` at each step. `mode: block` repeats the whole window; `mode: layer` refines each selected layer before moving to the next. Expert routing remains **dynamic on every visit**. Repeating a layer does not freeze its router.

`steps: 1` returns the exact original computation. Each experiment checks this and verifies that the baseline is unchanged after transformations. Unexpected caches, unsupported architectures, malformed input and non-finite outputs fail explicitly.

## Run a parameter sweep

```bash
looped-transformers sweep configs/qwen3-06b.yaml \
  --steps 1,2,3 --anchors 0,0.5 --modes block,layer
```

The sweep expands the windows/strategies in the supplied config and caps the search at 128 variants. Search on validation data, then move the chosen recipe to independent confirmation data. Bootstrap intervals after searching many configurations do not by themselves correct selection bias.

## Use your own benchmarks

Text JSONL:

```json
{"_meta":{"split":"validation","purpose":"screening","source":"Name and immutable revision of the source corpus"}}
{"id":"doc-001","text":"A complete held-out passage goes here."}
```

Multiple-choice JSONL:

```json
{"_meta":{"split":"validation","purpose":"screening","source":"Your benchmark and revision"}}
{"id":"qa-001","question":"What is 2 + 3?","choices":["4","5","6","7"],"answer":1}
```

Answers are zero-based indices. `_meta` is optional; when present, its split/purpose must agree with the experiment. IDs are optional and generated from source line numbers; supplied IDs must be unique. The loader rejects malformed records, duplicate JSON keys, invalid answers, and non-finite metadata.

```yaml
datasets:
  - path: ../data/heldout-text.jsonl
    kind: text
    split: validation
    purpose: screening
  - path: ../data/heldout-qa.jsonl
    kind: qa
    split: test
    purpose: confirmation
```

All paths resolve relative to the configuration file. Each evaluated file and ordered selection gets a SHA-256 fingerprint. Exact task-content overlap between screening and confirmation files in the same experiment is rejected, even if IDs or metadata differ. This does not detect paraphrases, training contamination, or overlap across separate historical runs.

Perplexity is `exp(total NLL / total predicted tokens)`, computed with a causal shift and overlapping windows. The first token of each document is unscored; every subsequent target is scored once. The stride changes available conditioning and must remain fixed for comparisons.

QA uses an explicit completion prompt and answer-letter log probabilities. It is a custom controlled comparison, **not an implementation of a model's official leaderboard evaluation template**. Questions exceeding the context window fail instead of being silently truncated.

See [the experiment protocol](docs/experiment-protocol.md) for dataset selection, confidence intervals and stopping criteria.

## Inspect, retain, and replay

```bash
looped-transformers inspect configs/demo.yaml
looped-transformers runs --root runs
looped-transformers show RUN_ID --root runs
looped-transformers export RUN_ID euler-block recipes/toy-euler.json --root runs
looped-transformers generate-recipe recipes/toy-euler.json --prompt 'A short test' --max-new-tokens 12
```

A recipe saves the checkpoint identity and runtime transformation, with an immutable resolved revision when available. It does not duplicate tied weights or pretend to be a standalone Hugging Face checkpoint. Loading a recipe validates its schema, backend and cache policy before using the original weights.

Each run directory contains:

- `config.json`: complete resolved experiment.
- `manifest.json`: model/tokenizer identity, dataset hashes, environment, source revision and measurement scope.
- `dataset-N.json`: per-item scores, aggregate metrics, hidden-state proximity and routing diagnostics.
- `summary.json`: compact comparisons, structural compute cost and invariant checks.
- `error.json`: failure/cancellation details when applicable.

`runs/registry.sqlite3` indexes completed, failed, cancelled and in-progress runs. Interrupted processes remain identifiable; after confirming no experiment is still active, `looped-transformers recover --root runs` marks stale running entries interrupted. Generated results, checkpoints and secrets are excluded from Git.

## Local service

```bash
looped-transformers serve --configs configs --runs runs --host 127.0.0.1 --port 8765
```

The service accepts only existing configuration files beneath the chosen directory. It runs one experiment at a time, holds one model, queues up to eight jobs and supports cooperative cancellation at example boundaries. The main endpoints are:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/configs` | Available experiments |
| POST | `/api/jobs` | Queue `{"config":"demo.yaml"}` |
| GET | `/api/jobs/{id}` | Status/progress |
| POST | `/api/jobs/{id}/cancel` | Request cancellation |
| GET | `/api/runs` and `/api/runs/{id}` | Persistent results |
| POST | `/api/generate` | Generate with a configuration and variant |
| GET | `/api/models` | Resident model |
| POST | `/api/models/unload` | Release resident model |

Interactive API documentation is at `/docs`. Set `LOOPED_TRANSFORMERS_API_KEY` to require bearer authentication; it is mandatory when binding beyond loopback. Enter the key in the dashboard's API access field. No external UI dependencies or hosted telemetry are used.

An optional CPU Docker setup is included:

```bash
cp .env.example .env
# Edit .env and choose your own local API key.
docker compose up --build
```

The container includes the offline backend and service. GPU/PyTorch/bitsandbytes are deliberately installed separately in the native HF workflow.

## Inkling 2 Mini as a reference

[Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) is a useful **native looped MoE reference**. Its 24 stored layers execute 48 visits: the middle layers `6..17` run three times. It includes learned inter-pass message gates, and its cache is indexed by logical visit. Those properties were trained into the model; the frozen wrapper here does not reproduce them.

For controlled native-loop-count experiments, the relevant separate checkpoint is [`thinkingmachines/TML-d24-varloop-research`](https://huggingface.co/thinkingmachines/TML-d24-varloop-research). See [the dedicated Inkling notes](docs/inkling-2-mini.md) for architecture, quantization/memory caveats, and the adapter requirements.

The [Qwen3.8-Flash-Next scaling plan](docs/scaling-qwen.md) covers its MoE, DeltaNet, sparse attention and multiple residual streams. The current generic adapter refuses that architecture until its state semantics have a dedicated implementation.

## Research and development

```bash
python -m pytest -q
ruff check .
```

The core/service tests need no network. Optional HF tests instantiate tiny real Transformers classes without downloading checkpoints. CI runs the dependency-light suite and a pinned CPU HF suite, plus the complete demo experiment.

Documentation:

- [Architecture and extension points](docs/architecture.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [Training-free looping, perplexity and proximity](docs/research-notes.md)
- [Inkling 2 Mini](docs/inkling-2-mini.md)
- [Scaling toward Qwen3.8-Flash-Next](docs/scaling-qwen.md)
- [Validation record](docs/validation.md)

Inspired by [Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1). This repository is an independent implementation of the documented damped-Euler experiment, not a reproduction of every method, cache strategy or reported result in that paper.
