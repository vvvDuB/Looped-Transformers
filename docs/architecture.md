# Architecture and operation

LoopLab provides one experiment engine through Python, a CLI, and a local HTTP dashboard. Its initial transformation changes execution order and residual updates while retaining the original weights. Its reusable output is a **runtime recipe**, not a newly trained or independently loadable checkpoint.

## Components

| Component | Responsibility |
| --- | --- |
| `config.py` | Strict model and loop settings; inclusive physical layer ranges |
| `experiment.py` | Dataset roles, named variants, experiment loading, relative paths |
| `backends/base.py` | Forward-result structure and token validation |
| `backends/toy.py` | Offline causal Transformer with sparse top-k MoE and immutable random weights |
| `backends/huggingface.py` | Explicit conventional decoder adapters on one device |
| `looping.py` | Damped Euler/naive refinement and reversible decoder hooks |
| `inference.py` | Resident-model lifecycle, serialized leases, complete-prefix generation |
| `datasets.py` | Strict JSONL parsing, source fingerprints, selected-row fingerprints |
| `metrics.py` | Perplexity, distribution/hidden/routing proximity, paired uncertainty |
| `runner.py` | Baseline, identity/restoration checks, paired evaluations, manifests |
| `store.py` | SQLite run index and atomic JSON artifacts |
| `server.py` | Local API, serialized experiment queue, job cancellation |
| `web/index.html` | Dashboard consuming the same API |
| `cli.py` | Commands for running, inspecting, generating, exporting, and serving |

The toy backend exercises real causal attention and expert selection, but its weights are random. Its scores are engineering fixtures. The HF adapter supports explicitly registered conventional layouts: `qwen3`, `qwen3_moe`, `qwen2_moe`, `olmoe`, and `mixtral`. Family registration does not certify every version, checkpoint, quantization, or kernel combination. Unsupported hybrid and native recurrent architectures fail before weights are loaded.

## Forward contract

A backend provides `tokenize`, `decode`, `forward`, and `info`. A forward takes one token sequence, an optional `LoopConfig`, and a trace flag. It returns aligned logits, optional physical-layer hidden states, optional routing events, and execution metadata.

All current forwards use the complete provided context with `use_cache=False`. Generation repeatedly executes that forward on the growing prefix. Output explicitly labels this `full_context_recompute`; it does not represent cached decoding performance. Generation respects the configured context bound and records a stop reason, token counts, time to the first generated token, and per-token elapsed times.

The HF implementation uses a single CPU, CUDA, or MPS device. Quantized loading is an optional CUDA path. Accelerate dispatch/offload hooks, compiled models, mutable attention caches, and unknown decoder layouts are rejected for looping. There is no hidden fallback that silently changes the requested transformation.

## Transformation and observation

Loop ranges use zero-based **inclusive** `start` and `end`. `steps` counts the original visit, so `steps: 1` is the identity control. `mode` chooses `block` or `layer`; `strategy` chooses `euler` or `naive`; `anchor` mixes the refined representation with the normal path.

Hooks are installed for the duration of one forward and removed even if the call raises. Replayed layer calls still invoke normal module hooks. A replay guard prevents recursively applying the intervention to itself. Neither loading a recipe nor enabling a variant updates parameters.

Routing remains dynamic: the backend observes the router on each visit. Some Transformers gates return the chosen experts directly; older registered gates expose logits from which top-k selections are derived. The event's `capture` field states which path was used. A per-layer loop is not a routing-freeze operation.

Hidden traces retain the last visit of each physical layer, with blended output at loop boundaries. They are useful diagnostics, not a complete graph of all intermediate states. Routing events keep `(layer, visit)` explicitly. Proximity on overlapping text windows measures the prediction positions associated with scored next-token targets, avoiding duplicate counting of the context overlap.

## Model lifecycle and concurrency

`ModelManager` holds at most one backend. Identical model configurations reuse it; a different configuration releases the old backend before creating the next. A lock covers a complete experiment or generation request, preventing simultaneous hook mutation of the same model.

The HTTP experiment queue has one worker and up to eight accepted unfinished jobs. Generation is a synchronous endpoint sharing the same model lease: it waits when an experiment holds that lease. Model unloading likewise waits for current work to release it. This is a local research scheduler, not a multi-tenant inference server.

Cancellation is cooperative. Queued jobs are skipped when the worker reaches them; active jobs stop at a progress boundary, normally after the current example. A long document can require multiple windows before reaching that boundary. Cancellation does not interrupt a GPU kernel or promise immediate memory release.

## Durable experiment records

The output root contains `registry.sqlite3` and one directory per 32-character run ID:

| Artifact | Contents |
| --- | --- |
| `config.json` | Resolved experiment configuration |
| `manifest.json` | Environment, model provenance, dataset fingerprints and roles |
| `dataset-0.json`, etc. | Per-item results and detailed proximity/routing summaries |
| `summary.json` | Completed run, compact results, checks, timing scope |
| `error.json` | Failure description and timestamp when a run raises |

JSON artifacts are replaced atomically. SQLite uses transactions and WAL. A run is marked completed only after its final baseline restoration check. The source-byte hash and selected-row hash distinguish the input file from the evaluated subset.

HTTP job IDs describe in-memory scheduling; run IDs identify durable experiment records. These are different IDs. Completed jobs expose their run ID. A cancelled active job reports `cancelled` in the queue; the persistent run also records `cancelled` with an error artifact. Cancellation requested after a terminal job is a no-op. A server restart loses queue history but retains experiment records. Use `looplab recover --root runs` only after interrupted work has stopped; it marks stale running records interrupted rather than resuming computation.

## CLI

```bash
looplab run configs/local-text.yaml
looplab inspect configs/local-text.yaml
looplab sweep configs/local-text.yaml --steps 1,2 --anchors 0,0.5 --modes block,layer
looplab generate configs/local-text.yaml --prompt "Explain a queue." --variant euler --max-new-tokens 32
looplab runs --root runs
looplab show RUN_ID --root runs
looplab export RUN_ID euler recipe.json --root runs
```

`inspect` loads the selected backend and prints its execution schedules; for HF it can download and allocate weights. It is not a lightweight remote metadata lookup. `sweep` expands the existing windows into named variants with a limit on search size. `export` requires a completed run and records the original model plus the chosen runtime intervention. Keep the original checkpoint available when using an exported recipe.

## HTTP API

Start locally with `looplab serve --configs configs --runs runs`. The dashboard is at `http://127.0.0.1:8765/`; FastAPI exposes the generated API schema at `/docs`. Config names are paths relative to the configured root. The API accepts existing configuration files, not arbitrary submitted Python or checkpoint modifications.

| Method and path | Operation |
| --- | --- |
| `GET /api/health` | Version and execution mode |
| `GET /api/configs` | Available configuration filenames |
| `GET /api/models` | Resident model status |
| `POST /api/models/unload` | Release the resident model after its lease ends |
| `GET /api/runs` | Persistent experiment index |
| `GET /api/runs/{run_id}` | Saved config and summary or failure |
| `POST /api/jobs` | Submit `{"config":"local-text.yaml"}`; returns HTTP 202 |
| `GET /api/jobs` | In-memory queue/history |
| `GET /api/jobs/{job_id}` | One job's status and progress |
| `POST /api/jobs/{job_id}/cancel` | Request cooperative cancellation |
| `POST /api/generate` | Synchronous complete-prefix generation |

Generation JSON contains `config`, `prompt`, optional `variant` (default `baseline`), `max_new_tokens` (HTTP limit 256), `temperature`, and `seed`. The API returns HTTP 429 when the experiment queue is full. Invalid or unknown variants are rejected.

If `LOOPLAB_API_KEY` is set, `/api/` calls require `Authorization: Bearer <key>`. The CLI requires that setting before binding beyond localhost. The server rejects writes from a different browser origin. Credentials are deployment settings and do not belong in experiment configurations.

## Extension points

Add a backend only after defining its layer and state semantics, capabilities, provenance, and reference tests. Add metrics with explicit units, masking, aggregation, and failure behavior. Keep benchmark prompting separate from model execution. Native looped and hybrid models need dedicated adapters; the [Inkling reference](inkling-2-mini.md) and [Qwen scaling guide](scaling-qwen.md) identify their current gaps.
