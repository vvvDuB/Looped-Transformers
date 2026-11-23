# Initial validation record

Date: **2026-11-23**. These checks validate the infrastructure, not a quality improvement in a pretrained language model.

## Environment and outcomes

- Python 3.12.14 on Linux, CPU execution.
- PyTorch 2.15.0+cpu, Transformers 5.24.0, NumPy 2.5.5.
- Full unit/integration suite: **118 tests passed**, including **62 parametrized subtests**, with one third-party Starlette/AnyIO deprecation warning.
- Ruff static checks: passed.
- Wheel build and installation into a clean target directory: passed; installed CLI and packaged dashboard returned a successful health check.
- End-to-end `configs/demo.yaml`: completed; four text documents, 490 predicted byte tokens, four QA items, baseline plus four variants.
- Identity and baseline-restoration checks: passed.
- CLI inspection, completed-run recipe export and recipe generation: passed; exported recipe generated eight tokens with the toy model.
- HTTP tests: real job execution and persistent results, generation, model reuse/unload, serial queue capacity, cooperative cancellation, configuration-path isolation, optional authentication and cross-origin write rejection.

The demo's byte vocabulary, random weights and synthetic examples make its scores unsuitable for claims about language understanding. No toy score is presented as a pretrained-model result.

## Actual architecture tests

The optional tests instantiate the real Hugging Face classes with tiny, random configurations:

- Qwen3 dense;
- Qwen3-MoE;
- Qwen2-MoE;
- OLMoE;
- Mixtral.

They exercise causal prefixes, real sparse router outputs, extra visits in both loop modes, exact identity with one step, unchanged parameters, restoration after exceptions, and rejection of cached/concurrent/unsupported forwards. These tests download no checkpoints and require no network.

## Scientific accounting checks

- Each noninitial document token is scored exactly once across overlapping context windows.
- Every scored target has a causal predecessor.
- Corpus NLL/perplexity aggregate by token count, with no hidden perplexity cap.
- Paired document bootstrap uses token weights for the corpus-mean NLL difference.
- Proximity uses full-distribution KL/JS and correctly aggregated hidden norms.
- Routing compares physical layer/visit/token selections and reports unavailable telemetry explicitly.
- Dataset file/selection hashes persist; conflicting split metadata and exact screening/confirmation overlap fail.
- Errors and cancellations are retained, and the final baseline is restored.

## Not yet validated

- Quality gains on pretrained Qwen or MoE checkpoints.
- GPU execution, memory fit, power consumption, or 4-bit bitsandbytes execution on a particular GPU.
- Production cached decode, CPU offload, multi-GPU execution, long-context QSA/DeltaNet state handling.
- Inkling 2 Mini or Qwen3.8-Flash-Next runtime integration.
- Building/running the optional Docker image in this execution environment.

The included GitHub Actions workflow performs the core and pinned HF architecture suites on pushes and pull requests. Its live run status is separate from the local validation above.
