"""Single-device Hugging Face adapters for cache-free frozen-weight experiments.

Supported architecture layouts are explicit. Hybrid/stateful or natively looped
models are rejected before checkpoint weights are downloaded. See docs for the
scope of this adapter; model-family registration is not a quality validation.
"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.metadata
import threading
from typing import Any, Sequence

import numpy as np

from .base import ModelOutput, validate_ids
from ..config import LoopConfig, ModelConfig
from ..looping import TorchLoopPatch, hidden_from_output


# Audited against upstream Transformers decoder layouts, not blind discovery.
# Mixtral's block name changed between Transformers 4.x and 5.x.
ADAPTERS = {
    "qwen3": {"layer_path": "model.layers", "routers": ()},
    "qwen3_moe": {"layer_path": "model.layers", "routers": ("mlp.gate",)},
    "olmoe": {"layer_path": "model.layers", "routers": ("mlp.gate",)},
    "qwen2_moe": {"layer_path": "model.layers", "routers": ("mlp.gate",)},
    "mixtral": {"layer_path": "model.layers", "routers": ("mlp.gate", "block_sparse_moe.gate")},
}


def _resolve(obj: Any, path: str) -> Any:
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def require_adapter(model_type: str) -> dict[str, Any]:
    if model_type not in ADAPTERS:
        raise ValueError(f"No validated cache-free adapter for model_type={model_type!r}. "
                         f"Available: {', '.join(sorted(ADAPTERS))}. "
                         "Inkling/native recurrent models and hybrid DeltaNet/QSA models need a dedicated adapter.")
    return ADAPTERS[model_type]


class HuggingFaceBackend:
    def __init__(self, config: ModelConfig, *, model: Any = None, tokenizer: Any = None):
        if config.backend != "hf":
            raise ValueError("HuggingFaceBackend requires backend='hf'")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the project's hf extra and a suitable PyTorch build: pip install -e '.[hf]'") from exc
        self.torch = torch
        self.config = config
        self._lock = threading.Lock()
        device = torch.device(config.device)
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("HF backend supports a single CPU, CUDA, or MPS device")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if config.load_in_4bit and device.type != "cuda":
            raise ValueError("4-bit bitsandbytes loading requires one CUDA device")
        self.device = device
        self.dtype = getattr(torch, config.dtype)
        self._loaded_tokenizer_revision: str | None = None
        common = {"revision": config.revision, "trust_remote_code": False,
                  "local_files_only": config.local_files_only}
        if model is None:
            hf_config = AutoConfig.from_pretrained(config.model_id, **common)
            self.adapter = require_adapter(hf_config.model_type)
            kwargs: dict[str, Any] = {**common, "config": hf_config,
                                      "torch_dtype": self.dtype, "attn_implementation": "eager"}
            if config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True)
                kwargs["device_map"] = {"": str(device)}
            model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
            if not config.load_in_4bit:
                model.to(device)
            tokenizer_options = dict(common)
            # Resolve model and tokenizer from the same immutable checkpoint revision.
            if getattr(model.config, "_commit_hash", None):
                tokenizer_options["revision"] = model.config._commit_hash
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(config.model_id, **tokenizer_options)
                self._loaded_tokenizer_revision = tokenizer_options.get("revision")
        else:
            self.adapter = require_adapter(model.config.model_type)
            if tokenizer is None:
                raise ValueError("an injected model requires an injected tokenizer")
            if config.load_in_4bit:
                raise ValueError("injected models do not support load_in_4bit")
            model.to(device=device, dtype=self.dtype)
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self._last_logits_dtype: str | None = None
        self._memory_measurement_reset = False
        self.architecture = self.model.config.model_type
        self.layers = _resolve(self.model, self.adapter["layer_path"])
        if not len(self.layers):
            raise ValueError("adapter found no decoder layers")
        self.num_layers = len(self.layers)
        self.num_experts = int(getattr(self.model.config, "num_experts",
                                      getattr(self.model.config, "num_local_experts", 0)))
        self.top_k = int(getattr(self.model.config, "num_experts_per_tok", 0))
        self.vocab_size = int(self.model.config.vocab_size)
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.max_context = min(config.max_context, int(getattr(self.model.config, "max_position_embeddings", config.max_context)))
        self._routers: dict[int, Any] = {}
        for index, layer in enumerate(self.layers):
            for path in self.adapter["routers"]:
                try:
                    self._routers[index] = _resolve(layer, path)
                    break
                except AttributeError:
                    continue
        if self.num_experts and not self._routers:
            raise ValueError("MoE adapter could not locate any routers; refusing silently missing instrumentation")

    def tokenize(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def forward(self, ids: Sequence[int], loop: LoopConfig | None = None,
                trace: bool = True) -> ModelOutput:
        tokens = validate_ids(ids, self.vocab_size, self.max_context)
        if loop is not None:
            loop.validate(self.num_layers)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent forward calls on one HF backend are unsupported; queue requests")
        torch = self.torch
        hidden: dict[str, np.ndarray] = {}
        routing: list[dict[str, Any]] = []
        visits = [0] * self.num_layers
        def to_numpy(value: Any) -> np.ndarray:
            return value.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        try:
            with ExitStack() as stack:
                # Install patch first: observer hooks run after the blended output.
                if loop is not None:
                    stack.enter_context(TorchLoopPatch(self.model, self.layers, loop))
                for index, layer in enumerate(self.layers):
                    def pre(mod: Any, args: tuple, kwargs: dict, index: int = index) -> None:
                        visits[index] += 1
                    handle = layer.register_forward_pre_hook(pre, with_kwargs=True)
                    stack.callback(handle.remove)
                    if trace:
                        def post(mod: Any, args: tuple, output: Any, index: int = index) -> None:
                            hidden[f"layer_{index}"] = to_numpy(hidden_from_output(output)[0])
                        handle = layer.register_forward_hook(post)
                        stack.callback(handle.remove)
                if trace:
                    for index, router in self._routers.items():
                        def capture(mod: Any, args: tuple, output: Any, index: int = index) -> None:
                            if isinstance(output, (tuple, list)) and len(output) == 3:
                                logits, weights, expert_ids = output
                                if expert_ids.dtype not in (torch.int32, torch.int64):
                                    raise TypeError("router tuple's third output must be expert indices")
                                source = "actual_selected_experts"
                            elif isinstance(output, torch.Tensor):
                                logits = output
                                weights, expert_ids = torch.topk(torch.softmax(logits.float(), dim=-1), self.top_k, dim=-1)
                                # Legacy linear gates expose logits; these registered
                                # model families choose top-k softmax probabilities.
                                source = "derived_topk_from_router_logits"
                            else:
                                raise TypeError("unsupported router output; adapter requires an update")
                            selected = expert_ids.detach().cpu().numpy().reshape(-1, self.top_k).copy()
                            if len(selected) != len(tokens):
                                raise ValueError("router trace token count does not match the input sequence")
                            routing.append({"layer": index, "visit": visits[index],
                                            "expert_ids": selected,
                                            "weights": to_numpy(weights).reshape(-1, self.top_k),
                                            "num_experts": self.num_experts, "capture": source})
                        handle = router.register_forward_hook(capture)
                        stack.callback(handle.remove)
                input_ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.inference_mode():
                    result = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
                self._last_logits_dtype = str(result.logits.dtype).removeprefix("torch.")
                logits = to_numpy(result.logits[0])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if logits.shape != (len(tokens), self.vocab_size):
                raise ValueError(f"unexpected logits shape {logits.shape}")
            if not np.isfinite(logits).all():
                raise FloatingPointError("model produced non-finite logits")
            return ModelOutput(logits=logits, hidden_states=hidden, routing=routing,
                               metadata={"layer_visits": sum(visits), "visits_per_layer": visits,
                                         "cache_policy": "full_context_recompute",
                                         "hidden_capture": "last_visit; blended output at loop boundaries"})
        finally:
            self._lock.release()

    def reset_peak_memory_stats(self) -> None:
        """Start a CUDA peak measurement after warmup without clearing allocations.

        Stats are per device in this process, not isolated to this model. The
        caller must hold the model-manager lease around reset and measurement.
        """
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
            self._memory_measurement_reset = True

    def info(self) -> dict[str, Any]:
        parameter_dtypes: dict[str, int] = {}
        for parameter in self.model.parameters():
            dtype = str(parameter.dtype).removeprefix("torch.")
            parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + parameter.numel()
        tokenizer_options = getattr(self.tokenizer, "init_kwargs", {})
        tokenizer_revision = tokenizer_options.get("_commit_hash") or self._loaded_tokenizer_revision
        data: dict[str, Any] = {
            "backend": "hf", "model_id": self.config.model_id, "architecture": self.architecture,
            "requested_revision": self.config.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "num_layers": self.num_layers, "num_experts": self.num_experts, "top_k": self.top_k,
            "vocab_size": self.vocab_size, "max_context": self.max_context,
            "parameter_count": self.model.num_parameters(),
            "parameter_tensor_elements": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device), "dtype": str(self.model.dtype).removeprefix("torch."),
            "requested_dtype": self.config.dtype, "parameter_dtypes": parameter_dtypes,
            "logits_dtype": self._last_logits_dtype,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": getattr(self.tokenizer, "name_or_path", None),
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_policy": "raw text; add_special_tokens=False; no chat template",
            "quantization": ("bitsandbytes_nf4" if self.config.load_in_4bit else
                             str(self.model.quantization_method) if getattr(self.model, "quantization_method", None) else None),
            "torch_version": self.torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "loop_support": "cache-free single-device experimental adapter",
            "routing_capture": "actual router tuple or top-k derived from legacy router logits" if self._routers else "dense model",
            "cache_policy": "full_context_recompute", "trust_remote_code": False,
            "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
        }
        if self.device.type == "cuda":
            data["gpu_name"] = self.torch.cuda.get_device_name(self.device)
            data["cuda_peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated(self.device)
            data["cuda_peak_reserved_bytes"] = self.torch.cuda.max_memory_reserved(self.device)
            data["cuda_allocated_bytes"] = self.torch.cuda.memory_allocated(self.device)
            data["cuda_measurement_scope"] = (
                "device-wide in this process since explicit reset" if self._memory_measurement_reset
                else "device-wide in this process since process start or most recent external reset")
        return data
