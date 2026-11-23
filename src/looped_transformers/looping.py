"""Residual looping and reversible PyTorch decoder hooks.

These operations implement frozen-weight interventions, not an Inkling converter.
The patch accepts explicit decoder layers from an architecture adapter. Every
call must be in evaluation mode with use_cache=False; no state is committed.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from .config import LoopConfig


def refine(h0: Any, natural: Any, fn: Callable[[Any], Any], config: LoopConfig) -> Any:
    """K evaluations, reusing the natural first evaluation, with optional anchor.

    For Euler, K substeps of size 1/K integrate the residual field g(x)-x.
    This is a hypothesis to test, not a guarantee of improved quality.
    K=1 returns the original object without arithmetic or extra computation.
    """
    if config.steps == 1:
        return natural
    x = h0
    for index in range(config.steps):
        y = natural if index == 0 else fn(x)
        x = y if config.strategy == "naive" else x + (y - x) / config.steps
    if config.anchor == 0:
        return x
    if config.anchor == 1:
        return natural
    return config.anchor * natural + (1 - config.anchor) * x


def hidden_from_output(output: Any) -> Any:
    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise TypeError("decoder layer must return a tensor or a tuple/list with a tensor first")


def replace_hidden(output: Any, hidden: Any) -> Any:
    if hasattr(output, "shape"):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError("unsupported decoder output")


def call_arguments(module: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Bind positional cache parameters as well as keyword arguments."""
    bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    for key, parameter in inspect.signature(module.forward).parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            values.update(values.pop(key, {}))
    return values


def assert_cache_free(values: dict[str, Any], *, require_explicit: bool = False) -> None:
    for name in ("past_key_value", "past_key_values", "cache", "cache_params",
                 "inference_params", "layer_past"):
        if values.get(name) is not None:
            raise RuntimeError(f"looping received {name}; cached execution is not supported")
    if values.get("use_cache") is True or (require_explicit and values.get("use_cache") is not False):
        raise RuntimeError("looping requires explicit use_cache=False")


def input_hidden(args: tuple, kwargs: dict) -> Any:
    if args and hasattr(args[0], "shape"):
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise TypeError("decoder inputs do not contain hidden_states")


class TorchLoopPatch:
    """Temporary cache-free hooks; single model / single forward at a time.

    The backend owns a lock over the entire call. Replays use module(*args), so
    normal module hooks run; a local guard prevents recursive loop application.
    No module is replaced and no parameter is modified.
    """

    def __init__(self, model: Any, layers: Any, config: LoopConfig):
        config.validate(len(layers))
        self.model, self.layers, self.config = model, layers, config
        self.handles: list[Any] = []
        self._calls: dict[int, tuple[tuple, dict]] = {}
        self._replay = False
        self._active = False

    def __enter__(self) -> "TorchLoopPatch":
        if self._active or getattr(self.model, "_looped_transformers_active_patch", None) is not None:
            raise RuntimeError("a loop patch is already active on this model")
        if self.model.training:
            raise RuntimeError("looping requires model.eval()")
        if hasattr(self.model, "_orig_mod"):
            raise RuntimeError("torch.compile models require a dedicated loop adapter")
        if any(getattr(m, "_hf_hook", None) is not None for m in self.model.modules()):
            raise RuntimeError("Accelerate dispatch/offload hooks are unsupported by the loop adapter")
        self.model._looped_transformers_active_patch = self
        self._active = True
        try:
            def guard(mod: Any, args: tuple, kwargs: dict) -> None:
                assert_cache_free(call_arguments(mod, args, kwargs), require_explicit=True)
            self.handles.append(self.model.register_forward_pre_hook(guard, with_kwargs=True))
            if self.config.steps == 1:
                return self
            for index in range(self.config.start, self.config.end + 1):
                def before(mod: Any, args: tuple, kwargs: dict, index: int = index) -> None:
                    assert_cache_free(call_arguments(mod, args, kwargs))
                    if not self._replay:
                        self._calls[index] = (args, dict(kwargs))
                self.handles.append(self.layers[index].register_forward_pre_hook(before, with_kwargs=True))
            if self.config.mode == "layer":
                for index in range(self.config.start, self.config.end + 1):
                    def after(mod: Any, args: tuple, kwargs: dict, output: Any,
                              index: int = index) -> Any:
                        if self._replay:
                            return output
                        natural = hidden_from_output(output)
                        x = refine(input_hidden(args, kwargs), natural,
                                   lambda h: self._reenter(index, h), self.config)
                        return replace_hidden(output, x)
                    self.handles.append(self.layers[index].register_forward_hook(after, with_kwargs=True))
            else:
                def after_window(mod: Any, args: tuple, kwargs: dict, output: Any) -> Any:
                    if self._replay:
                        return output
                    natural = hidden_from_output(output)
                    h0 = input_hidden(*self._calls[self.config.start])
                    def window(h: Any) -> Any:
                        for index in range(self.config.start, self.config.end + 1):
                            h = self._reenter(index, h)
                        return h
                    return replace_hidden(output, refine(h0, natural, window, self.config))
                self.handles.append(self.layers[self.config.end].register_forward_hook(
                    after_window, with_kwargs=True))
            return self
        except BaseException:
            self.remove()
            raise

    def _reenter(self, index: int, hidden: Any) -> Any:
        args, kwargs = self._calls[index]
        new_args, new_kwargs = list(args), dict(kwargs)
        if new_args and hasattr(new_args[0], "shape"):
            new_args[0] = hidden
        else:
            new_kwargs["hidden_states"] = hidden
        old_replay = self._replay
        self._replay = True
        try:
            return hidden_from_output(self.layers[index](*new_args, **new_kwargs))
        finally:
            self._replay = old_replay

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self._calls.clear()
        self._replay = False
        if getattr(self.model, "_looped_transformers_active_patch", None) is self:
            delattr(self.model, "_looped_transformers_active_patch")
        self._active = False

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()
