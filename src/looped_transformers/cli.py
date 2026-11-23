"""Command-line entry point; all runs use the same Python and HTTP engine."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

from .experiment import Variant, load_experiment
from .inference import ModelManager, generate
from .runner import run_experiment
from .store import RunStore, write_json


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="looped-transformers", description="Frozen MoE transformation and evaluation lab")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep", "inspect", "generate"):
        sub = commands.add_parser(name)
        sub.add_argument("config", help="Experiment JSON/YAML; paths resolve relative to this file")
        if name in {"run", "sweep"}:
            sub.add_argument("--output", help="Override run storage directory")
        if name == "sweep":
            sub.add_argument("--steps", default="1,2,3")
            sub.add_argument("--anchors", default="0,0.5")
            sub.add_argument("--modes", default="block,layer")
        if name == "generate":
            sub.add_argument("--prompt", required=True)
            sub.add_argument("--variant", default="baseline")
            sub.add_argument("--max-new-tokens", type=int, default=32)
            sub.add_argument("--temperature", type=float, default=0.0)
            sub.add_argument("--seed", type=int, default=42)
    runs = commands.add_parser("runs")
    runs.add_argument("--root", default="runs")
    runs.add_argument("--limit", type=int, default=100)
    show = commands.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--root", default="runs")
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("variant")
    export.add_argument("output")
    export.add_argument("--root", default="runs")
    recipe = commands.add_parser("generate-recipe")
    recipe.add_argument("recipe")
    recipe.add_argument("--prompt", required=True)
    recipe.add_argument("--max-new-tokens", type=int, default=32)
    recipe.add_argument("--temperature", type=float, default=0.0)
    recipe.add_argument("--seed", type=int, default=42)
    recover = commands.add_parser("recover")
    recover.add_argument("--root", default="runs")
    serve = commands.add_parser("serve")
    serve.add_argument("--configs", default="configs")
    serve.add_argument("--runs", default="runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "runs":
            _print(RunStore(args.root).list(args.limit))
        elif args.command == "show":
            _print(RunStore(args.root).get(args.run_id))
        elif args.command == "recover":
            _print({"marked_interrupted": RunStore(args.root).recover_interrupted()})
        elif args.command == "generate-recipe":
            from .recipes import load_recipe
            config, loop = load_recipe(args.recipe)
            manager = ModelManager()
            try:
                with manager.lease(config) as backend:
                    _print(generate(backend, args.prompt, loop, args.max_new_tokens, args.temperature, args.seed))
            finally:
                manager.unload()
        elif args.command == "export":
            row = RunStore(args.root).get(args.run_id)
            if row["status"] != "completed":
                raise ValueError("Only a completed run can be exported")
            variant = next((v for v in row["config"]["variants"] if v["name"] == args.variant), None)
            if variant is None:
                raise ValueError(f"Unknown variant: {args.variant}")
            recipe = {"schema_version": 1, "kind": "looped_transformers_runtime_recipe", "model": row["config"]["model"],
                      "resolved_model": row["summary"]["model"], "variant": variant,
                      "source_run": args.run_id, "requires": "looped-transformers==0.1.0",
                      "weights_modified": False, "cache_policy": "full_context_recompute",
                      "note": "Load the original checkpoint and apply this runtime recipe; this is not a standalone weight checkpoint."}
            write_json(Path(args.output), recipe)
            _print({"saved": str(Path(args.output).resolve()), "recipe": recipe})
        elif args.command == "serve":
            if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("LOOPED_TRANSFORMERS_API_KEY"):
                raise ValueError("Set LOOPED_TRANSFORMERS_API_KEY before binding beyond localhost")
            import uvicorn
            from .server import create_app
            uvicorn.run(create_app(args.configs, args.runs), host=args.host, port=args.port)
        else:
            experiment = load_experiment(args.config)
            if getattr(args, "output", None):
                experiment = replace(experiment, output_dir=str(Path(args.output).resolve()))
            if args.command == "sweep":
                steps = sorted({int(s) for s in args.steps.split(",")})
                anchors = sorted({float(s) for s in args.anchors.split(",")})
                modes = list(dict.fromkeys(args.modes.split(",")))
                if len(steps) * len(anchors) * len(modes) * len(experiment.variants) > 128:
                    raise ValueError("Sweep exceeds 128 variants; narrow the search")
                windows = {(v.loop.start, v.loop.end, v.loop.strategy) for v in experiment.variants}
                variants = []
                for start, end, strategy in sorted(windows):
                    for mode in modes:
                        for k in steps:
                            for anchor in anchors:
                                if k == 1 and (anchor != anchors[0] or mode != modes[0]):
                                    continue
                                loop = replace(experiment.variants[0].loop, start=start, end=end,
                                               strategy=strategy, mode=mode, steps=k, anchor=anchor)
                                name = f"w{start}-{end}-{mode}-k{k}-a{anchor:g}-{strategy}".replace(".", "p")
                                variants.append(Variant(name, loop))
                experiment = replace(experiment, variants=tuple(variants))
            if args.command in {"run", "sweep"}:
                _print(run_experiment(experiment, progress=lambda line: print(line, file=sys.stderr)))
            else:
                manager = ModelManager()
                try:
                    with manager.lease(experiment.model) as backend:
                        if args.command == "inspect":
                            info = backend.info()
                            schedules = []
                            for variant in experiment.variants:
                                variant.loop.validate(info["num_layers"])
                                start, end, k = variant.loop.start, variant.loop.end, variant.loop.steps
                                if variant.loop.mode == "block":
                                    schedule = list(range(start)) + list(range(start, end + 1)) * k + list(range(end + 1, info["num_layers"]))
                                else:
                                    schedule = [layer for layer in range(info["num_layers"]) for _ in range(k if start <= layer <= end else 1)]
                                schedules.append({"variant": variant.name, "loop": asdict(variant.loop),
                                                  "physical_layers": info["num_layers"], "layer_evaluations": len(schedule),
                                                  "schedule": schedule})
                            _print({"model": info, "variants": schedules})
                        else:
                            variant = next((v for v in experiment.variants if v.name == args.variant), None)
                            if args.variant != "baseline" and variant is None:
                                raise ValueError(f"Unknown variant {args.variant}")
                            _print(generate(backend, args.prompt, loop=variant.loop if variant else None,
                                            max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed))
                finally:
                    manager.unload()
    except (ValueError, KeyError, OSError, RuntimeError, ImportError, TypeError) as exc:
        print(f"looped-transformers: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
