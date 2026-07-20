#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imagewam.benchmark import (  # noqa: E402
    BENCHMARK_NAME,
    CANONICAL_TASKS,
    FORWARD_MODES,
    BenchmarkProtocol,
    benchmark_run_dir,
    build_selection,
    candidate_environment,
    read_json,
    resolve_task_spec,
    run_candidate_process,
    unavailable_result,
    validate_result,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible FLUX.2 sequential versus packed_flex benchmark harness."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Run directory; defaults to .omx/benchmarks/block-sparse-packed-chunk-forward/<UTC>.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe", help="Record environment/task capability without launching model weights."
    )
    probe.add_argument("--task-config", action="append", choices=CANONICAL_TASKS)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate existing candidate JSON files.")
    evaluate.add_argument("result", nargs="+", type=Path)

    run = subparsers.add_parser(
        "run",
        help=(
            "Launch each candidate in a fresh subprocess. The runner must write the path in "
            "IMAGEWAM_BENCHMARK_RESULT_PATH using schema version 1."
        ),
    )
    run.add_argument("--task-config", action="append", choices=CANONICAL_TASKS)
    run.add_argument(
        "runner",
        nargs=argparse.REMAINDER,
        help="Candidate runner command and arguments, written after --.",
    )
    return parser


def _protocol(args: argparse.Namespace) -> BenchmarkProtocol:
    protocol = BenchmarkProtocol(
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        seed=args.seed,
    )
    protocol.validate()
    return protocol


def _output_root(args: argparse.Namespace) -> Path:
    return args.output_root or benchmark_run_dir(REPO_ROOT)


def _task_configs(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.task_config or CANONICAL_TASKS)


def _probe(args: argparse.Namespace) -> int:
    protocol = _protocol(args)
    output_root = _output_root(args)
    results = []
    for task_config in _task_configs(args):
        task_spec = resolve_task_spec(REPO_ROOT, task_config)
        task_name = Path(task_config).stem
        for candidate in FORWARD_MODES:
            result_path = output_root / f"{task_name}-{candidate}.json"
            result = unavailable_result(
                repo_root=REPO_ROOT,
                task_spec=task_spec,
                candidate=candidate,
                protocol=protocol,
                output_path=result_path,
                reason="target CUDA benchmark was not executed by probe mode",
            )
            write_immutable_json(result_path, result)
            results.append(result)
    selection = build_selection(results)
    selection["output_path"] = str(output_root / "selection.json")
    write_immutable_json(output_root / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    output_root = _output_root(args)
    results = [read_json(path) for path in args.result]
    for result in results:
        validate_result(result)
    selection = build_selection(results)
    selection["output_path"] = str(output_root / "selection.json")
    write_immutable_json(output_root / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    import torch

    protocol = _protocol(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The official benchmark requires CUDA. Use `probe` to record unavailable hardware; "
            "packed_flex must remain opt-in."
        )
    runner = list(args.runner)
    if runner and runner[0] == "--":
        runner = runner[1:]
    if not runner:
        raise ValueError("`run` requires a candidate runner command after `--`.")
    output_root = _output_root(args)
    results = []
    for task_config in _task_configs(args):
        task_name = Path(task_config).stem
        for candidate in FORWARD_MODES:
            result_path = output_root / f"{task_name}-{candidate}.json"
            environment = candidate_environment(
                task_config=task_config,
                candidate=candidate,
                protocol=protocol,
                result_path=result_path,
            )
            results.append(
                run_candidate_process(
                    command=runner,
                    cwd=REPO_ROOT,
                    environment=environment,
                    result_path=result_path,
                )
            )
    selection = build_selection(results)
    selection["output_path"] = str(output_root / "selection.json")
    write_immutable_json(output_root / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0 if selection["packed_default_enabled"] else 2


def main() -> int:
    args = _parser().parse_args()
    if args.command == "probe":
        return _probe(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "run":
        return _run(args)
    raise AssertionError(f"Unhandled command for {BENCHMARK_NAME}: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
