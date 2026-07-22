#!/usr/bin/env python3
"""Launch reproducible FLUX.2 chunkwise RoPE ablations on a GPU server.

The launcher deliberately starts one distributed training job at a time. Each
run gets an isolated output directory, a full command manifest, a streamed log,
and machine-readable final validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROPE_SCHEMES = (
    "current",
    "chronological_image",
    "temporal_action",
    "chronological_full",
    "no_chunk_time",
)
LOCKED_OVERRIDE_KEYS = frozenset(
    {
        "model.chunkwise_causal.rope_scheme",
        "model.chunkwise_causal.forward_mode",
        "seed",
        "max_steps",
        "eval_every",
        "eval_num_samples",
        "save_every",
        "output_dir",
        "wandb.group",
        "wandb.name",
        "wandb.mode",
    }
)
METRIC_DIRECTIONS = {
    "val_loss": "lower",
    "action_l1": "lower",
    "action_l2": "lower",
    "infer_psnr": "higher",
    "infer_ssim": "higher",
}

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EVAL_PATTERN = re.compile(
    rf"\[eval\] step=(?P<step>\d+) samples=(?P<num_samples>\d+) "
    rf"val_loss=(?P<val_loss>{_FLOAT}) infer_psnr=(?P<infer_psnr>{_FLOAT}) "
    rf"infer_ssim=(?P<infer_ssim>{_FLOAT})"
    rf"(?: action_l2=(?P<action_l2>{_FLOAT}))?"
    rf"(?: action_l1=(?P<action_l1>{_FLOAT}))?"
)
TRAIN_PATTERN = re.compile(
    rf"\[train\] epoch=(?P<epoch>\d+) step=(?P<step>\d+)/\d+ "
    rf"loss=(?P<loss>{_FLOAT})"
)


@dataclass(frozen=True)
class RunSpec:
    scheme: str
    seed: int
    output_dir: str
    log_path: str
    command: list[str]


def _csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def parse_schemes(raw: str) -> list[str]:
    schemes = _csv_values(raw)
    unknown = sorted(set(schemes).difference(ROPE_SCHEMES))
    if unknown:
        raise ValueError(f"Unknown RoPE schemes: {unknown}; expected a subset of {list(ROPE_SCHEMES)}")
    if not schemes:
        raise ValueError("At least one RoPE scheme is required.")
    return schemes


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value) for value in _csv_values(raw)]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def validate_extra_overrides(overrides: Iterable[str]) -> None:
    collisions = []
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in LOCKED_OVERRIDE_KEYS:
            collisions.append(key)
    if collisions:
        raise ValueError(
            "Use dedicated launcher flags instead of overriding locked keys: "
            f"{sorted(set(collisions))}"
        )


def parse_metrics(lines: Iterable[str]) -> dict[str, float | int]:
    final_eval: dict[str, float | int] = {}
    final_train: dict[str, float | int] = {}
    for line in lines:
        eval_match = EVAL_PATTERN.search(line)
        if eval_match:
            final_eval = {
                key: (int(value) if key in {"step", "num_samples"} else float(value))
                for key, value in eval_match.groupdict().items()
                if value is not None
            }
        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            final_train = {
                "train_epoch": int(train_match.group("epoch")),
                "train_step": int(train_match.group("step")),
                "train_loss": float(train_match.group("loss")),
            }
    return {**final_train, **final_eval}


def parse_metrics_jsonl(path: Path) -> dict[str, float | int]:
    final_train: dict[str, float | int] = {}
    final_eval: dict[str, float | int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("phase") == "train":
            final_train = {
                "train_step": int(record["step"]),
                "train_loss": float(record["train/loss"]),
            }
        elif record.get("phase") == "eval":
            final_eval = {
                "step": int(record["step"]),
                "num_samples": int(record["eval/num_samples"]),
                "val_loss": float(record["eval/val_loss"]),
                "infer_psnr": float(record["eval/psnr_rd"]),
                "infer_ssim": float(record["eval/ssim_rd"]),
            }
            for source, target in (
                ("eval/action_l2", "action_l2"),
                ("eval/action_l1", "action_l1"),
            ):
                if source in record:
                    final_eval[target] = float(record[source])
    return {**final_train, **final_eval}


def build_command(args: argparse.Namespace, scheme: str, seed: int, output_dir: Path) -> list[str]:
    name = f"rope-{scheme}-seed-{seed}"
    overrides = [
        f"model.chunkwise_causal.rope_scheme={scheme}",
        f"model.chunkwise_causal.forward_mode={args.forward_mode}",
        f"seed={seed}",
        f"max_steps={args.max_steps}",
        f"eval_every={args.eval_every}",
        f"eval_num_samples={args.eval_num_samples}",
        f"save_every={args.save_every}",
        f"output_dir={output_dir}",
        f"wandb.group={args.wandb_group}",
        f"wandb.name={name}",
        f"wandb.mode={args.wandb_mode}",
    ]
    return [
        "bash",
        "scripts/flux2/run_train_flux2_klein_imagewam.sh",
        *args.override,
        *overrides,
    ]


def _git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * float(quantile)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_interval(
    differences: list[float],
    *,
    samples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("Paired bootstrap requires at least one difference.")
    generator = random.Random(seed)
    count = len(differences)
    means = [
        statistics.fmean(differences[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _run_one(
    spec: RunSpec,
    *,
    repo_root: Path,
    base_env: dict[str, str],
    dry_run: bool,
    require_eval_metrics: bool,
) -> dict[str, object]:
    output_dir = Path(spec.output_dir)
    log_path = Path(spec.log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(spec.command)
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"\n[rope-ablation] scheme={spec.scheme} seed={spec.seed}")
    print(f"[rope-ablation] output={output_dir}")
    print(f"[rope-ablation] command={command_text}")

    if dry_run:
        result = {
            "scheme": spec.scheme,
            "seed": spec.seed,
            "status": "dry_run",
            "returncode": None,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "command": spec.command,
        }
        _write_json(output_dir / "run.json", result)
        return result

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        run_env = base_env.copy()
        run_env["IMAGEWAM_METRICS_JSONL"] = str(metrics_path)
        process = subprocess.Popen(
            spec.command,
            cwd=repo_root,
            env=run_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        returncode = process.wait()

    if metrics_path.exists():
        metrics = parse_metrics_jsonl(metrics_path)
    else:
        metrics = parse_metrics(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
    missing_required_metrics = require_eval_metrics and "val_loss" not in metrics
    status = "completed" if returncode == 0 and not missing_required_metrics else "failed"
    effective_returncode = 1 if returncode == 0 and missing_required_metrics else returncode
    result = {
        "scheme": spec.scheme,
        "seed": spec.seed,
        "status": status,
        "returncode": effective_returncode,
        "process_returncode": returncode,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "command": spec.command,
    }
    if missing_required_metrics:
        result["validation_error"] = "Training exited successfully but emitted no eval/val_loss metric."
    _write_json(output_dir / "run.json", result)
    return result


def write_summaries(output_root: Path, results: list[dict[str, object]]) -> None:
    _write_json(output_root / "summary.json", results)
    metric_names = sorted(
        {
            key
            for result in results
            for key in dict(result.get("metrics", {}))
        }
    )
    columns = ["scheme", "seed", "status", "returncode", *metric_names]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scheme": result["scheme"],
                    "seed": result["seed"],
                    "status": result["status"],
                    "returncode": result["returncode"],
                    **dict(result.get("metrics", {})),
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for scheme in ROPE_SCHEMES:
        scheme_results = [
            result for result in results
            if result["scheme"] == scheme and result["status"] == "completed"
        ]
        if not scheme_results:
            continue
        row: dict[str, object] = {"scheme": scheme, "runs": len(scheme_results)}
        for metric_name in metric_names:
            values = [
                float(dict(result.get("metrics", {}))[metric_name])
                for result in scheme_results
                if metric_name in dict(result.get("metrics", {}))
            ]
            if values:
                row[f"{metric_name}_mean"] = statistics.fmean(values)
                row[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate_columns = sorted({key for row in aggregate_rows for key in row}, key=lambda key: (key not in {"scheme", "runs"}, key))
    with (output_root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    completed_by_scheme_seed = {
        (str(result["scheme"]), int(result["seed"])): result
        for result in results
        if result["status"] == "completed"
    }
    comparison_rows: list[dict[str, object]] = []
    baseline_seeds = {
        seed for scheme, seed in completed_by_scheme_seed if scheme == "current"
    }
    for scheme in ROPE_SCHEMES:
        if scheme == "current":
            continue
        paired_seeds = sorted(
            baseline_seeds.intersection(
                seed for candidate_scheme, seed in completed_by_scheme_seed
                if candidate_scheme == scheme
            )
        )
        for metric_name, direction in METRIC_DIRECTIONS.items():
            differences = []
            for seed in paired_seeds:
                baseline_metrics = dict(completed_by_scheme_seed[("current", seed)].get("metrics", {}))
                candidate_metrics = dict(completed_by_scheme_seed[(scheme, seed)].get("metrics", {}))
                if metric_name in baseline_metrics and metric_name in candidate_metrics:
                    differences.append(
                        float(candidate_metrics[metric_name]) - float(baseline_metrics[metric_name])
                    )
            if not differences:
                continue
            if len(differences) < 3:
                ci_low = None
                ci_high = None
                decision = "insufficient"
            else:
                ci_low, ci_high = paired_bootstrap_interval(differences)
                if direction == "lower":
                    decision = "better" if ci_high < 0 else "worse" if ci_low > 0 else "inconclusive"
                else:
                    decision = "better" if ci_low > 0 else "worse" if ci_high < 0 else "inconclusive"
            comparison_rows.append(
                {
                    "baseline": "current",
                    "candidate": scheme,
                    "metric": metric_name,
                    "direction": direction,
                    "paired_seeds": len(differences),
                    "mean_difference": statistics.fmean(differences),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "decision": decision,
                }
            )
    comparison_columns = (
        "baseline",
        "candidate",
        "metric",
        "direction",
        "paired_seeds",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "decision",
    )
    with (output_root / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns)
        writer.writeheader()
        writer.writerows(comparison_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", default=",".join(ROPE_SCHEMES))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--task-type", choices=("libero", "robotwin"), default="robotwin")
    parser.add_argument("--flux2-variant", choices=("4b", "9b"), default="4b")
    parser.add_argument("--gpus-per-run", type=int, default=int(os.environ.get("GPU_PER_NODE", "8")))
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES value, for example 0,1,2,3")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-num-samples", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--forward-mode", choices=("sequential", "packed_flex"), default="packed_flex")
    parser.add_argument("--wandb-group", default="flux2-rope-ablation")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    schemes = parse_schemes(args.schemes)
    seeds = parse_seeds(args.seeds)
    validate_extra_overrides(args.override)
    if args.gpus_per_run < 1:
        raise ValueError("--gpus-per-run must be positive.")
    if args.devices is not None and len(_csv_values(args.devices)) != args.gpus_per_run:
        raise ValueError("--devices count must equal --gpus-per-run.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = (args.output_root or repo_root / "runs" / "rope_ablation" / timestamp).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env.update(
        {
            "TASK_TYPE": args.task_type,
            "FLUX2_VARIANT": args.flux2_variant,
            "GPU_PER_NODE": str(args.gpus_per_run),
        }
    )
    if args.devices is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.devices

    specs: list[RunSpec] = []
    for scheme in schemes:
        for seed in seeds:
            run_dir = output_root / scheme / f"seed_{seed}"
            specs.append(
                RunSpec(
                    scheme=scheme,
                    seed=seed,
                    output_dir=str(run_dir),
                    log_path=str(run_dir / "launcher.log"),
                    command=build_command(args, scheme, seed, run_dir),
                )
            )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status": _git_value(repo_root, "status", "--short"),
        "task_type": args.task_type,
        "flux2_variant": args.flux2_variant,
        "gpus_per_run": args.gpus_per_run,
        "devices": args.devices,
        "runs": [asdict(spec) for spec in specs],
    }
    _write_json(output_root / "manifest.json", manifest)

    results: list[dict[str, object]] = []
    for spec in specs:
        result_path = Path(spec.output_dir) / "run.json"
        if args.skip_completed and result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                print(f"[rope-ablation] skipping completed {spec.scheme} seed={spec.seed}")
                results.append(existing)
                continue
        base_env["RUN_ID"] = f"rope_{spec.scheme}_seed_{spec.seed}"
        result = _run_one(
            spec,
            repo_root=repo_root,
            base_env=base_env,
            dry_run=args.dry_run,
            require_eval_metrics=args.eval_every > 0,
        )
        results.append(result)
        write_summaries(output_root, results)
        if result["status"] == "failed" and not args.keep_going:
            print(f"[rope-ablation] failed; see {spec.log_path}", file=sys.stderr)
            return int(result["returncode"] or 1)

    write_summaries(output_root, results)
    print(f"\n[rope-ablation] summary={output_root / 'aggregate.csv'}")
    return 0 if all(result["status"] != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
