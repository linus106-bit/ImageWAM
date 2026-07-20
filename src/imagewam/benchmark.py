from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_NAME = "block-sparse-packed-chunk-forward"
CANONICAL_TASKS = (
    "configs/task/robotwin_flux2_klein_4b_base_imagewam.yaml",
    "configs/task/robotwin_flux2_klein_9b_base_imagewam.yaml",
)
FORWARD_MODES = ("sequential", "packed_flex")
CALL_COUNT_KEYS = ("outer_model", "mot", "flex_attention", "backward")


@dataclass(frozen=True)
class BenchmarkProtocol:
    warmup_steps: int = 10
    measured_steps: int = 30
    seed: int = 0

    def validate(self) -> None:
        if self.warmup_steps != 10:
            raise ValueError("Official benchmark runs require exactly 10 warm-up steps.")
        if self.measured_steps < 30:
            raise ValueError("Official benchmark runs require at least 30 measured steps.")
        if self.seed < 0:
            raise ValueError("Benchmark seed must be non-negative.")


def utc_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def benchmark_run_dir(repo_root: Path, run_id: str | None = None) -> Path:
    return (
        Path(repo_root)
        / ".omx"
        / "benchmarks"
        / BENCHMARK_NAME
        / (run_id or utc_run_id())
    )


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def git_metadata(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def runtime_metadata() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "gpu": device_name,
        "driver": _nvidia_driver_version() if cuda_available else None,
    }


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


def resolve_task_spec(repo_root: Path, task_config: str) -> dict[str, Any]:
    from omegaconf import OmegaConf

    root = Path(repo_root)
    task_path = root / task_config
    if not task_path.is_file():
        raise FileNotFoundError(f"Task config does not exist: {task_path}")
    task = OmegaConf.to_container(OmegaConf.load(task_path), resolve=False)
    if not isinstance(task, dict):
        raise ValueError(f"Task config must contain a mapping: {task_path}")
    model_name = _resolve_default_name(task.get("defaults", ()), "override /model")
    model_path = root / "configs" / "model" / f"{model_name}.yaml"
    model = OmegaConf.to_container(OmegaConf.load(model_path), resolve=False)
    if not isinstance(model, dict):
        raise ValueError(f"Model config must contain a mapping: {model_path}")
    chunkwise = model.get("chunkwise_causal", {})
    if not isinstance(chunkwise, dict):
        raise ValueError(f"Missing model.chunkwise_causal mapping in {model_path}")
    return {
        "task_config": task_config,
        "model_config": str(model_path.relative_to(root)),
        "batch_size": int(task["batch_size"]),
        "gradient_accumulation_steps": int(task["gradient_accumulation_steps"]),
        "precision": str(task["mixed_precision"]),
        "num_chunks": int(chunkwise["num_chunks"]),
        "actions_per_chunk": int(chunkwise["actions_per_chunk"]),
        "total_action_horizon": int(chunkwise["num_chunks"])
        * int(chunkwise["actions_per_chunk"]),
        "forward_mode_default": str(chunkwise.get("forward_mode", "sequential")),
        "sparse_packing": str(chunkwise.get("sparse_packing", "batch_padded")),
        "sparse_block_size": int(chunkwise.get("sparse_block_size", 128)),
        "sparse_alignment": str(chunkwise.get("sparse_alignment", "none")),
        "layout_schema_version": int(chunkwise.get("packed_layout_schema_version", 1)),
    }


def _resolve_default_name(defaults: Any, key: str) -> str:
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        raise ValueError("Hydra defaults must be a sequence.")
    for entry in defaults:
        if isinstance(entry, Mapping) and key in entry:
            return str(entry[key])
    raise ValueError(f"Hydra defaults do not define {key!r}.")


def unavailable_result(
    *,
    repo_root: Path,
    task_spec: Mapping[str, Any],
    candidate: str,
    protocol: BenchmarkProtocol,
    output_path: Path,
    reason: str,
) -> dict[str, Any]:
    _validate_candidate(candidate)
    protocol.validate()
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "unavailable",
        "reason": reason,
        "candidate": candidate,
        "source": git_metadata(repo_root),
        "task": dict(task_spec),
        "runtime": runtime_metadata(),
        "distributed": {"world_size": 1, "rank": 0, "backend": "none", "zero_stage": 0},
        "protocol": asdict(protocol),
        "correctness": {
            "passed": False,
            "finite_loss": None,
            "finite_gradients": None,
            "oom": None,
        },
        "metrics": _empty_metrics(),
        "output_path": str(output_path),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "compile_warmup_seconds": None,
        "forward_seconds": {"median": None, "p95": None},
        "backward_seconds": {"median": None, "p95": None},
        "total_step_seconds": {"median": None, "p95": None},
        "logical_trajectories_per_second": {"microbatch": None, "optimizer_step": None},
        "peak_memory_bytes": {"allocated": None, "reserved": None},
        "packed": {
            "tokens": None,
            "theoretical_dense_token_pairs": None,
            "visited_sparse_blocks": None,
            "block_sparsity": None,
        },
        "call_counts": {key: None for key in CALL_COUNT_KEYS},
        "per_rank": [],
        "max_rank": {},
    }


def validate_result(result: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "benchmark",
        "status",
        "candidate",
        "source",
        "task",
        "runtime",
        "distributed",
        "protocol",
        "correctness",
        "metrics",
        "output_path",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"Benchmark result is missing fields: {missing}.")
    if int(result["schema_version"]) != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark schema: {result['schema_version']!r}.")
    if result["benchmark"] != BENCHMARK_NAME:
        raise ValueError(f"Unexpected benchmark name: {result['benchmark']!r}.")
    _validate_candidate(str(result["candidate"]))
    if result["status"] not in {"completed", "unavailable", "failed"}:
        raise ValueError(f"Unsupported benchmark status: {result['status']!r}.")
    _require_fields(result["source"], ("commit", "dirty"), "source")
    _require_fields(
        result["task"],
        (
            "task_config",
            "model_config",
            "batch_size",
            "gradient_accumulation_steps",
            "precision",
            "num_chunks",
            "actions_per_chunk",
            "total_action_horizon",
            "forward_mode_default",
            "sparse_packing",
            "sparse_block_size",
            "sparse_alignment",
            "layout_schema_version",
        ),
        "task",
    )
    _require_fields(
        result["runtime"],
        ("torch", "cuda_available", "cuda_runtime", "gpu", "driver"),
        "runtime",
    )
    _require_fields(
        result["distributed"],
        ("world_size", "rank", "backend", "zero_stage"),
        "distributed",
    )
    protocol = result["protocol"]
    _require_fields(protocol, ("warmup_steps", "measured_steps", "seed"), "protocol")
    _require_fields(
        result["correctness"],
        ("passed", "finite_loss", "finite_gradients", "oom"),
        "correctness",
    )
    BenchmarkProtocol(
        warmup_steps=int(protocol["warmup_steps"]),
        measured_steps=int(protocol["measured_steps"]),
        seed=int(protocol["seed"]),
    ).validate()
    metrics = result["metrics"]
    _require_fields(
        metrics,
        (
            "compile_warmup_seconds",
            "forward_seconds",
            "backward_seconds",
            "total_step_seconds",
            "logical_trajectories_per_second",
            "peak_memory_bytes",
            "packed",
            "call_counts",
            "per_rank",
            "max_rank",
        ),
        "metrics",
    )
    _require_fields(
        metrics["packed"],
        (
            "tokens",
            "theoretical_dense_token_pairs",
            "visited_sparse_blocks",
            "block_sparsity",
        ),
        "metrics.packed",
    )
    call_counts = metrics.get("call_counts", {})
    missing_counts = sorted(set(CALL_COUNT_KEYS) - call_counts.keys())
    if missing_counts:
        raise ValueError(f"Benchmark result is missing call counts: {missing_counts}.")
    if result["status"] == "completed":
        _validate_completed_metrics(result)


def _require_fields(value: Any, fields: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Benchmark {label} must be a mapping.")
    missing = sorted(set(fields) - value.keys())
    if missing:
        raise ValueError(f"Benchmark {label} is missing fields: {missing}.")


def _validate_candidate(candidate: str) -> None:
    if candidate not in FORWARD_MODES:
        raise ValueError(f"Unsupported benchmark candidate: {candidate!r}.")


def _validate_completed_metrics(result: Mapping[str, Any]) -> None:
    if result["source"]["dirty"] is not False:
        raise ValueError("Completed benchmark results require a clean git source.")
    if result["runtime"]["cuda_available"] is not True:
        raise ValueError("Completed benchmark results require CUDA runtime evidence.")
    correctness = result["correctness"]
    if correctness.get("oom") is not False:
        raise ValueError("Completed benchmark results must report oom=false.")
    for key in ("finite_loss", "finite_gradients"):
        if correctness.get(key) is not True:
            raise ValueError(f"Completed benchmark results must report {key}=true.")
    metrics = result["metrics"]
    positive_values = (
        metrics["total_step_seconds"]["median"],
        metrics["total_step_seconds"]["p95"],
        metrics["logical_trajectories_per_second"]["microbatch"],
        metrics["logical_trajectories_per_second"]["optimizer_step"],
    )
    if any(not _is_finite_positive(value) for value in positive_values):
        raise ValueError("Completed benchmark results require finite positive timing/throughput.")
    if not _is_finite_positive(metrics["peak_memory_bytes"]["allocated"]):
        raise ValueError("Completed benchmark results require finite positive GPU memory.")
    if not _is_finite_nonnegative(metrics["compile_warmup_seconds"]):
        raise ValueError("Completed benchmark results require finite compile/warm-up time.")
    packed = metrics["packed"]
    for key in ("tokens", "theoretical_dense_token_pairs", "visited_sparse_blocks"):
        if not _is_finite_nonnegative(packed[key]):
            raise ValueError(f"Completed benchmark results require finite packed.{key}.")
    if not _is_unit_interval(packed["block_sparsity"]):
        raise ValueError("Completed benchmark block_sparsity must be in [0,1].")
    _validate_call_counts(result)
    _validate_rank_aggregates(result)


def _validate_call_counts(result: Mapping[str, Any]) -> None:
    counts = result["metrics"]["call_counts"]
    if any(not isinstance(counts[key], int) or counts[key] < 0 for key in CALL_COUNT_KEYS):
        raise ValueError("Completed benchmark call counts must be non-negative integers.")
    measured_steps = int(result["protocol"]["measured_steps"])
    chunks = int(result["task"]["num_chunks"])
    multiplier = 1 if result["candidate"] == "packed_flex" else chunks
    expected = measured_steps * multiplier
    for key in ("outer_model", "mot", "backward"):
        if counts[key] != expected:
            raise ValueError(
                f"Completed {result['candidate']} result requires {key}={expected}, "
                f"got {counts[key]}."
            )
    if result["candidate"] == "packed_flex" and counts["flex_attention"] <= 0:
        raise ValueError("Completed packed_flex results require FlexAttention calls.")
    if result["candidate"] == "sequential" and counts["flex_attention"] != 0:
        raise ValueError("Sequential benchmark results must not report FlexAttention calls.")


def _validate_rank_aggregates(result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    per_rank = metrics["per_rank"]
    world_size = int(result["distributed"]["world_size"])
    measured_steps = int(result["protocol"]["measured_steps"])
    if not isinstance(per_rank, list) or len(per_rank) != world_size:
        raise ValueError("Completed benchmark results require one raw record per rank.")
    ranks = [int(rank["rank"]) for rank in per_rank]
    if sorted(ranks) != list(range(world_size)):
        raise ValueError("Per-rank benchmark records must have unique contiguous rank ids.")
    for rank in per_rank:
        for phase in ("forward_seconds", "backward_seconds", "total_step_seconds"):
            if len(rank[phase]) != measured_steps:
                raise ValueError(
                    f"Rank {rank['rank']} {phase} must contain {measured_steps} samples."
                )
    recomputed = summarize_rank_measurements(
        per_rank=per_rank,
        local_batch=int(result["task"]["batch_size"]),
        world_size=world_size,
        gradient_accumulation=int(result["task"]["gradient_accumulation_steps"]),
    )
    for phase in ("forward_seconds", "backward_seconds", "total_step_seconds"):
        for statistic in ("median", "p95"):
            _require_close(
                metrics[phase][statistic],
                recomputed[phase][statistic],
                f"metrics.{phase}.{statistic}",
            )
    for key in ("microbatch", "optimizer_step"):
        _require_close(
            metrics["logical_trajectories_per_second"][key],
            recomputed["logical_trajectories_per_second"][key],
            f"metrics.logical_trajectories_per_second.{key}",
        )
    for key in ("allocated", "reserved"):
        _require_close(
            metrics["peak_memory_bytes"][key],
            recomputed["peak_memory_bytes"][key],
            f"metrics.peak_memory_bytes.{key}",
        )


def _require_close(actual: Any, expected: Any, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"{label} does not match raw per-rank evidence.")


def _is_finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def _is_finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _is_unit_interval(value: Any) -> bool:
    return _is_finite_nonnegative(value) and float(value) <= 1.0


def summarize_rank_measurements(
    *,
    per_rank: Sequence[Mapping[str, Any]],
    local_batch: int,
    world_size: int,
    gradient_accumulation: int,
) -> dict[str, Any]:
    if len(per_rank) != int(world_size) or not per_rank:
        raise ValueError("Per-rank measurements must contain exactly one entry per rank.")
    summaries: dict[str, dict[str, float]] = {}
    for phase in ("forward_seconds", "backward_seconds", "total_step_seconds"):
        rank_medians = []
        rank_p95s = []
        for rank in per_rank:
            values = [float(value) for value in rank[phase]]
            if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"Invalid {phase} measurements.")
            rank_medians.append(statistics.median(values))
            rank_p95s.append(_percentile(values, 0.95))
        summaries[phase] = {"median": max(rank_medians), "p95": max(rank_p95s)}
    max_total = summaries["total_step_seconds"]["median"]
    optimizer_seconds = max(
        float(rank["optimizer_step_seconds"]) for rank in per_rank
    )
    if max_total <= 0.0 or optimizer_seconds <= 0.0:
        raise ValueError("Throughput denominators must be positive.")
    peak_allocated = max(int(rank["peak_allocated_bytes"]) for rank in per_rank)
    peak_reserved = max(int(rank["peak_reserved_bytes"]) for rank in per_rank)
    return {
        **summaries,
        "logical_trajectories_per_second": {
            "microbatch": int(local_batch) * int(world_size) / max_total,
            "optimizer_step": int(local_batch)
            * int(world_size)
            * int(gradient_accumulation)
            / optimizer_seconds,
        },
        "peak_memory_bytes": {"allocated": peak_allocated, "reserved": peak_reserved},
        "per_rank": [dict(rank) for rank in per_rank],
        "max_rank": {
            "total_step_seconds": max_total,
            "optimizer_step_seconds": optimizer_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
    }


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_pair(
    sequential: Mapping[str, Any], packed: Mapping[str, Any]
) -> dict[str, Any]:
    validate_result(sequential)
    validate_result(packed)
    if sequential["candidate"] != "sequential" or packed["candidate"] != "packed_flex":
        raise ValueError("Gate evaluation requires sequential then packed_flex results.")
    if sequential["task"]["task_config"] != packed["task"]["task_config"]:
        raise ValueError("Gate candidates must use the same task config.")
    if _comparison_fingerprint(sequential) != _comparison_fingerprint(packed):
        raise ValueError(
            "Gate candidates must share identical source, task, runtime, distributed, and "
            "protocol fingerprints."
        )
    if sequential["status"] != "completed" or packed["status"] != "completed":
        return {
            "passed": False,
            "reason": "performance evidence unavailable",
            "checks": {},
        }
    sequential_metrics = sequential["metrics"]
    packed_metrics = packed["metrics"]
    sequential_median = float(sequential_metrics["total_step_seconds"]["median"])
    packed_median = float(packed_metrics["total_step_seconds"]["median"])
    sequential_p95 = float(sequential_metrics["total_step_seconds"]["p95"])
    packed_p95 = float(packed_metrics["total_step_seconds"]["p95"])
    sequential_throughput = float(
        sequential_metrics["logical_trajectories_per_second"]["optimizer_step"]
    )
    packed_throughput = float(
        packed_metrics["logical_trajectories_per_second"]["optimizer_step"]
    )
    sequential_memory = float(sequential_metrics["peak_memory_bytes"]["allocated"])
    packed_memory = float(packed_metrics["peak_memory_bytes"]["allocated"])
    latency_gain = packed_median <= sequential_median * 0.90
    throughput_gain = packed_throughput >= sequential_throughput * 1.10
    checks = {
        "median_or_throughput": latency_gain or throughput_gain,
        "p95_within_5_percent": packed_p95 <= sequential_p95 * 1.05,
        "memory_within_10_percent": packed_memory <= sequential_memory * 1.10,
        "correctness": bool(sequential["correctness"]["passed"])
        and bool(packed["correctness"]["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "reason": "all performance and correctness thresholds passed"
        if all(checks.values())
        else "one or more performance or correctness thresholds failed",
        "checks": checks,
        "ratios": {
            "median_step": packed_median / sequential_median,
            "optimizer_throughput": packed_throughput / sequential_throughput,
            "p95_step": packed_p95 / sequential_p95,
            "peak_allocated_memory": packed_memory / sequential_memory,
        },
    }


def _comparison_fingerprint(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": result["source"],
        "task": result["task"],
        "runtime": result["runtime"],
        "distributed": result["distributed"],
        "protocol": result["protocol"],
    }


def build_selection(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {}
    for result in results:
        key = (str(result["task"]["task_config"]), str(result["candidate"]))
        if key in indexed:
            raise ValueError(f"Duplicate benchmark result for task/candidate: {key}.")
        indexed[key] = result
    task_gates = {}
    for task_config in CANONICAL_TASKS:
        sequential = indexed.get((task_config, "sequential"))
        packed = indexed.get((task_config, "packed_flex"))
        if sequential is None or packed is None:
            task_gates[task_config] = {
                "passed": False,
                "reason": "missing sequential or packed result",
                "checks": {},
            }
        else:
            task_gates[task_config] = evaluate_pair(sequential, packed)
    packed_default = all(gate["passed"] for gate in task_gates.values())
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "selected_forward_mode": "packed_flex" if packed_default else "sequential",
        "packed_default_enabled": packed_default,
        "reason": "4B and 9B gates passed"
        if packed_default
        else "4B/9B performance evidence is incomplete or below threshold",
        "task_gates": task_gates,
    }


def candidate_environment(
    *,
    task_config: str,
    candidate: str,
    protocol: BenchmarkProtocol,
    result_path: Path,
) -> dict[str, str]:
    _validate_candidate(candidate)
    protocol.validate()
    return {
        "IMAGEWAM_BENCHMARK_NAME": BENCHMARK_NAME,
        "IMAGEWAM_BENCHMARK_TASK_CONFIG": task_config,
        "IMAGEWAM_BENCHMARK_FORWARD_MODE": candidate,
        "IMAGEWAM_BENCHMARK_WARMUP_STEPS": str(protocol.warmup_steps),
        "IMAGEWAM_BENCHMARK_MEASURED_STEPS": str(protocol.measured_steps),
        "IMAGEWAM_BENCHMARK_SEED": str(protocol.seed),
        "IMAGEWAM_BENCHMARK_RESULT_PATH": str(result_path),
        "TORCHINDUCTOR_CACHE_DIR": str(
            result_path.parent / f".torchinductor-{Path(task_config).stem}-{candidate}"
        ),
        "PYTHONHASHSEED": str(protocol.seed),
    }


def run_candidate_process(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    result_path: Path,
) -> dict[str, Any]:
    if not command:
        raise ValueError("Candidate runner command cannot be empty.")
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result: {result_path}")
    process_environment = os.environ.copy()
    process_environment.update(environment)
    subprocess.run(list(command), cwd=cwd, env=process_environment, check=True)
    if not result_path.is_file():
        raise RuntimeError(
            "Candidate runner completed without writing IMAGEWAM_BENCHMARK_RESULT_PATH: "
            f"{result_path}"
        )
    result = read_json(result_path)
    validate_result(result)
    if Path(result["output_path"]).resolve() != result_path.resolve():
        raise ValueError(
            "Candidate result output_path does not match IMAGEWAM_BENCHMARK_RESULT_PATH: "
            f"{result['output_path']!r} != {str(result_path)!r}."
        )
    return result
