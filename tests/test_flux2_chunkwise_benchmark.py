import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from imagewam.benchmark import (
    BENCHMARK_NAME,
    BENCHMARK_SCHEMA_VERSION,
    CANONICAL_TASKS,
    BenchmarkProtocol,
    build_selection,
    candidate_environment,
    evaluate_pair,
    resolve_task_spec,
    summarize_rank_measurements,
    unavailable_result,
    validate_result,
    write_immutable_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _completed_result(task_config: str, candidate: str) -> dict:
    chunks = 4
    measured_steps = 30
    multiplier = 1 if candidate == "packed_flex" else chunks
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "completed",
        "candidate": candidate,
        "source": {"commit": "abc", "dirty": False},
        "task": {
            "task_config": task_config,
            "model_config": "configs/model/example.yaml",
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "precision": "bf16",
            "num_chunks": 4,
            "actions_per_chunk": 16,
            "total_action_horizon": 64,
            "forward_mode_default": "sequential",
            "sparse_packing": "batch_padded",
            "sparse_block_size": 128,
            "sparse_alignment": "none",
            "layout_schema_version": 1,
        },
        "runtime": {
            "torch": "2.7.1",
            "cuda_available": True,
            "cuda_runtime": "12.8",
            "gpu": "test-gpu",
            "driver": "test-driver",
        },
        "distributed": {"world_size": 1, "rank": 0, "backend": "none", "zero_stage": 0},
        "protocol": {"warmup_steps": 10, "measured_steps": 30, "seed": 0},
        "correctness": {
            "passed": True,
            "finite_loss": True,
            "finite_gradients": True,
            "oom": False,
        },
        "metrics": {
            "compile_warmup_seconds": 3.0,
            "forward_seconds": {"median": 0.6, "p95": 0.7},
            "backward_seconds": {"median": 0.4, "p95": 0.5},
            "total_step_seconds": {"median": 1.0, "p95": 1.2},
            "logical_trajectories_per_second": {"microbatch": 4.0, "optimizer_step": 8.0},
            "peak_memory_bytes": {"allocated": 1000, "reserved": 1200},
            "packed": {
                "tokens": 100,
                "theoretical_dense_token_pairs": 10000,
                "visited_sparse_blocks": 4,
                "block_sparsity": 0.75,
            },
            "call_counts": {
                "outer_model": measured_steps * multiplier,
                "mot": measured_steps * multiplier,
                "flex_attention": measured_steps if candidate == "packed_flex" else 0,
                "backward": measured_steps * multiplier,
            },
            "per_rank": [],
            "max_rank": {},
        },
        "output_path": "result.json",
    }
    _set_gate_metrics(
        result,
        total_median=1.0,
        total_p95=1.2,
        optimizer_throughput=8.0,
        peak_memory=1000,
    )
    return result


def _samples(median: float, p95: float) -> list[float]:
    return [median] * 27 + [p95] * 3


def _set_gate_metrics(
    result: dict,
    *,
    total_median: float,
    total_p95: float,
    optimizer_throughput: float,
    peak_memory: int,
) -> None:
    task = result["task"]
    optimizer_seconds = (
        task["batch_size"] * task["gradient_accumulation_steps"] / optimizer_throughput
    )
    per_rank = [
        {
            "rank": 0,
            "forward_seconds": _samples(0.6, 0.7),
            "backward_seconds": _samples(0.4, 0.5),
            "total_step_seconds": _samples(total_median, total_p95),
            "optimizer_step_seconds": optimizer_seconds,
            "peak_allocated_bytes": peak_memory,
            "peak_reserved_bytes": peak_memory + 200,
        }
    ]
    summary = summarize_rank_measurements(
        per_rank=per_rank,
        local_batch=task["batch_size"],
        world_size=1,
        gradient_accumulation=task["gradient_accumulation_steps"],
    )
    result["metrics"].update(summary)


class BenchmarkProtocolTests(unittest.TestCase):
    def test_official_protocol_is_strict(self):
        BenchmarkProtocol().validate()
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            BenchmarkProtocol(warmup_steps=9).validate()
        with self.assertRaisesRegex(ValueError, "at least 30"):
            BenchmarkProtocol(measured_steps=29).validate()

    def test_candidate_environment_is_complete_and_deterministic(self):
        environment = candidate_environment(
            task_config=CANONICAL_TASKS[0],
            candidate="packed_flex",
            protocol=BenchmarkProtocol(seed=7),
            result_path=Path("/tmp/result.json"),
        )
        self.assertEqual(environment["IMAGEWAM_BENCHMARK_FORWARD_MODE"], "packed_flex")
        self.assertEqual(environment["IMAGEWAM_BENCHMARK_WARMUP_STEPS"], "10")
        self.assertEqual(environment["IMAGEWAM_BENCHMARK_MEASURED_STEPS"], "30")
        self.assertEqual(environment["PYTHONHASHSEED"], "7")
        self.assertIn("packed_flex", environment["TORCHINDUCTOR_CACHE_DIR"])

    def test_canonical_configs_resolve_geometry_and_keep_sequential_default(self):
        for task_config, expected_batch, expected_accumulation in (
            (CANONICAL_TASKS[0], 12, 4),
            (CANONICAL_TASKS[1], 4, 2),
        ):
            with self.subTest(task_config=task_config):
                spec = resolve_task_spec(REPO_ROOT, task_config)
                self.assertEqual(spec["num_chunks"], 4)
                self.assertEqual(spec["actions_per_chunk"], 16)
                self.assertEqual(spec["total_action_horizon"], 64)
                self.assertEqual(spec["batch_size"], expected_batch)
                self.assertEqual(spec["gradient_accumulation_steps"], expected_accumulation)
                self.assertEqual(spec["forward_mode_default"], "sequential")
                self.assertEqual(spec["sparse_packing"], "batch_padded")


class BenchmarkResultTests(unittest.TestCase):
    def test_unavailable_result_validates_without_claiming_correctness(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = unavailable_result(
                repo_root=REPO_ROOT,
                task_spec=resolve_task_spec(REPO_ROOT, CANONICAL_TASKS[0]),
                candidate="packed_flex",
                protocol=BenchmarkProtocol(),
                output_path=Path(tmp_dir) / "result.json",
                reason="no CUDA",
            )
        validate_result(result)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["correctness"]["passed"])

    def test_immutable_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "result.json"
            write_immutable_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_immutable_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_rank_summary_uses_max_rank_and_optimizer_formula(self):
        summary = summarize_rank_measurements(
            per_rank=(
                {
                    "rank": 0,
                    "forward_seconds": [0.4, 0.6],
                    "backward_seconds": [0.3, 0.5],
                    "total_step_seconds": [1.0, 1.2],
                    "optimizer_step_seconds": 2.0,
                    "peak_allocated_bytes": 100,
                    "peak_reserved_bytes": 120,
                },
                {
                    "rank": 1,
                    "forward_seconds": [0.5, 0.7],
                    "backward_seconds": [0.4, 0.6],
                    "total_step_seconds": [1.1, 1.3],
                    "optimizer_step_seconds": 2.5,
                    "peak_allocated_bytes": 110,
                    "peak_reserved_bytes": 140,
                },
            ),
            local_batch=4,
            world_size=2,
            gradient_accumulation=3,
        )
        self.assertAlmostEqual(summary["total_step_seconds"]["median"], 1.2)
        self.assertAlmostEqual(
            summary["logical_trajectories_per_second"]["optimizer_step"], 9.6
        )
        self.assertEqual(summary["peak_memory_bytes"]["allocated"], 110)

    def test_gate_accepts_speedup_inside_tail_and_memory_limits(self):
        sequential = _completed_result(CANONICAL_TASKS[0], "sequential")
        packed = _completed_result(CANONICAL_TASKS[0], "packed_flex")
        _set_gate_metrics(
            packed,
            total_median=0.89,
            total_p95=1.25,
            optimizer_throughput=9.0,
            peak_memory=1090,
        )
        decision = evaluate_pair(sequential, packed)
        self.assertTrue(decision["passed"])

    def test_gate_rejects_mismatched_fingerprint_and_stale_aggregates(self):
        sequential = _completed_result(CANONICAL_TASKS[0], "sequential")
        packed = _completed_result(CANONICAL_TASKS[0], "packed_flex")
        packed["source"]["commit"] = "different"
        with self.assertRaisesRegex(ValueError, "identical source"):
            evaluate_pair(sequential, packed)

        packed = _completed_result(CANONICAL_TASKS[0], "packed_flex")
        packed["metrics"]["total_step_seconds"]["median"] = 0.01
        with self.assertRaisesRegex(ValueError, "raw per-rank evidence"):
            validate_result(packed)

    def test_selection_requires_both_4b_and_9b_gates(self):
        results = []
        for task_config in CANONICAL_TASKS:
            sequential = _completed_result(task_config, "sequential")
            packed = _completed_result(task_config, "packed_flex")
            _set_gate_metrics(
                packed,
                total_median=0.89,
                total_p95=1.25,
                optimizer_throughput=9.0,
                peak_memory=1090,
            )
            results.extend((sequential, packed))
        self.assertEqual(build_selection(results)["selected_forward_mode"], "packed_flex")

        failed_9b = deepcopy(results)
        _set_gate_metrics(
            failed_9b[-1],
            total_median=0.89,
            total_p95=1.25,
            optimizer_throughput=9.0,
            peak_memory=1200,
        )
        decision = build_selection(failed_9b)
        self.assertEqual(decision["selected_forward_mode"], "sequential")
        self.assertFalse(decision["packed_default_enabled"])


if __name__ == "__main__":
    unittest.main()
