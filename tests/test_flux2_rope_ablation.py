import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.flux2.run_rope_ablation import (
    build_command,
    parse_metrics,
    parse_metrics_jsonl,
    paired_bootstrap_interval,
    parse_schemes,
    parse_seeds,
    validate_extra_overrides,
    write_summaries,
)


class Flux2RopeAblationLauncherTest(unittest.TestCase):
    def test_parses_candidate_lists_and_rejects_unknown_scheme(self):
        self.assertEqual(parse_schemes("current,chronological_full"), ["current", "chronological_full"])
        self.assertEqual(parse_seeds("42, 43"), [42, 43])
        with self.assertRaisesRegex(ValueError, "Unknown RoPE schemes"):
            parse_schemes("current,magic")
        with self.assertRaisesRegex(ValueError, "locked keys"):
            validate_extra_overrides(["seed=9"])

    def test_build_command_contains_reproducibility_overrides(self):
        args = argparse.Namespace(
            forward_mode="packed_flex",
            max_steps=20,
            eval_every=10,
            eval_num_samples=8,
            save_every=20,
            wandb_group="test-group",
            wandb_mode="offline",
            override=["batch_size=1"],
        )
        command = build_command(args, "temporal_action", 43, Path("/tmp/run"))
        self.assertIn("model.chunkwise_causal.rope_scheme=temporal_action", command)
        self.assertIn("model.chunkwise_causal.forward_mode=packed_flex", command)
        self.assertIn("seed=43", command)
        self.assertIn("output_dir=/tmp/run", command)
        self.assertLess(command.index("batch_size=1"), command.index("seed=43"))

    def test_extracts_last_train_and_eval_record(self):
        metrics = parse_metrics(
            [
                "INFO [train] epoch=0 step=10/20 loss=1.25 lr=1e-4",
                "INFO [eval] step=10 samples=8 val_loss=2.0 infer_psnr=10.0 infer_ssim=0.5 action_l2=0.4 action_l1=0.3",
                "INFO [train] epoch=0 step=20/20 loss=1.00 lr=1e-4",
                "INFO [eval] step=20 samples=8 val_loss=1.5 infer_psnr=11.0 infer_ssim=0.6 action_l2=0.2 action_l1=0.1",
            ]
        )
        self.assertEqual(metrics["step"], 20)
        self.assertEqual(metrics["train_step"], 20)
        self.assertEqual(metrics["val_loss"], 1.5)
        self.assertEqual(metrics["action_l1"], 0.1)

    def test_reads_full_precision_jsonl_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.jsonl"
            path.write_text(
                '{"phase":"train","step":10,"train/loss":1.123456789}\n'
                '{"phase":"eval","step":10,"eval/num_samples":8,'
                '"eval/val_loss":0.123456789,"eval/psnr_rd":19.25,'
                '"eval/ssim_rd":0.8125,"eval/action_l1":0.03125}\n',
                encoding="utf-8",
            )
            metrics = parse_metrics_jsonl(path)
        self.assertEqual(metrics["train_loss"], 1.123456789)
        self.assertEqual(metrics["val_loss"], 0.123456789)
        self.assertEqual(metrics["action_l1"], 0.03125)

    def test_summary_writes_seed_rows_and_variant_aggregates(self):
        results = [
            {
                "scheme": "current",
                "seed": 42,
                "status": "completed",
                "returncode": 0,
                "metrics": {"val_loss": 2.0},
            },
            {
                "scheme": "current",
                "seed": 43,
                "status": "completed",
                "returncode": 0,
                "metrics": {"val_loss": 1.0},
            },
            {
                "scheme": "chronological_full",
                "seed": 42,
                "status": "completed",
                "returncode": 0,
                "metrics": {"val_loss": 1.0},
            },
            {
                "scheme": "chronological_full",
                "seed": 43,
                "status": "completed",
                "returncode": 0,
                "metrics": {"val_loss": 0.5},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            write_summaries(output_root, results)
            aggregate = (output_root / "aggregate.csv").read_text(encoding="utf-8")
            self.assertIn("val_loss_mean", aggregate)
            self.assertIn("1.5", aggregate)
            self.assertTrue((output_root / "summary.json").exists())
            comparisons = (output_root / "paired_comparisons.csv").read_text(encoding="utf-8")
            self.assertIn("chronological_full,val_loss,lower,2", comparisons)
            self.assertIn("insufficient", comparisons)

    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap_interval([-1.0, -0.5, -0.25], samples=1000, seed=7)
        second = paired_bootstrap_interval([-1.0, -0.5, -0.25], samples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertLess(first[1], 0.0)


if __name__ == "__main__":
    unittest.main()
