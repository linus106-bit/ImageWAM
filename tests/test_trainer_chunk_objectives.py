import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from imagewam.trainer import Wan22Trainer


class _Accelerator:
    def __init__(self, model=None):
        self.model = model
        self.loaded = []
        self.backward_losses = []

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def load_state(self, input_dir):
        self.loaded.append(input_dir)

    def wait_for_everyone(self):
        pass

    def backward(self, loss):
        self.backward_losses.append(loss.detach().item())
        loss.backward()


def _trainer(model=None):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _Accelerator(model)
    return trainer


def _chunk_model(losses=(1.0, 2.0, 3.0, 4.0), **overrides):
    values = {
        "stack": "flux2",
        "resolved_chunk_count": 4,
        "resolved_actions_per_chunk": 16,
        "resolved_total_action_horizon": 64,
        "chunkwise_enabled": True,
        "cache_type": "observation_prefix",
        "supports_chunkwise_training_losses": True,
    }
    values.update(overrides)

    def iter_training_losses(_sample):
        for index, value in enumerate(losses):
            yield torch.tensor(value), {
                "loss_video": value,
                "chunk_count": 1,
                f"chunk/{index}/loss_video": value,
            }

    values["iter_training_losses"] = iter_training_losses
    values["training_loss"] = lambda _sample: (torch.tensor(99.0), {"legacy": 1.0})
    return SimpleNamespace(**values)


class TrainerChunkObjectivesTest(unittest.TestCase):
    def test_validation_sums_all_chunk_losses_and_metrics(self):
        loss, metrics = _trainer()._validation_training_loss(_chunk_model(), {})

        self.assertAlmostEqual(loss.item(), 10.0)
        self.assertAlmostEqual(metrics["loss_video"], 10.0)
        self.assertAlmostEqual(metrics["chunk_count"], 4.0)
        self.assertAlmostEqual(metrics["chunk/3/loss_video"], 4.0)

    def test_chunkwise_iterator_requires_capability_and_exact_k(self):
        trainer = _trainer()
        unsupported = _chunk_model(supports_chunkwise_training_losses=False)
        with self.assertRaisesRegex(RuntimeError, "supports_chunkwise_training_losses"):
            trainer._chunkwise_loss_iterator(unsupported, {})

        with self.assertRaisesRegex(RuntimeError, "expected 4, got 3"):
            trainer._validation_training_loss(_chunk_model(losses=(1.0, 2.0, 3.0)), {})

    def test_inconsistent_or_non_flux_k_greater_than_one_fails_safe(self):
        trainer = _trainer()
        with self.assertRaisesRegex(ValueError, "chunkwise_enabled=True"):
            trainer._chunkwise_loss_iterator(_chunk_model(chunkwise_enabled=False), {})
        with self.assertRaisesRegex(ValueError, "only for the FLUX.2 stack"):
            trainer._chunkwise_loss_iterator(_chunk_model(stack="wan22"), {})

    def test_four_chunk_objectives_backward_before_one_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer_steps = 0

        model = _chunk_model()

        def iter_training_losses(_sample):
            for index in range(4):
                loss = parameter * float(index + 1)
                yield loss, {"loss_video": loss, "chunk_count": 1}

        model.iter_training_losses = iter_training_losses
        trainer = _trainer(model)
        loss, metrics, _, _ = trainer._backward_training_objectives(model, {})

        optimizer.step()
        optimizer_steps += 1
        optimizer.zero_grad(set_to_none=True)

        self.assertEqual(trainer.accelerator.backward_losses, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(optimizer_steps, 1)
        self.assertAlmostEqual(loss.item(), 10.0)
        self.assertAlmostEqual(metrics["loss_video"], 10.0)
        self.assertAlmostEqual(metrics["chunk_count"], 4.0)
        self.assertAlmostEqual(parameter.item(), 0.0)

    def test_k1_and_non_chunkwise_preserve_training_loss_path(self):
        model = _chunk_model(
            stack="wan22",
            resolved_chunk_count=1,
            resolved_actions_per_chunk=16,
            resolved_total_action_horizon=16,
            chunkwise_enabled=False,
            supports_chunkwise_training_losses=False,
        )

        loss, metrics = _trainer()._validation_training_loss(model, {})
        self.assertAlmostEqual(loss.item(), 99.0)
        self.assertEqual(metrics, {"legacy": 1.0})

    def test_trainer_state_persists_and_validates_chunkwise_metadata(self):
        trainer = _trainer(_chunk_model())
        trainer.global_step = 7
        trainer.epoch = 2
        trainer.batch_in_epoch = 3
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer._save_trainer_state(tmp_dir)
            payload = json.loads((Path(tmp_dir) / "trainer_state.json").read_text())

            self.assertEqual(payload["resolved_chunk_count"], 4)
            self.assertEqual(payload["resolved_actions_per_chunk"], 16)
            self.assertEqual(payload["resolved_total_action_horizon"], 64)
            self.assertIs(payload["chunkwise_enabled"], True)
            self.assertEqual(payload["cache_type"], "observation_prefix")
            self.assertIs(payload["supports_chunkwise_training_losses"], True)
            trainer._validate_resume_chunkwise_metadata(payload, tmp_dir)

    def test_resume_rejects_legacy_or_mismatched_chunkwise_full_state(self):
        trainer = _trainer(_chunk_model())
        with self.assertRaisesRegex(ValueError, "Legacy full-state checkpoints can only resume with K=1"):
            trainer._validate_resume_chunkwise_metadata({"global_step": 1}, "legacy")

        payload = {**trainer._chunkwise_training_metadata(), "global_step": 1}
        payload["resolved_chunk_count"] = 2
        with self.assertRaisesRegex(ValueError, "resolved_chunk_count"):
            trainer._validate_resume_chunkwise_metadata(payload, "mismatch")

    def test_legacy_full_state_is_allowed_for_k1(self):
        model = _chunk_model(
            resolved_chunk_count=1,
            resolved_actions_per_chunk=16,
            resolved_total_action_horizon=16,
            chunkwise_enabled=False,
        )
        _trainer(model)._validate_resume_chunkwise_metadata({"global_step": 1}, "legacy-state")


if __name__ == "__main__":
    unittest.main()
