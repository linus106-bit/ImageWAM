import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from imagewam.chunkwise import chunkwise_loss_contribution
from imagewam.models.backbones.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

try:
    import accelerate  # noqa: F401
except ModuleNotFoundError:
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.Accelerator = object
    sys.modules["accelerate"] = accelerate_stub

from imagewam.trainer import Wan22Trainer


class _Accelerator:
    def __init__(self, model=None):
        self.model = model
        self.loaded = []
        self.backward_losses = []
        self.num_processes = 1
        self.distributed_type = "NO"
        self.state = SimpleNamespace(deepspeed_plugin=None)

    def unwrap_model(self, model):
        return getattr(model, "module", model)

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


class _ChunkModel(SimpleNamespace):
    def prepare_chunkwise_training_inputs(self, sample):
        self.prepared_samples.append(sample)
        return {"sample": sample}

    def __call__(
        self,
        sample=None,
        *,
        prepared_chunkwise_inputs=None,
        chunk_index=None,
        tiled=False,
    ):
        del tiled
        if prepared_chunkwise_inputs is not None:
            if sample is not None:
                raise ValueError("sample and prepared inputs are mutually exclusive")
            if self.chunkwise_forward_mode == "packed_flex":
                if chunk_index is not None:
                    raise ValueError("chunk_index is invalid for packed inputs")
                return self.packed_loss_fn()
            if chunk_index is None:
                raise ValueError("chunk_index is required")
            return self.chunk_loss_fn(int(chunk_index))
        if chunk_index is not None:
            raise ValueError("chunk_index requires prepared inputs")
        return self.training_loss(sample)


class _PreparedWrapper:
    """Minimal DDP/DeepSpeed-shaped wrapper that records outer forward calls."""

    def __init__(self, module):
        self.module = module
        self.forward_chunks = []

    def __call__(self, *args, **kwargs):
        self.forward_chunks.append(kwargs.get("chunk_index"))
        return self.module(*args, **kwargs)


def _chunk_model(losses=(1.0, 2.0, 3.0, 4.0), **overrides):
    values = {
        "stack": "flux2",
        "resolved_chunk_count": 4,
        "resolved_actions_per_chunk": 16,
        "resolved_total_action_horizon": 64,
        "chunkwise_causal_enabled": True,
        "chunkwise_cache_type": "observation_prefix",
        "supports_chunkwise_training_losses": True,
        "supports_chunkwise_prepared_forward": True,
        "chunkwise_forward_mode": "sequential",
        "chunkwise_sparse_packing": "interleaved",
        "chunkwise_sparse_block_size": 128,
        "chunkwise_sparse_alignment": "none",
        "chunkwise_packed_layout_schema_version": 3,
        "chunkwise_packed_capability": {"supported": True},
        "prepared_samples": [],
    }
    values.update(overrides)

    def chunk_loss_fn(index):
        value = losses[index]
        return torch.tensor(value), {
            "loss_video": value,
            "chunk_count": 1,
            f"chunk/{index}/loss_video": value,
        }

    values["chunk_loss_fn"] = chunk_loss_fn
    values["packed_loss_fn"] = lambda: (
        torch.tensor(sum(losses)),
        {"loss_video": float(sum(losses)), "chunk_count": float(len(losses))},
    )
    values["training_loss"] = lambda _sample: (torch.tensor(99.0), {"legacy": 1.0})
    return _ChunkModel(**values)


class TrainerChunkObjectivesTest(unittest.TestCase):
    def test_production_scheduler_batch_one_weights_work_in_training_and_validation(self):
        parameter = torch.nn.Parameter(torch.tensor([[2.0]]))
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=8, shift=2.0)
        scalar_weight = scheduler.training_weight(torch.tensor([4.0]))
        model = _chunk_model()

        def chunk_loss_fn(_index):
            return chunkwise_loss_contribution(
                video_squared_error=parameter.square(),
                action_squared_error=parameter.square(),
                valid_target=torch.tensor([True]),
                video_weight=scalar_weight,
                action_weight=scalar_weight,
                video_denominator=torch.ones(1),
                action_denominator=torch.ones(1),
                lambda_video=1.0,
                lambda_action=1.0,
            )

        model.chunk_loss_fn = chunk_loss_fn
        trainer = _trainer(model)

        validation_loss, _ = trainer._validation_training_loss(model, {})
        training_loss, _, _, _ = trainer._backward_training_objectives(model, {})

        self.assertTrue(torch.isfinite(validation_loss))
        self.assertTrue(torch.isfinite(training_loss))
        self.assertIsNotNone(parameter.grad)

    def test_validation_sums_all_chunk_losses_and_metrics(self):
        loss, metrics = _trainer()._validation_training_loss(_chunk_model(), {})

        self.assertAlmostEqual(loss.item(), 10.0)
        self.assertAlmostEqual(metrics["loss_video"], 10.0)
        self.assertAlmostEqual(metrics["chunk_count"], 4.0)
        self.assertAlmostEqual(metrics["chunk/3/loss_video"], 4.0)

    def test_prepare_chunkwise_forward_requires_capabilities_and_prepare_callable(self):
        trainer = _trainer()
        unsupported = _chunk_model(supports_chunkwise_training_losses=False)
        with self.assertRaisesRegex(RuntimeError, "supports_chunkwise_training_losses"):
            trainer._prepare_chunkwise_forward(unsupported, {})

        unsupported_prepared = _chunk_model(supports_chunkwise_prepared_forward=False)
        with self.assertRaisesRegex(RuntimeError, "supports_chunkwise_prepared_forward"):
            trainer._prepare_chunkwise_forward(unsupported_prepared, {})

        missing_prepare = _chunk_model()
        missing_prepare.prepare_chunkwise_training_inputs = None
        with self.assertRaisesRegex(RuntimeError, "prepare_chunkwise_training_inputs"):
            trainer._prepare_chunkwise_forward(missing_prepare, {})

    def test_prepare_chunkwise_forward_strictly_validates_mode_sample_and_prepared_result(self):
        trainer = _trainer()
        with self.assertRaisesRegex(ValueError, "forward_mode"):
            trainer._prepare_chunkwise_forward(
                _chunk_model(chunkwise_forward_mode="packed_typo"), {}
            )
        with self.assertRaisesRegex(ValueError, "dict sample"):
            trainer._prepare_chunkwise_forward(_chunk_model(), object())

        malformed = _chunk_model()
        malformed.prepare_chunkwise_training_inputs = lambda _sample: []
        with self.assertRaisesRegex(ValueError, "must return a dict"):
            trainer._prepare_chunkwise_forward(malformed, {})

    def test_inconsistent_or_non_flux_k_greater_than_one_fails_safe(self):
        trainer = _trainer()
        with self.assertRaisesRegex(ValueError, "chunkwise_enabled=True"):
            trainer._prepare_chunkwise_forward(_chunk_model(chunkwise_causal_enabled=False), {})
        with self.assertRaisesRegex(ValueError, "only for the FLUX.2 stack"):
            trainer._prepare_chunkwise_forward(_chunk_model(stack="wan22"), {})

    def test_four_chunk_objectives_backward_before_one_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer_steps = 0

        model = _chunk_model()

        def chunk_loss_fn(index):
            loss = parameter * float(index + 1)
            return loss, {"loss_video": loss, "chunk_count": 1}

        model.chunk_loss_fn = chunk_loss_fn
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

    def test_packed_chunk_objective_uses_one_forward_and_one_backward(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        model = _chunk_model(chunkwise_forward_mode="packed_flex")

        def packed_loss_fn():
            loss = parameter * 10.0
            return loss, {"loss_video": loss, "chunk_count": 4.0}

        model.packed_loss_fn = packed_loss_fn
        trainer = _trainer(model)
        loss, metrics, _, _ = trainer._backward_training_objectives(model, {"batch": 1})

        self.assertEqual(trainer.accelerator.backward_losses, [10.0])
        self.assertEqual(model.prepared_samples, [{"batch": 1}])
        self.assertAlmostEqual(loss.item(), 10.0)
        self.assertAlmostEqual(metrics["loss_video"], 10.0)
        self.assertAlmostEqual(metrics["chunk_count"], 4.0)
        self.assertAlmostEqual(parameter.grad.item(), 10.0)

    def test_packed_chunk_objective_uses_outer_prepared_wrapper_once(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        module = _chunk_model(chunkwise_forward_mode="packed_flex")

        def packed_loss_fn():
            return parameter * 10.0, {"loss_video": parameter * 10.0, "chunk_count": 4.0}

        module.packed_loss_fn = packed_loss_fn
        wrapped = _PreparedWrapper(module)
        trainer = _trainer(wrapped)

        loss, metrics, _, _ = trainer._backward_training_objectives(wrapped, {"batch": 1})

        self.assertEqual(module.prepared_samples, [{"batch": 1}])
        self.assertEqual(wrapped.forward_chunks, [None])
        self.assertEqual(trainer.accelerator.backward_losses, [10.0])
        self.assertAlmostEqual(loss.item(), 10.0)
        self.assertAlmostEqual(metrics["chunk_count"], 4.0)
        self.assertAlmostEqual(parameter.grad.item(), 10.0)

    def test_packed_validation_uses_one_prepared_forward(self):
        model = _chunk_model(chunkwise_forward_mode="packed_flex")
        calls = []

        def packed_loss_fn():
            calls.append("packed")
            return torch.tensor(10.0), {"loss_video": 10.0, "chunk_count": 4.0}

        model.packed_loss_fn = packed_loss_fn
        loss, metrics = _trainer(model)._validation_training_loss(model, {"batch": 1})

        self.assertEqual(calls, ["packed"])
        self.assertEqual(model.prepared_samples, [{"batch": 1}])
        self.assertAlmostEqual(loss.item(), 10.0)
        self.assertAlmostEqual(metrics["chunk_count"], 4.0)

    def test_chunk_objectives_use_outer_prepared_wrapper_for_every_forward(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        module = _chunk_model()

        def chunk_loss_fn(index):
            loss = parameter * float(index + 1)
            return loss, {"chunk_count": 1.0}

        module.chunk_loss_fn = chunk_loss_fn
        wrapped = _PreparedWrapper(module)
        trainer = _trainer(wrapped)

        trainer._backward_training_objectives(wrapped, {"batch": 1})

        self.assertEqual(module.prepared_samples, [{"batch": 1}])
        self.assertEqual(wrapped.forward_chunks, [0, 1, 2, 3])
        self.assertEqual(trainer.accelerator.backward_losses, [1.0, 2.0, 3.0, 4.0])

    def test_distributed_contract_rejects_unwrapped_multi_process_and_zero3(self):
        module = _chunk_model()
        trainer = _trainer(module)
        trainer.accelerator.num_processes = 2
        with self.assertRaisesRegex(RuntimeError, "did not wrap"):
            trainer._validate_chunkwise_distributed_contract()

        wrapped = _PreparedWrapper(module)
        trainer.model = wrapped
        trainer.accelerator.distributed_type = "MULTI_CPU"
        trainer._validate_chunkwise_distributed_contract()

        trainer.accelerator.distributed_type = "FSDP"
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            trainer._validate_chunkwise_distributed_contract()
        trainer.accelerator.distributed_type = "TPU"
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            trainer._validate_chunkwise_distributed_contract()
        trainer.accelerator.distributed_type = "DEEPSPEED"

        trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(
            deepspeed_config={"zero_optimization": {"stage": 3}}
        )
        with self.assertRaisesRegex(RuntimeError, "ZeRO stage 3"):
            trainer._validate_chunkwise_distributed_contract()

    def test_k1_and_non_chunkwise_preserve_training_loss_path(self):
        model = _chunk_model(
            stack="wan22",
            resolved_chunk_count=1,
            resolved_actions_per_chunk=16,
            resolved_total_action_horizon=16,
            chunkwise_causal_enabled=False,
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
            self.assertEqual(payload["forward_mode"], "sequential")
            self.assertEqual(payload["sparse_packing"], "interleaved")
            self.assertEqual(payload["sparse_block_size"], 128)
            self.assertEqual(payload["sparse_alignment"], "none")
            self.assertEqual(payload["packed_layout_schema_version"], 3)
            self.assertIn("torch_major_minor", payload)
            self.assertIs(payload["supports_chunkwise_training_losses"], True)
            trainer._validate_resume_chunkwise_metadata(payload, tmp_dir)
            self.assertEqual(list(Path(tmp_dir).glob(".trainer_state.json.*.tmp")), [])

    def test_canonical_model_metadata_wins_over_legacy_aliases(self):
        model = _chunk_model(chunkwise_enabled=False, cache_type="legacy-cache")
        metadata = _trainer(model)._chunkwise_training_metadata()

        self.assertIs(metadata["chunkwise_enabled"], True)
        self.assertEqual(metadata["cache_type"], "observation_prefix")

    def test_resume_rejects_legacy_or_mismatched_chunkwise_full_state(self):
        trainer = _trainer(_chunk_model())
        with self.assertRaisesRegex(ValueError, "Legacy full-state checkpoints can only resume with K=1"):
            trainer._validate_resume_chunkwise_metadata({"global_step": 1}, "legacy")

        payload = {**trainer._chunkwise_training_metadata(), "global_step": 1}
        payload["resolved_chunk_count"] = 2
        with self.assertRaisesRegex(ValueError, "resolved_chunk_count"):
            trainer._validate_resume_chunkwise_metadata(payload, "mismatch")

        payload = {**trainer._chunkwise_training_metadata(), "global_step": 1}
        payload["forward_mode"] = "packed_flex"
        with self.assertRaisesRegex(ValueError, "forward_mode"):
            trainer._validate_resume_chunkwise_metadata(payload, "mode-mismatch")

        inconsistent = _chunk_model(chunkwise_causal_enabled=False)
        with self.assertRaisesRegex(ValueError, "Legacy full-state checkpoints can only resume with K=1"):
            _trainer(inconsistent)._validate_resume_chunkwise_metadata(
                {"global_step": 1}, "inconsistent-legacy"
            )

    def test_legacy_full_state_is_allowed_for_k1(self):
        model = _chunk_model(
            resolved_chunk_count=1,
            resolved_actions_per_chunk=16,
            resolved_total_action_horizon=16,
            chunkwise_causal_enabled=False,
        )
        _trainer(model)._validate_resume_chunkwise_metadata({"global_step": 1}, "legacy-state")

    def test_resume_rejects_invalid_progress_before_loading_accelerator_state(self):
        trainer = _trainer(_chunk_model())
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = trainer._chunkwise_training_metadata()
            payload["global_step"] = -1
            (Path(tmp_dir) / "trainer_state.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "global_step"):
                trainer.load_training_state(tmp_dir)

        self.assertEqual(trainer.accelerator.loaded, [])

    def test_resume_requires_epoch_and_batch_progress_as_a_pair(self):
        trainer = _trainer(_chunk_model())
        payload = {**trainer._chunkwise_training_metadata(), "global_step": 1, "epoch": 2}
        with self.assertRaisesRegex(ValueError, "both `epoch` and `batch_in_epoch`"):
            trainer._validate_trainer_state_progress(payload, "partial-progress")


if __name__ == "__main__":
    unittest.main()
