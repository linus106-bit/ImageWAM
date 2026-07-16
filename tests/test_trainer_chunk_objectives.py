import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from imagewam.trainer import Wan22Trainer


class _Accelerator:
    def __init__(self, model=None):
        self.model = model
        self.loaded = []

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def load_state(self, input_dir):
        self.loaded.append(input_dir)

    def wait_for_everyone(self):
        pass


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


def test_validation_sums_all_chunk_losses_and_metrics():
    trainer = _trainer()
    loss, metrics = trainer._validation_training_loss(_chunk_model(), {})

    assert loss.item() == pytest.approx(10.0)
    assert metrics["loss_video"] == pytest.approx(10.0)
    assert metrics["chunk_count"] == pytest.approx(4.0)
    assert metrics["chunk/3/loss_video"] == pytest.approx(4.0)


def test_chunkwise_iterator_requires_capability_and_exact_k():
    trainer = _trainer()
    unsupported = _chunk_model(supports_chunkwise_training_losses=False)
    with pytest.raises(RuntimeError, match="supports_chunkwise_training_losses"):
        trainer._chunkwise_loss_iterator(unsupported, {})

    with pytest.raises(RuntimeError, match="expected 4, got 3"):
        trainer._validation_training_loss(_chunk_model(losses=(1.0, 2.0, 3.0)), {})


def test_k1_and_non_chunkwise_preserve_training_loss_path():
    trainer = _trainer()
    model = _chunk_model(
        stack="wan22",
        resolved_chunk_count=1,
        resolved_actions_per_chunk=16,
        resolved_total_action_horizon=16,
        chunkwise_enabled=False,
        supports_chunkwise_training_losses=False,
    )

    loss, metrics = trainer._validation_training_loss(model, {})
    assert loss.item() == pytest.approx(99.0)
    assert metrics == {"legacy": 1.0}


def test_trainer_state_persists_and_validates_chunkwise_metadata(tmp_path):
    model = _chunk_model()
    trainer = _trainer(model)
    trainer.global_step = 7
    trainer.epoch = 2
    trainer.batch_in_epoch = 3
    trainer._save_trainer_state(str(tmp_path))

    payload = json.loads((tmp_path / "trainer_state.json").read_text())
    assert payload["resolved_chunk_count"] == 4
    assert payload["resolved_actions_per_chunk"] == 16
    assert payload["resolved_total_action_horizon"] == 64
    assert payload["chunkwise_enabled"] is True
    assert payload["cache_type"] == "observation_prefix"
    assert payload["supports_chunkwise_training_losses"] is True
    trainer._validate_resume_chunkwise_metadata(payload, str(tmp_path))


def test_resume_rejects_legacy_or_mismatched_chunkwise_full_state(tmp_path):
    trainer = _trainer(_chunk_model())
    with pytest.raises(ValueError, match="Legacy full-state checkpoints can only resume with K=1"):
        trainer._validate_resume_chunkwise_metadata({"global_step": 1}, str(tmp_path))

    payload = {**trainer._chunkwise_training_metadata(), "global_step": 1}
    payload["resolved_chunk_count"] = 2
    with pytest.raises(ValueError, match="resolved_chunk_count"):
        trainer._validate_resume_chunkwise_metadata(payload, str(tmp_path))


def test_legacy_full_state_is_allowed_for_k1():
    model = _chunk_model(
        resolved_chunk_count=1,
        resolved_actions_per_chunk=16,
        resolved_total_action_horizon=16,
        chunkwise_enabled=False,
    )
    _trainer(model)._validate_resume_chunkwise_metadata({"global_step": 1}, "legacy-state")
