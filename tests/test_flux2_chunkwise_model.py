import os
import socket
import types
import unittest
from collections import namedtuple
from dataclasses import dataclass
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from imagewam.chunkwise import PackedBlockSparseMask
from imagewam.models.backbones.imagewam import ImageWAM


def _bare_model() -> ImageWAM:
    model = ImageWAM.__new__(ImageWAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.text_encoder = None
    model.tokenizer = None
    model.vae = nn.Identity()
    return model


class _Scheduler:
    num_train_timesteps = 1

    @staticmethod
    def sample_training_t(batch_size, device, dtype):
        return torch.ones(batch_size, device=device, dtype=dtype)

    @staticmethod
    def add_noise(clean, _noise, _timestep):
        return clean

    @staticmethod
    def training_target(clean, _noise, _timestep):
        return torch.zeros_like(clean)

    @staticmethod
    def training_weight(timestep):
        return torch.ones_like(timestep)


class _RecordingScheduler:
    num_train_timesteps = 8

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.sample_count = 0

    def sample_training_t(self, batch_size, device, dtype):
        self.sample_count += 1
        self.calls.append(f"{self.name}:sample:{self.sample_count}")
        return torch.full(
            (batch_size,),
            float(self.sample_count),
            device=device,
            dtype=dtype,
        )

    def add_noise(self, clean, noise, timestep):
        self.calls.append(f"{self.name}:add:{int(timestep[0].item())}")
        while timestep.ndim < clean.ndim:
            timestep = timestep.unsqueeze(-1)
        return clean + (0.125 * noise.to(dtype=clean.dtype)) + timestep.to(dtype=clean.dtype)

    def training_target(self, clean, noise, timestep):
        del clean
        self.calls.append(f"{self.name}:target:{int(timestep[0].item())}")
        while timestep.ndim < noise.ndim:
            timestep = timestep.unsqueeze(-1)
        return noise.to(dtype=torch.float32) + timestep.to(dtype=torch.float32)

    def training_weight(self, timestep):
        return torch.ones_like(timestep)


class _VideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix_lengths = []
        self.prefix_ordinals = []

    def pre_dit(
        self,
        *,
        x,
        context,
        context_mask,
        ref_image_hidden_states,
        target_img_ids,
        ref_img_ids,
        **_kwargs,
    ):
        self.prefix_lengths.append(int(ref_image_hidden_states.shape[1]))
        self.prefix_ordinals.append(ref_img_ids[0, :, 0].tolist())
        img_tokens = torch.cat([ref_image_hidden_states, x], dim=1)
        return {
            "tokens": {"txt": context, "img": img_tokens},
            "freqs": {"txt": torch.zeros_like(context), "img": torch.zeros_like(img_tokens)},
            "t_mod": {},
            "text_mask": context_mask,
            "target_len": int(x.shape[1]),
            "cond_len": int(ref_image_hidden_states.shape[1]),
            "target_img_ids": target_img_ids,
        }

    @staticmethod
    def post_dit(tokens, pre_state):
        start = int(pre_state["cond_len"])
        return tokens["img"][:, start : start + int(pre_state["target_len"])]


class _ActionExpert(nn.Module):
    @staticmethod
    def pre_dit(action_tokens, **_kwargs):
        batch_size, seq_len = action_tokens.shape[:2]
        return {
            "tokens": action_tokens,
            "ids": torch.zeros(batch_size, seq_len, 4),
            "t_mod": {},
        }

    @staticmethod
    def post_dit(tokens, _pre_state):
        return tokens


class _Mot(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.masks = []

    def forward(self, embeds_all, attention_mask, **_kwargs):
        mask = attention_mask["double_joint"]
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().clone()
        self.masks.append(mask)
        return {
            "video": {
                "txt": embeds_all["video"]["txt"] * self.scale,
                "img": embeds_all["video"]["img"] * self.scale,
            },
            "action": embeds_all["action"] * self.scale,
        }

    def forward_flux2_interleaved(self, *, video_pre_states, action_pre_states, attention_mask):
        self.masks.append(attention_mask["double_joint"])
        return [
            {
                "video": {
                    "txt": video_pre["tokens"]["txt"] * self.scale,
                    "img": video_pre["tokens"]["img"] * self.scale,
                },
                "action": action_pre["tokens"] * self.scale,
            }
            for video_pre, action_pre in zip(video_pre_states, action_pre_states, strict=True)
        ]


def _raise_legacy_training_loss(*_args, **_kwargs):
    raise AssertionError("chunkwise forward must not call legacy training_loss")


def _chunkwise_forward_contract_model(input_scale=1.0):
    model = _bare_model()
    model.stack = "flux2"
    model.chunkwise_causal_enabled = True
    model.resolved_chunk_count = 2
    model.resolved_actions_per_chunk = 1
    model.resolved_total_action_horizon = 2
    model.chunkwise_forward_mode = "sequential"
    model.chunkwise_sparse_packing = "interleaved"
    model.chunkwise_sparse_alignment = "none"
    model.chunkwise_sparse_block_size = 128
    model.proprio_encoder = None
    model.supports_chunkwise_prepared_forward = True
    model.loss_lambda_video = 1.0
    model.loss_lambda_action = 1.0
    model.train_video_scheduler = _Scheduler()
    model.train_action_scheduler = _Scheduler()
    model.video_expert = _VideoExpert()
    model.action_expert = _ActionExpert()
    model.mot = _Mot()
    model.dit = model.mot
    model.training_loss = _raise_legacy_training_loss
    observations = [
        {
            "tokens": torch.full((1, 1, 2), input_scale * float(index + 1)),
            "clean_ids": torch.tensor([[[10.0 + index, 0, 0, 0]]]),
            "target_ids": torch.tensor([[[float(index), 0, 0, 0]]]),
        }
        for index in range(3)
    ]
    prepared = {
        "observations": observations,
        "text_hidden_states": torch.zeros(1, 2, 2),
        "text_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "proprio": None,
        "action": torch.ones(1, 2, 2) * input_scale,
        "action_is_pad": torch.tensor([[False, True]]),
        "action_dim_is_pad": torch.zeros(1, 2, dtype=torch.bool),
        "target_valid": torch.tensor([[True, False]]),
        "observation_valid": torch.tensor([[True, False, True]]),
        "video_denominator": torch.tensor([2.0]),
        "action_denominator": torch.tensor([2.0]),
    }
    model._build_flux2_chunkwise_inputs = lambda *_args, **_kwargs: prepared
    return model


def _free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_chunkwise_forward_worker(rank, world_size, port, queue):
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        model = _chunkwise_forward_contract_model(input_scale=float(rank + 1))
        ddp_model = DistributedDataParallel(model)
        prepared = ddp_model.module.prepare_chunkwise_training_inputs({})
        loss_value = 0.0
        chunk_count = 0.0
        for chunk_index in range(ddp_model.module.resolved_chunk_count):
            loss, metrics = ddp_model(
                prepared_chunkwise_inputs=prepared,
                chunk_index=chunk_index,
            )
            loss_value += float(loss.detach().item())
            chunk_count += float(metrics["chunk_count"])
            loss.backward()
        queue.put(
            {
                "rank": rank,
                "loss": loss_value,
                "grad": float(ddp_model.module.mot.scale.grad.detach().item()),
                "chunk_count": chunk_count,
                "masks": len(ddp_model.module.mot.masks),
            }
        )
    except BaseException as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _accelerate_chunkwise_forward_worker(rank, world_size, port, queue):
    try:
        os.environ.update(
            {
                "ACCELERATE_USE_CPU": "true",
                "LOCAL_RANK": str(rank),
                "LOCAL_WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
            }
        )
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True)
        model = _chunkwise_forward_contract_model(input_scale=float(rank + 1))
        prepared_model = accelerator.prepare(model)
        unwrapped_model = accelerator.unwrap_model(prepared_model)
        prepared = unwrapped_model.prepare_chunkwise_training_inputs({})
        loss_value = 0.0
        chunk_count = 0.0
        for chunk_index in range(unwrapped_model.resolved_chunk_count):
            loss, metrics = prepared_model(
                prepared_chunkwise_inputs=prepared,
                chunk_index=chunk_index,
            )
            loss_value += float(loss.detach().item())
            chunk_count += float(metrics["chunk_count"])
            accelerator.backward(loss)
        accelerator.wait_for_everyone()
        queue.put(
            {
                "distributed_type": str(accelerator.distributed_type),
                "grad": float(unwrapped_model.mot.scale.grad.detach().item()),
                "loss": loss_value,
                "masks": len(unwrapped_model.mot.masks),
                "rank": rank,
                "chunk_count": chunk_count,
            }
        )
    except BaseException as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class Flux2ChunkwiseModelTest(unittest.TestCase):
    def test_packed_payload_concatenates_nested_modulation_structures(self):
        modulation = namedtuple("Modulation", ("shift", "scale", "gate"))

        @dataclass(frozen=True)
        class Container:
            primary: object
            auxiliary: object

        payloads = []
        for value in (1.0, 2.0):
            tensor = torch.full((2, 1), value)
            payloads.append(
                {
                    "tuple": (
                        modulation(tensor, tensor + 1, tensor + 2),
                        modulation(tensor + 3, tensor + 4, tensor + 5),
                    ),
                    "list": [tensor + 6],
                    "dataclass": Container(primary=tensor + 7, auxiliary=None),
                }
            )

        packed = ImageWAM._cat_packed_payload(payloads)

        self.assertEqual(tuple(packed["tuple"][0].shift.shape), (4, 1))
        self.assertEqual(packed["tuple"][0].shift[:, 0].tolist(), [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(packed["tuple"][1].gate[:, 0].tolist(), [6.0, 6.0, 7.0, 7.0])
        self.assertEqual(packed["list"][0][:, 0].tolist(), [7.0, 7.0, 8.0, 8.0])
        self.assertEqual(packed["dataclass"].primary[:, 0].tolist(), [8.0, 8.0, 9.0, 9.0])
        self.assertIsNone(packed["dataclass"].auxiliary)

        with self.assertRaisesRegex(TypeError, "optional payloads"):
            ImageWAM._cat_packed_payload([None, torch.ones(1)])

    def test_disabled_chunkwise_config_resolves_effective_k1_and_capability(self):
        disabled = ImageWAM(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            text_dim=2,
            stack="flux2",
            chunkwise_causal={"enabled": False, "num_chunks": 4},
        )
        enabled = ImageWAM(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            text_dim=2,
            stack="flux2",
            chunkwise_causal={"enabled": True, "num_chunks": 4},
        )
        explicit_k1 = ImageWAM(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            text_dim=2,
            stack="flux2",
            chunkwise_causal={"enabled": True, "num_chunks": 1},
        )
        non_flux_disabled = ImageWAM(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            text_dim=2,
            stack="wan22",
            chunkwise_causal={"enabled": False, "num_chunks": 4},
        )

        self.assertEqual(disabled.resolved_chunk_count, 1)
        self.assertFalse(disabled.supports_chunkwise_training_losses)
        self.assertTrue(enabled.supports_chunkwise_training_losses)
        self.assertFalse(explicit_k1.supports_chunkwise_training_losses)
        self.assertEqual(non_flux_disabled.resolved_chunk_count, 1)
        self.assertFalse(non_flux_disabled.supports_chunkwise_training_losses)
        with self.assertRaisesRegex(ValueError, "only by the FLUX.2 stack"):
            ImageWAM(
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                text_dim=2,
                stack="wan22",
                chunkwise_causal={"enabled": True, "num_chunks": 4},
            )


    def test_chunkwise_forward_mode_defaults_to_sequential_and_validates_sparse_config(self):
        sequential = ImageWAM(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            text_dim=2,
            stack="flux2",
            chunkwise_causal={"enabled": True, "num_chunks": 4},
        )
        self.assertEqual(sequential.chunkwise_forward_mode, "sequential")
        self.assertEqual(sequential.chunkwise_sparse_packing, "interleaved")
        self.assertEqual(sequential.chunkwise_sparse_block_size, 128)
        self.assertEqual(sequential.chunkwise_sparse_alignment, "none")
        self.assertTrue(sequential.chunkwise_packed_capability["supported"])

        for bad_config, message in (
            ({"forward_mode": "bad"}, "forward_mode"),
            ({"sparse_packing": "auto"}, "sparse_packing"),
            ({"sparse_block_size": 0}, "sparse_block_size"),
            ({"sparse_alignment": "auto"}, "sparse_alignment"),
            ({"packed_layout_schema_version": 999}, "packed_layout_schema_version"),
        ):
            config = {"enabled": True, "num_chunks": 4, **bad_config}
            with self.subTest(config=bad_config), self.assertRaisesRegex(ValueError, message):
                ImageWAM(
                    nn.Identity(),
                    nn.Identity(),
                    nn.Identity(),
                    nn.Identity(),
                    text_dim=2,
                    stack="flux2",
                    chunkwise_causal=config,
                )

    def test_packed_flex_fails_fast_without_cuda_capability(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            ImageWAM(
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                text_dim=2,
                stack="flux2",
                chunkwise_causal={
                    "enabled": True,
                    "num_chunks": 4,
                    "forward_mode": "packed_flex",
                },
            )

    def test_k1_forces_sequential_even_with_sparse_settings(self):
        model = ImageWAM(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            text_dim=2,
            stack="flux2",
            chunkwise_causal={
                "enabled": True,
                "num_chunks": 1,
                "forward_mode": "packed_flex",
                "sparse_packing": "batch_padded",
                "sparse_alignment": "segment",
            },
        )
        self.assertEqual(model.chunkwise_forward_mode, "sequential")
        self.assertFalse(model.supports_chunkwise_training_losses)

    def test_input_builder_accepts_endpoint_observations_and_uses_temporal_boundaries(self):
        model = _bare_model()
        model.resolved_chunk_count = 4
        model.resolved_actions_per_chunk = 16
        model.resolved_total_action_horizon = 64
        model.proprio_encoder = nn.Identity()

        def encode(_self, image, *, time_value):
            del time_value
            value = image[:, :1, :1, :1].flatten(1)
            return value[:, None, :].expand(-1, 1, 2).clone(), torch.zeros(image.shape[0], 1, 4)

        model._encode_flux2_image_tokens = types.MethodType(encode, model)
        model._encode_flux2_text = types.MethodType(
            lambda _self, _sample: (torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool)),
            model,
        )
        video = torch.stack([torch.full((3, 16, 16), float(i)) for i in range(5)], dim=1)[None]
        built = model._build_flux2_chunkwise_inputs(
            {
                "video": video,
                "action": torch.zeros(1, 64, 2),
                "action_is_pad": torch.tensor([[False] * 63 + [True]]),
                "action_dim_is_pad": torch.tensor([[False, True]]),
                "image_is_pad": torch.tensor([[False, False, False, True, False]]),
                "proprio": torch.zeros(1, 64, 2),
            }
        )

        self.assertEqual(built["boundary_indices"], (0, 16, 32, 48, 64))
        self.assertEqual(
            [entry["tokens"][0, 0, 0].item() for entry in built["observations"]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [entry["clean_ids"][0, 0, 0].item() for entry in built["observations"]],
            [10, 11, 12, 13, 14],
        )
        self.assertEqual(
            [entry["target_ids"][0, 0, 0].item() for entry in built["observations"]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(built["target_valid"].tolist(), [[True, True, False, True]])
        self.assertEqual(
            built["observation_valid"].tolist(), [[True, True, True, False, True]]
        )
        self.assertEqual(built["video_denominator"].tolist(), [6.0])
        self.assertEqual(built["action_denominator"].tolist(), [63.0])

    def test_input_builder_rejects_dense_video_and_geometry_mismatches(self):
        model = _bare_model()
        model.resolved_chunk_count = 4
        model.resolved_actions_per_chunk = 16
        model.resolved_total_action_horizon = 64
        model.proprio_encoder = nn.Identity()
        model._encode_flux2_image_tokens = types.MethodType(
            lambda _self, image, *, time_value: (
                torch.zeros(image.shape[0], 1, 2),
                torch.zeros(image.shape[0], 1, 4),
            ),
            model,
        )
        model._encode_flux2_text = types.MethodType(
            lambda _self, _sample: (
                torch.zeros(1, 2, 2),
                torch.ones(1, 2, dtype=torch.bool),
            ),
            model,
        )

        valid_sample = {
            "video": torch.zeros(1, 3, 5, 16, 16),
            "action": torch.zeros(1, 64, 2),
            "image_is_pad": torch.zeros(1, 5, dtype=torch.bool),
            "proprio": torch.zeros(1, 64, 2),
        }

        invalid_samples = {
            "dense video": {**valid_sample, "video": torch.zeros(1, 3, 65, 16, 16)},
            "image padding": {
                **valid_sample,
                "image_is_pad": torch.zeros(1, 65, dtype=torch.bool),
            },
            "action horizon": {**valid_sample, "action": torch.zeros(1, 63, 2)},
            "short proprio": {**valid_sample, "proprio": torch.zeros(1, 63, 2)},
            "long proprio": {**valid_sample, "proprio": torch.zeros(1, 65, 2)},
        }
        for name, sample in invalid_samples.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "geometry|image_is_pad"):
                model._build_flux2_chunkwise_inputs(sample)

    def test_state_positions_follow_each_samples_valid_text_length(self):
        model = _bare_model()
        model.resolved_actions_per_chunk = 2
        model.proprio_dim = 1
        model.proprio_encoder = nn.Linear(1, 2, bias=False)
        model.pack_proprio_after_text = True
        inputs = {
            "text_hidden_states": torch.zeros(2, 3, 2),
            "text_attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
            "proprio": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        }

        _context, context_mask, state_positions = model._flux2_chunk_context(inputs, 1)

        self.assertEqual(state_positions.tolist(), [2, 1])
        self.assertEqual(context_mask.sum(dim=1).tolist(), [3, 2])

    def test_prepared_forwards_grow_prefix_and_yield_differentiable_zero_anchor(self):
        model = _bare_model()
        model.stack = "flux2"
        model.chunkwise_causal_enabled = True
        model.resolved_chunk_count = 2
        model.resolved_actions_per_chunk = 1
        model.resolved_total_action_horizon = 2
        model.proprio_encoder = None
        model.supports_chunkwise_prepared_forward = True
        model.loss_lambda_video = 1.0
        model.loss_lambda_action = 1.0
        model.train_video_scheduler = _Scheduler()
        model.train_action_scheduler = _Scheduler()
        model.video_expert = _VideoExpert()
        model.action_expert = _ActionExpert()
        model.mot = _Mot()
        observations = [
            {
                "tokens": torch.full((1, 1, 2), float(index + 1)),
                "clean_ids": torch.tensor([[[10.0 + index, 0, 0, 0]]]),
                "target_ids": torch.tensor([[[float(index), 0, 0, 0]]]),
            }
            for index in range(3)
        ]
        prepared = {
            "observations": observations,
            "text_hidden_states": torch.zeros(1, 2, 2),
            "text_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "proprio": None,
            "action": torch.ones(1, 2, 2),
            "action_is_pad": torch.tensor([[False, True]]),
            "action_dim_is_pad": torch.zeros(1, 2, dtype=torch.bool),
            "target_valid": torch.tensor([[True, False]]),
            "observation_valid": torch.tensor([[True, False, True]]),
            "video_denominator": torch.tensor([2.0]),
            "action_denominator": torch.tensor([2.0]),
        }
        model._build_flux2_chunkwise_inputs = lambda *_args, **_kwargs: prepared

        prepared_inputs = model.prepare_chunkwise_training_inputs({})
        contributions = [
            model(
                prepared_chunkwise_inputs=prepared_inputs,
                chunk_index=chunk_index,
            )
            for chunk_index in range(model.resolved_chunk_count)
        ]

        self.assertEqual(len(contributions), 2)
        self.assertEqual(model.video_expert.prefix_lengths, [1, 2])
        self.assertEqual(model.video_expert.prefix_ordinals, [[10.0], [10.0, 11.0]])
        # Second chunk order is text(2), O0(1), padded O1(1), padded target(1), action(1).
        self.assertFalse(model.mot.masks[1][:, :, 3].any())
        self.assertFalse(model.mot.masks[1][:, :, 4].any())
        self.assertGreater(contributions[0][0].item(), 0.0)
        self.assertEqual(contributions[1][0].item(), 0.0)
        self.assertTrue(contributions[1][0].requires_grad)
        for loss, _metrics in contributions:
            loss.backward()
        self.assertIsNotNone(model.mot.scale.grad)

        with self.assertRaisesRegex(RuntimeError, "bypass distributed wrappers"):
            list(model.iter_training_losses({}))

    def test_k1_iterator_uses_legacy_loss(self):
        model = _bare_model()
        model.stack = "flux2"
        model.chunkwise_causal_enabled = True
        model.resolved_chunk_count = 1
        anchor = nn.Parameter(torch.tensor(2.0))
        model.register_parameter("anchor", anchor)
        model.training_loss = lambda _sample, tiled=False: (model.anchor * 3, {"legacy": 1.0})

        contributions = list(model.iter_training_losses({}))

        self.assertEqual(len(contributions), 1)
        self.assertEqual(contributions[0][0].item(), 6.0)
        self.assertEqual(contributions[0][1], {"legacy": 1.0})

    def test_forward_accepts_prepared_chunkwise_inputs_for_one_chunk(self):
        model = _chunkwise_forward_contract_model()
        prepared = model.prepare_chunkwise_training_inputs({})

        loss, metrics = model(prepared_chunkwise_inputs=prepared, chunk_index=0)

        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(metrics["chunk_count"], 1.0)
        self.assertIn("chunk/0/loss_video", metrics)
        self.assertEqual(len(model.mot.masks), 1)
        loss.backward()
        self.assertIsNotNone(model.mot.scale.grad)

    def test_prepared_forward_api_validates_sequential_and_packed_modes(self):
        sequential = _chunkwise_forward_contract_model()
        prepared = sequential.prepare_chunkwise_training_inputs({})

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            sequential(sample={}, prepared_chunkwise_inputs=prepared, chunk_index=0)
        with self.assertRaisesRegex(ValueError, "required"):
            sequential(prepared_chunkwise_inputs=prepared)
        with self.assertRaisesRegex(ValueError, "requires `prepared_chunkwise_inputs`"):
            sequential(chunk_index=0)

        packed = _chunkwise_forward_contract_model()
        packed.chunkwise_forward_mode = "packed_flex"
        with self.assertRaisesRegex(ValueError, "invalid with packed"):
            packed(prepared_chunkwise_inputs=prepared, chunk_index=0)
        malformed = {**prepared, "target_valid": torch.ones(1, 1, dtype=torch.bool)}
        with self.assertRaisesRegex(ValueError, "target_valid"):
            sequential(prepared_chunkwise_inputs=malformed, chunk_index=0)
        with self.assertRaisesRegex(ValueError, "target_valid"):
            packed(prepared_chunkwise_inputs=malformed)

    def test_prepared_forward_rejects_malformed_packed_tensor_contracts(self):
        model = _chunkwise_forward_contract_model()
        prepared = model.prepare_chunkwise_training_inputs({})

        malformed_ids = {
            **prepared,
            "observations": [dict(observation) for observation in prepared["observations"]],
        }
        malformed_ids["observations"][0]["clean_ids"] = torch.zeros(1, 2, 4)
        with self.assertRaisesRegex(ValueError, "clean_ids.*token length"):
            model._validate_flux2_chunkwise_prepared_inputs(malformed_ids)

        for key in ("action_is_pad", "action_dim_is_pad", "target_valid", "observation_valid"):
            malformed_mask = {**prepared, key: prepared[key].float()}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "bool dtype"):
                model._validate_flux2_chunkwise_prepared_inputs(malformed_mask)

        for key, value in (
            ("video_denominator", torch.tensor([0.0])),
            ("action_denominator", torch.tensor([float("nan")])),
        ):
            malformed_denominator = {**prepared, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "finite positive"):
                model._validate_flux2_chunkwise_prepared_inputs(malformed_denominator)

    def test_prepared_forward_requires_proprio_geometry_when_encoder_is_enabled(self):
        model = _chunkwise_forward_contract_model()
        prepared = model.prepare_chunkwise_training_inputs({})
        model.proprio_encoder = nn.Linear(3, 2)

        with self.assertRaisesRegex(ValueError, "proprio"):
            model._validate_flux2_chunkwise_prepared_inputs(prepared)
        malformed = {**prepared, "proprio": torch.zeros(1, 1, 3)}
        with self.assertRaisesRegex(ValueError, "proprio"):
            model._validate_flux2_chunkwise_prepared_inputs(malformed)

        valid = {**prepared, "proprio": torch.zeros(1, 2, 3)}
        model._validate_flux2_chunkwise_prepared_inputs(valid)

    def test_packed_forward_matches_sequential_and_builds_interleaved_ownership(self):
        def run_model(forward_mode):
            model = _chunkwise_forward_contract_model()
            model.chunkwise_forward_mode = forward_mode
            calls = []
            model.train_video_scheduler = _RecordingScheduler("video", calls)
            model.train_action_scheduler = _RecordingScheduler("action", calls)
            prepared = model.prepare_chunkwise_training_inputs({})
            captured = {}
            noise_calls = []

            def fake_randn_like(tensor):
                noise_calls.append(tuple(tensor.shape))
                return torch.full_like(tensor, float(len(noise_calls)))

            def fake_sparse_mask(*, layout, query_valid, key_valid, state_positions, device, num_heads):
                del device, num_heads
                captured["layout"] = layout
                captured["query_valid"] = query_valid.detach().clone()
                captured["key_valid"] = key_valid.detach().clone()
                captured["state_positions"] = (
                    None if state_positions is None else state_positions.detach().clone()
                )
                return PackedBlockSparseMask(
                    layout=layout,
                    block_mask=None,
                    query_valid=query_valid,
                    key_valid=key_valid,
                    state_positions=state_positions,
                )

            if forward_mode == "packed_flex":
                with mock.patch("imagewam.models.backbones.imagewam.torch.randn_like", side_effect=fake_randn_like), mock.patch(
                    "imagewam.models.backbones.imagewam.build_packed_block_sparse_mask", side_effect=fake_sparse_mask
                ) as sparse_builder:
                    with mock.patch.object(
                        model.mot,
                        "forward_flux2_interleaved",
                        wraps=model.mot.forward_flux2_interleaved,
                    ) as mot_forward:
                        loss, metrics = model(prepared_chunkwise_inputs=prepared)
                self.assertEqual(sparse_builder.call_count, 1)
                self.assertEqual(mot_forward.call_count, 1)
            else:
                loss = None
                metrics = {}
                with mock.patch("imagewam.models.backbones.imagewam.torch.randn_like", side_effect=fake_randn_like):
                    for chunk_index in range(model.resolved_chunk_count):
                        chunk_loss, chunk_metrics = model(
                            prepared_chunkwise_inputs=prepared,
                            chunk_index=chunk_index,
                        )
                        loss = chunk_loss if loss is None else loss + chunk_loss
                        for key, value in chunk_metrics.items():
                            metrics[key] = metrics.get(key, 0.0) + value
                captured = None

            loss.backward()
            return (
                loss.detach(),
                metrics,
                model.mot.scale.grad.detach().clone(),
                calls,
                noise_calls,
                list(model.video_expert.prefix_lengths),
                list(model.video_expert.prefix_ordinals),
                captured,
            )

        (
            sequential_loss,
            sequential_metrics,
            sequential_grad,
            sequential_calls,
            sequential_noise_calls,
            sequential_prefix_lengths,
            sequential_prefix_ordinals,
            _,
        ) = run_model("sequential")
        (
            packed_loss,
            packed_metrics,
            packed_grad,
            packed_calls,
            packed_noise_calls,
            packed_prefix_lengths,
            packed_prefix_ordinals,
            captured,
        ) = run_model("packed_flex")

        expected_calls = [
            "video:sample:1",
            "video:add:1",
            "video:target:1",
            "action:sample:1",
            "action:add:1",
            "action:target:1",
            "video:sample:2",
            "video:add:2",
            "video:target:2",
            "action:sample:2",
            "action:add:2",
            "action:target:2",
        ]
        self.assertEqual(sequential_calls, expected_calls)
        self.assertEqual(packed_calls, expected_calls)
        expected_noise_calls = [(1, 1, 2), (1, 1, 2), (1, 1, 2), (1, 1, 2)]
        self.assertEqual(sequential_noise_calls, expected_noise_calls)
        self.assertEqual(packed_noise_calls, expected_noise_calls)
        self.assertEqual(sequential_prefix_lengths, [1, 2])
        self.assertEqual(packed_prefix_lengths, [1, 2])
        self.assertEqual(sequential_prefix_ordinals, [[10.0], [10.0, 11.0]])
        self.assertEqual(packed_prefix_ordinals, [[10.0], [10.0, 11.0]])
        self.assertTrue(torch.allclose(packed_loss, sequential_loss, atol=1e-6))
        self.assertTrue(torch.allclose(packed_grad, sequential_grad, atol=1e-6))
        self.assertAlmostEqual(float(packed_metrics["loss_video"]), float(sequential_metrics["loss_video"]), places=6)
        self.assertAlmostEqual(float(packed_metrics["loss_action"]), float(sequential_metrics["loss_action"]), places=6)
        self.assertEqual(packed_metrics["chunk_count"], 2.0)
        for key in (
            "loss_video",
            "loss_action",
            "chunk_count",
            "packed/attention_sequence_length",
            "chunk/0/loss_video",
            "chunk/0/loss_action",
            "chunk/0/observation_prefix_tokens",
            "chunk/1/loss_video",
            "chunk/1/loss_action",
            "chunk/1/observation_prefix_tokens",
        ):
            self.assertIn(key, packed_metrics)
        per_chunk_total = (
            packed_metrics["chunk/0/loss_video"]
            + packed_metrics["chunk/0/loss_action"]
            + packed_metrics["chunk/1/loss_video"]
            + packed_metrics["chunk/1/loss_action"]
        )
        self.assertTrue(torch.allclose(packed_loss, per_chunk_total))
        self.assertTrue(packed_metrics["chunk/1/loss_video"].requires_grad)
        self.assertEqual(float(packed_metrics["chunk/1/loss_video"]), 0.0)

        self.assertIsNotNone(captured)
        layout = captured["layout"]
        self.assertEqual(layout.sparse_packing, "interleaved")
        self.assertEqual(layout.total_token_count, 11)
        self.assertEqual(
            captured["query_valid"].tolist(),
            [
                [True, True, True, True, True, False, False, False, True, True, True],
            ],
        )
        self.assertEqual(
            captured["key_valid"].tolist(),
            [
                [True, True, True, True, True, False, False, False, False, False, False],
            ],
        )

    def test_ddp_forward_synchronizes_chunkwise_gradients_across_cpu_ranks(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is unavailable")
        if not dist.is_gloo_available():
            self.skipTest("torch.distributed gloo backend is unavailable")

        world_size = 2
        context = mp.get_context("spawn")
        queue = context.Queue()
        port = _free_tcp_port()
        processes = [
            context.Process(
                target=_ddp_chunkwise_forward_worker,
                args=(rank, world_size, port, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        results = [queue.get(timeout=30) for _ in range(world_size)]
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)

        errors = [result for result in results if "error" in result]
        self.assertEqual(errors, [])
        self.assertEqual({result["rank"] for result in results}, {0, 1})
        ordered = sorted(results, key=lambda result: result["rank"])
        self.assertEqual([result["masks"] for result in ordered], [2, 2])
        self.assertEqual(
            [result["chunk_count"] for result in ordered],
            [2.0, 2.0],
        )
        grads = [result["grad"] for result in ordered]
        self.assertNotEqual(results[0]["loss"], results[1]["loss"])
        self.assertAlmostEqual(grads[0], grads[1], places=6)

    def test_accelerate_forward_synchronizes_chunkwise_gradients_across_cpu_ranks(self):
        try:
            import accelerate  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("accelerate is unavailable")
        if not dist.is_available():
            self.skipTest("torch.distributed is unavailable")
        if not dist.is_gloo_available():
            self.skipTest("torch.distributed gloo backend is unavailable")

        world_size = 2
        context = mp.get_context("spawn")
        queue = context.Queue()
        port = _free_tcp_port()
        processes = [
            context.Process(
                target=_accelerate_chunkwise_forward_worker,
                args=(rank, world_size, port, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        results = [queue.get(timeout=30) for _ in range(world_size)]
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)

        errors = [result for result in results if "error" in result]
        self.assertEqual(errors, [])
        self.assertEqual({result["rank"] for result in results}, {0, 1})
        ordered = sorted(results, key=lambda result: result["rank"])
        self.assertEqual([result["masks"] for result in ordered], [2, 2])
        self.assertEqual(
            [result["chunk_count"] for result in ordered],
            [2.0, 2.0],
        )
        self.assertTrue(
            all("MULTI_CPU" in result["distributed_type"] for result in ordered)
        )
        grads = [result["grad"] for result in ordered]
        self.assertNotEqual(results[0]["loss"], results[1]["loss"])
        self.assertAlmostEqual(grads[0], grads[1], places=6)


if __name__ == "__main__":
    unittest.main()
