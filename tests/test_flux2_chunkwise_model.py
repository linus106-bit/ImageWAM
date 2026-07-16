import types
import unittest

import torch
import torch.nn as nn

from imagewam.models.backbones.imagewam import ImageWAM


def _bare_model() -> ImageWAM:
    model = ImageWAM.__new__(ImageWAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
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
        return {
            "tokens": {"txt": context, "img": torch.cat([ref_image_hidden_states, x], dim=1)},
            "freqs": {},
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
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.masks = []

    def forward(self, embeds_all, attention_mask, **_kwargs):
        self.masks.append(attention_mask["double_joint"].detach().clone())
        return {
            "video": {
                "txt": embeds_all["video"]["txt"] * self.scale,
                "img": embeds_all["video"]["img"] * self.scale,
            },
            "action": embeds_all["action"] * self.scale,
        }


class Flux2ChunkwiseModelTest(unittest.TestCase):
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

        self.assertEqual(disabled.resolved_chunk_count, 1)
        self.assertFalse(disabled.supports_chunkwise_training_losses)
        self.assertTrue(enabled.supports_chunkwise_training_losses)
        self.assertFalse(explicit_k1.supports_chunkwise_training_losses)

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

    def test_iterator_grows_prefix_and_yields_differentiable_zero_anchor(self):
        model = _bare_model()
        model.stack = "flux2"
        model.chunkwise_causal_enabled = True
        model.resolved_chunk_count = 2
        model.resolved_actions_per_chunk = 1
        model.resolved_total_action_horizon = 2
        model.proprio_encoder = None
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

        contributions = list(model.iter_training_losses({}))

        self.assertEqual(len(contributions), 2)
        self.assertEqual(model.video_expert.prefix_lengths, [1, 2])
        self.assertEqual(model.video_expert.prefix_ordinals, [[10.0], [10.0, 11.0]])
        self.assertGreater(contributions[0][0].item(), 0.0)
        self.assertEqual(contributions[1][0].item(), 0.0)
        self.assertTrue(contributions[1][0].requires_grad)
        for loss, _metrics in contributions:
            loss.backward()
        self.assertIsNotNone(model.mot.scale.grad)

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


if __name__ == "__main__":
    unittest.main()
