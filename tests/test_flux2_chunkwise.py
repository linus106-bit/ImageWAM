import unittest
import torch

from imagewam.chunkwise import (
    build_chunkwise_causal_mask,
    chunkwise_loss_contribution,
    resolve_chunkwise_geometry,
)
from imagewam.models.backbones.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


class Flux2ChunkwiseTest(unittest.TestCase):
    def test_default_geometry_has_five_observation_anchors_and_64_actions(self):
        geometry = resolve_chunkwise_geometry(4, 16, num_frames=None)

        self.assertEqual(geometry.num_frames, 65)
        self.assertEqual(geometry.total_action_horizon, 64)
        self.assertEqual(geometry.observation_indices, (0, 16, 32, 48, 64))


    def test_k1_geometry_preserves_endpoint_pair_contract(self):
        geometry = resolve_chunkwise_geometry(1, 16, num_frames=17)

        self.assertEqual(geometry.num_frames, 17)
        self.assertEqual(geometry.observation_indices, (0, 16))


    def test_invalid_geometry_fails_fast(self):
        for num_chunks, actions_per_chunk, num_frames in (
            (0, 16, None),
            (4, 0, None),
            (4, 16, 17),
        ):
            with self.subTest(
                num_chunks=num_chunks,
                actions_per_chunk=actions_per_chunk,
                num_frames=num_frames,
            ):
                with self.assertRaises(ValueError):
                    resolve_chunkwise_geometry(num_chunks, actions_per_chunk, num_frames=num_frames)


    def test_chunkwise_mask_has_no_future_leakage_and_state_is_self_only(self):
        masks = build_chunkwise_causal_mask(
            text_attention_mask=torch.tensor([[True, True, True, False]]),
            state_positions=torch.tensor([2]),
            observation_token_lengths=(2, 2),
            clean_observation_valid=torch.tensor([[True, True]]),
            target_length=2,
            target_valid=torch.tensor([True]),
            action_padding_mask=torch.tensor([[False, True]]),
        )
        mask = masks["double_joint"][0]
        # Physical order: text(4), O0(2), O1(2), target(2), action(2).
        o0 = slice(4, 6)
        o1 = slice(6, 8)
        target = slice(8, 10)
        action = slice(10, 12)

        self.assertTrue(mask[o0, o0].all())
        self.assertFalse(mask[o0, o1].any())
        self.assertTrue(mask[o1, o0].all() and mask[o1, o1].all())
        self.assertTrue(mask[target, o0].all() and mask[target, o1].all())
        self.assertTrue(mask[action, target].all())
        self.assertEqual(mask[2].sum().item(), 1)
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[:, 3].any())  # padded text key
        self.assertFalse(mask[:, 11].any())  # padded action key
        self.assertTrue(torch.equal(masks["double_joint"], masks["single"]))
        self.assertNotEqual(masks["double_joint"].data_ptr(), masks["single"].data_ptr())

    def test_chunkwise_mask_clears_padded_clean_and_target_image_keys_per_example(self):
        mask = build_chunkwise_causal_mask(
            text_attention_mask=torch.ones(2, 1, dtype=torch.bool),
            observation_token_lengths=(2, 1),
            clean_observation_valid=torch.tensor([[True, False], [False, True]]),
            target_length=2,
            target_valid=torch.tensor([False, True]),
            action_padding_mask=torch.zeros(2, 1, dtype=torch.bool),
        )["double_joint"]

        # Physical order: text(1), O0(2), O1(1), target(2), action(1).
        self.assertFalse(mask[0, :, 3].any())
        self.assertFalse(mask[0, :, 4:6].any())
        self.assertTrue(mask[0, :, 1:3].any())
        self.assertFalse(mask[1, :, 1:3].any())
        self.assertTrue(mask[1, :, 3].any())
        self.assertTrue(mask[1, :, 4:6].any())


    def test_full_window_denominators_make_chunk_contributions_additive(self):
        video_error = torch.tensor([[4.0, 4.0], [9.0, 9.0]])
        action_error = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
        video_denominator = torch.tensor([4.0, 2.0])
        action_denominator = torch.tensor([4.0, 2.0])
        valid_target = torch.tensor([True, False])

        total, metrics = chunkwise_loss_contribution(
            video_squared_error=video_error,
            action_squared_error=action_error,
            valid_target=valid_target,
            video_weight=torch.ones(2),
            action_weight=torch.ones(2),
            video_denominator=video_denominator,
            action_denominator=action_denominator,
            lambda_video=0.5,
            lambda_action=1.0,
        )

        # batch 0: .5*(8/4) + 2/4 = 1.5; batch 1: 0 + 8/2 = 4
        self.assertAlmostEqual(total.item(), 2.75)
        self.assertAlmostEqual(metrics["loss_video"].item(), 0.5)
        self.assertAlmostEqual(metrics["loss_action"].item(), 2.25)

    def test_production_scheduler_scalar_weights_work_for_batch_one(self):
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=8, shift=2.0)
        parameter = torch.nn.Parameter(torch.tensor([[2.0, 3.0]]))
        scalar_weight = scheduler.training_weight(torch.tensor([4.0]))
        self.assertEqual(scalar_weight.ndim, 0)

        total, _ = chunkwise_loss_contribution(
            video_squared_error=parameter.square(),
            action_squared_error=parameter.square(),
            valid_target=torch.tensor([True]),
            video_weight=scalar_weight,
            action_weight=scalar_weight,
            video_denominator=torch.tensor([2.0]),
            action_denominator=torch.tensor([2.0]),
            lambda_video=1.0,
            lambda_action=1.0,
        )
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(parameter.grad)

    def test_chunkwise_loss_rejects_non_vector_scheduler_weights(self):
        common = {
            "video_squared_error": torch.ones(2, 1),
            "action_squared_error": torch.ones(2, 1),
            "valid_target": torch.ones(2, dtype=torch.bool),
            "video_denominator": torch.ones(2),
            "action_denominator": torch.ones(2),
            "lambda_video": 1.0,
            "lambda_action": 1.0,
        }
        for bad_weight in (torch.ones(2, 1), torch.ones(3)):
            with self.subTest(shape=tuple(bad_weight.shape)), self.assertRaisesRegex(
                ValueError, "scalar or \\[B\\]"
            ):
                chunkwise_loss_contribution(
                    **common,
                    video_weight=bad_weight,
                    action_weight=torch.ones(2),
                )


if __name__ == "__main__":
    unittest.main()
