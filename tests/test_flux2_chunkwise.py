import unittest
import torch

from imagewam.chunkwise import (
    build_chunkwise_causal_mask,
    chunkwise_loss_contribution,
    resolve_chunkwise_geometry,
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
            target_length=2,
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


if __name__ == "__main__":
    unittest.main()
