import unittest
from unittest import mock

import torch

from imagewam.chunkwise import (
    PACKED_CHUNK_LAYOUT_SCHEMA_VERSION,
    PackedBlockSparseMask,
    build_chunkwise_causal_mask,
    build_packed_block_sparse_mask,
    build_packed_chunk_layout,
    cache_packed_block_topology,
    chunkwise_loss_contribution,
    clear_packed_topology_cache,
    packed_flex_attention,
    packed_topology_cache_size,
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
        self.assertTrue(mask[target, target].all())
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[target, action].any())
        self.assertFalse(mask[action, target].any())
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



    def test_packed_layout_accounts_for_segments_and_alignment(self):
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "chunk_ordinal": 0,
                    "text_length": 3,
                    "observation_token_lengths": (2,),
                    "target_length": 1,
                    "action_length": 2,
                },
                {
                    "chunk_ordinal": 1,
                    "text_length": 3,
                    "observation_token_lengths": (2, 2),
                    "target_length": 1,
                    "action_length": 2,
                },
            ),
            sparse_block_size=8,
            sparse_alignment="segment",
        )

        self.assertEqual(layout.schema_version, PACKED_CHUNK_LAYOUT_SCHEMA_VERSION)
        self.assertEqual(layout.sparse_packing, "interleaved")
        self.assertEqual(layout.sparse_alignment, "segment")
        self.assertEqual(layout.total_token_count, 24)
        self.assertEqual(layout.segments[0].local_start, 0)
        self.assertEqual(layout.segments[0].physical_end, 8)
        self.assertIsNone(layout.segments[0].padding_range)
        self.assertEqual(layout.segments[1].local_start, 8)
        self.assertEqual(layout.segments[1].physical_end, 18)
        self.assertEqual(layout.segments[1].padding_range, (18, 24))
        self.assertEqual(layout.global_to_local(8), (1, 0, "text"))
        self.assertEqual(layout.local_to_global(1, 4), 12)
        self.assertEqual(layout.pattern_owner.count(0), 8)
        self.assertEqual(layout.pattern_owner.count(1), 16)
        self.assertIn("segment", layout.signature)

    def test_packed_layout_dense_reference_blocks_cross_pattern_edges(self):
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_packing="batch_padded",
            sparse_block_size=4,
        )
        from imagewam.chunkwise import PackedBlockSparseMask

        sparse_mask = PackedBlockSparseMask(layout=layout, block_mask=None)
        dense = sparse_mask.to_dense_reference(
            text_attention_masks=(torch.ones(1, 2, dtype=torch.bool),) * 2,
            clean_observation_valid=(torch.ones(1, 1, dtype=torch.bool),) * 2,
            target_valid=(torch.ones(1, dtype=torch.bool),) * 2,
            action_padding_masks=(torch.zeros(1, 1, dtype=torch.bool),) * 2,
        )[0]

        first = slice(layout.segments[0].local_start, layout.segments[0].physical_end)
        second = slice(layout.segments[1].local_start, layout.segments[1].physical_end)
        self.assertFalse(dense[first, second].any())
        self.assertFalse(dense[second, first].any())
        self.assertTrue(dense[first, first].any())
        self.assertTrue(dense[second, second].any())

    def test_interleaved_topology_matches_dreamzero_cross_chunk_edges(self):
        clear_packed_topology_cache()
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "chunk_ordinal": 0,
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
                {
                    "chunk_ordinal": 1,
                    "text_length": 2,
                    "observation_token_lengths": (1, 1),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_packing="interleaved",
            sparse_block_size=4,
        )
        calls = []
        build_packed_block_sparse_mask(
            layout=layout,
            query_valid=torch.tensor([[True, True, True, True, True, False, False, False, True, True, True]]),
            key_valid=torch.tensor([[True, True, True, True, True, False, False, False, True, True, True]]),
            device="cpu",
            create_block_mask_fn=lambda mask_mod, **_kwargs: calls.append(mask_mod),
        )
        mask_mod = calls[0]

        def allowed(query_index, key_index):
            return bool(
                mask_mod(
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(query_index),
                    torch.tensor(key_index),
                )
            )

        # Segment 1 target/action read canonical instruction and O0..O1.
        for query_index in (9, 10):
            self.assertTrue(allowed(query_index, 0))
            self.assertTrue(allowed(query_index, 2))
            self.assertTrue(allowed(query_index, 8))
            self.assertFalse(allowed(query_index, 3))
            self.assertFalse(allowed(query_index, 4))
        self.assertTrue(allowed(9, 9))
        self.assertFalse(allowed(9, 10))
        self.assertTrue(allowed(10, 10))
        self.assertFalse(allowed(10, 9))

        # Clean O1 reads earlier clean history, but O0 cannot see future O1.
        self.assertTrue(allowed(8, 0))
        self.assertTrue(allowed(8, 2))
        self.assertTrue(allowed(8, 8))
        self.assertFalse(allowed(8, 9))
        self.assertFalse(allowed(3, 8))
        clear_packed_topology_cache()

    def test_interleaved_state_positions_are_chunk_local_and_self_only(self):
        clear_packed_topology_cache()
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
                {
                    "text_length": 2,
                    "observation_token_lengths": (1, 1),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_packing="interleaved",
            sparse_block_size=4,
        )
        sparse = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=torch.tensor([[True, True, True, True, True, False, False, False, True, True, True]]),
            key_valid=torch.tensor([[True, True, True, True, True, False, False, False, True, True, True]]),
            state_positions=torch.tensor([[1, 6]]),
            device="cpu",
            create_block_mask_fn=lambda *_args, **_kwargs: object(),
        )
        captured = {}

        def fake_flex(query, _key, _value, *, score_mod, **_kwargs):
            captured["score_mod"] = score_mod
            return torch.ones_like(query)

        query = torch.randn(1, 1, layout.total_token_count, 2)
        packed_flex_attention(query=query, key=query, value=query, mask=sparse, flex_attention_fn=fake_flex)
        score_mod = captured["score_mod"]

        def modified(query_index, key_index):
            return score_mod(
                torch.tensor(1.0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(query_index),
                torch.tensor(key_index),
            )

        self.assertTrue(torch.isfinite(modified(1, 1)))
        self.assertTrue(torch.isneginf(modified(1, 0)))
        self.assertTrue(torch.isfinite(modified(3, 1)))
        self.assertTrue(torch.isfinite(modified(6, 6)))
        self.assertTrue(torch.isneginf(modified(6, 0)))
        self.assertTrue(torch.isfinite(modified(9, 6)))
        self.assertTrue(torch.isneginf(modified(9, 1)))
        self.assertTrue(torch.isneginf(modified(8, 6)))
        clear_packed_topology_cache()

    def test_packed_topology_cache_is_lru_and_excludes_dynamic_values(self):
        clear_packed_topology_cache()
        key = ("cpu", "interleaved", "none", 128, PACKED_CHUNK_LAYOUT_SCHEMA_VERSION, (1, 2, 3))
        first = object()
        self.assertIs(cache_packed_block_topology(key, first), first)
        self.assertIs(cache_packed_block_topology(key, object()), first)
        for index in range(20):
            cache_packed_block_topology(("cpu", index), object())
        self.assertLessEqual(packed_topology_cache_size(), 16)
        clear_packed_topology_cache()


    def test_packed_block_sparse_mask_uses_create_block_mask_without_dense_allocation(self):
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 3,
                    "observation_token_lengths": (2,),
                    "target_length": 2,
                    "action_length": 1,
                },
            ),
            sparse_block_size=4,
        )
        calls = []

        def fake_create(mask_mod, *, B, H, Q_LEN, KV_LEN, device, BLOCK_SIZE):
            calls.append(
                {
                    "mask_mod": mask_mod,
                    "B": B,
                    "H": H,
                    "Q_LEN": Q_LEN,
                    "KV_LEN": KV_LEN,
                    "device": device,
                    "BLOCK_SIZE": BLOCK_SIZE,
                }
            )
            return {"block_mask": len(calls)}

        with mock.patch.object(PackedBlockSparseMask, "to_dense_reference", side_effect=AssertionError("dense")), mock.patch(
            "torch.zeros", side_effect=AssertionError("dense allocation")
        ):
            sparse = build_packed_block_sparse_mask(
                layout=layout,
                query_valid=torch.ones(2, layout.total_token_count, dtype=torch.bool),
                key_valid=torch.ones(2, layout.total_token_count, dtype=torch.bool),
                state_positions=torch.tensor([0, 1]),
                device="cpu",
                num_heads=3,
                create_block_mask_fn=fake_create,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["B"], None)
        self.assertEqual(calls[0]["H"], 3)
        self.assertEqual(calls[0]["Q_LEN"], layout.total_token_count)
        self.assertEqual(calls[0]["KV_LEN"], layout.total_token_count)
        self.assertEqual(calls[0]["BLOCK_SIZE"], 4)
        self.assertEqual(sparse.block_mask, {"block_mask": 1})
        # Structural predicate keeps target-image noise and action noise isolated.
        mask_mod = calls[0]["mask_mod"]
        self.assertTrue(bool(mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(5), torch.tensor(6))))
        self.assertTrue(bool(mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(7), torch.tensor(7))))
        self.assertFalse(bool(mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(6), torch.tensor(7))))
        self.assertFalse(bool(mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(7), torch.tensor(6))))

        aligned_layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_block_size=4,
            sparse_alignment="segment",
        )
        aligned_calls = []
        build_packed_block_sparse_mask(
            layout=aligned_layout,
            query_valid=torch.tensor([[True, True, True, True, True, False, False, False]]),
            key_valid=torch.tensor([[True, True, True, True, True, False, False, False]]),
            device="cpu",
            create_block_mask_fn=lambda mask_mod, **_kwargs: aligned_calls.append(mask_mod),
        )
        padding_index = aligned_layout.segments[0].padding_range[0]
        self.assertTrue(
            bool(
                aligned_calls[0](
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(padding_index),
                    torch.tensor(padding_index),
                )
            )
        )

    def test_packed_sparse_mask_snapshots_dynamic_tensors_and_reuses_structural_topology(self):
        clear_packed_topology_cache()
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_block_size=4,
        )
        create_calls = []

        def fake_create(*args, **kwargs):
            create_calls.append((args, kwargs))
            return object()

        query_valid = torch.ones(1, layout.total_token_count, dtype=torch.bool)
        key_valid = torch.ones(1, layout.total_token_count, dtype=torch.bool)
        first = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=query_valid,
            key_valid=key_valid,
            state_positions=torch.tensor([0]),
            device="cpu",
            create_block_mask_fn=fake_create,
        )
        query_valid[:, 4] = False
        key_valid[:, 2] = False
        second = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=query_valid,
            key_valid=key_valid,
            state_positions=torch.tensor([1]),
            device="cpu",
            create_block_mask_fn=fake_create,
        )

        self.assertIs(first.block_mask, second.block_mask)
        self.assertEqual(len(create_calls), 1)
        self.assertEqual(first.signature, second.signature)
        self.assertNotEqual(first.query_valid.data_ptr(), query_valid.data_ptr())
        self.assertTrue(first.query_valid[0, 4])
        self.assertFalse(second.query_valid[0, 4])
        self.assertTrue(first.key_valid[0, 2])
        self.assertFalse(second.key_valid[0, 2])
        self.assertEqual(first.state_positions.tolist(), [0])
        self.assertEqual(second.state_positions.tolist(), [1])
        self.assertFalse(first.key_valid[0, 0])
        self.assertFalse(second.key_valid[0, 1])
        self.assertTrue(first.query_valid[0, 0])
        self.assertTrue(second.query_valid[0, 1])
        clear_packed_topology_cache()

    def test_packed_flex_state_edges_match_causal_contract_and_invalid_queries_zero(self):
        clear_packed_topology_cache()
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_block_size=4,
        )
        sparse = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=torch.tensor([[True, False, True, True, True]]),
            key_valid=torch.ones(1, layout.total_token_count, dtype=torch.bool),
            state_positions=torch.tensor([0]),
            device="cpu",
            create_block_mask_fn=lambda *_args, **_kwargs: object(),
        )
        captured = {}

        def fake_flex(query, _key, _value, *, score_mod, **_kwargs):
            captured["score_mod"] = score_mod
            return torch.ones_like(query)

        query = torch.randn(1, 1, layout.total_token_count, 2)
        output = packed_flex_attention(
            query=query,
            key=query,
            value=query,
            mask=sparse,
            flex_attention_fn=fake_flex,
        )
        score_mod = captured["score_mod"]

        def modified(query_index, key_index):
            return score_mod(
                torch.tensor(1.0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(query_index),
                torch.tensor(key_index),
            )

        self.assertTrue(torch.isfinite(modified(0, 0)))  # state query self-edge
        self.assertTrue(torch.isneginf(modified(0, 1)))
        self.assertTrue(torch.isneginf(modified(2, 0)))  # observation cannot read state
        self.assertTrue(torch.isfinite(modified(3, 0)))  # current target can read state
        self.assertTrue(torch.isfinite(modified(4, 0)))  # current action can read state
        self.assertTrue(torch.isfinite(modified(1, 1)))  # invalid-query dummy edge
        self.assertTrue(torch.equal(output[:, :, 1], torch.zeros_like(output[:, :, 1])))
        self.assertTrue(torch.equal(output[:, :, 0], torch.ones_like(output[:, :, 0])))
        clear_packed_topology_cache()

    def test_packed_topology_cache_separates_head_count(self):
        clear_packed_topology_cache()
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_block_size=4,
        )
        create_calls = []

        def fake_create(*_args, **kwargs):
            create_calls.append(kwargs["H"])
            return {"heads": kwargs["H"]}

        query_valid = torch.ones(1, layout.total_token_count, dtype=torch.bool)
        key_valid = torch.ones(1, layout.total_token_count, dtype=torch.bool)
        one_head = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=query_valid,
            key_valid=key_valid,
            device="cpu",
            num_heads=1,
            create_block_mask_fn=fake_create,
        )
        two_heads = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=query_valid,
            key_valid=key_valid,
            device="cpu",
            num_heads=2,
            create_block_mask_fn=fake_create,
        )

        self.assertEqual(create_calls, [1, 2])
        self.assertEqual(one_head.block_mask, {"heads": 1})
        self.assertEqual(two_heads.block_mask, {"heads": 2})
        clear_packed_topology_cache()

    def test_packed_topology_cache_moves_touched_entries_before_eviction(self):
        clear_packed_topology_cache()
        retained = object()
        cache_packed_block_topology(("cpu", 0), retained)
        for index in range(1, 16):
            cache_packed_block_topology(("cpu", index), object())
        self.assertIs(cache_packed_block_topology(("cpu", 0), object()), retained)
        cache_packed_block_topology(("cpu", 16), object())

        self.assertIs(cache_packed_block_topology(("cpu", 0), object()), retained)
        self.assertEqual(packed_topology_cache_size(), 16)
        # Key 1 was least recently used and should have been evicted, so a new object is stored.
        replacement = object()
        self.assertIs(cache_packed_block_topology(("cpu", 1), replacement), replacement)
        clear_packed_topology_cache()

    def test_dense_reference_matches_sequential_masks_for_partial_segment_blocks(self):
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 3,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
                {
                    "text_length": 3,
                    "observation_token_lengths": (1, 2),
                    "target_length": 2,
                    "action_length": 1,
                },
            ),
            sparse_packing="batch_padded",
            sparse_block_size=4,
            sparse_alignment="segment",
        )
        sparse = PackedBlockSparseMask(layout=layout, block_mask=None)
        text_masks = (
            torch.tensor([[True, False, True]]),
            torch.tensor([[True, True, False]]),
        )
        clean_valid = (torch.tensor([[True]]), torch.tensor([[True, False]]))
        target_valid = (torch.tensor([True]), torch.tensor([False]))
        action_pad = (torch.tensor([[False]]), torch.tensor([[True]]))
        states = (torch.tensor([0]), torch.tensor([1]))

        dense = sparse.to_dense_reference(
            text_attention_masks=text_masks,
            clean_observation_valid=clean_valid,
            target_valid=target_valid,
            action_padding_masks=action_pad,
            state_positions=states,
        )

        for index, segment in enumerate(layout.segments):
            expected = build_chunkwise_causal_mask(
                text_attention_mask=text_masks[index],
                observation_token_lengths=segment.clean_observation_lengths,
                clean_observation_valid=clean_valid[index],
                target_length=segment.target_length,
                target_valid=target_valid[index],
                action_padding_mask=action_pad[index],
                state_positions=states[index],
            )["double_joint"]
            actual = dense[:, segment.local_start : segment.physical_end, segment.local_start : segment.physical_end]
            self.assertTrue(torch.equal(actual, expected))
            if segment.padding_range is not None:
                pad_start, pad_end = segment.padding_range
                self.assertFalse(dense[:, pad_start:pad_end, :].any())
                self.assertFalse(dense[:, :, pad_start:pad_end].any())



    def test_mot_mixed_attention_dispatches_packed_sparse_mask_to_flex_adapter(self):
        from imagewam.models.backbones import mot as mot_module

        clear_packed_topology_cache()
        layout = build_packed_chunk_layout(
            chunks=(
                {
                    "text_length": 2,
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_block_size=4,
        )
        sparse = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=torch.ones(1, layout.total_token_count, dtype=torch.bool),
            key_valid=torch.ones(1, layout.total_token_count, dtype=torch.bool),
            device="cpu",
            create_block_mask_fn=lambda *args, **kwargs: {"structural": True},
        )
        module = mot_module.MoT.__new__(mot_module.MoT)
        torch.nn.Module.__init__(module)
        module.num_heads = 2
        module.num_kv_heads = 2
        module.attn_head_dim = 3
        module.gqa_implementation = "repeat"
        module.mot_checkpoint_mixed_attn = False
        module.train(False)
        qkv = torch.randn(1, layout.total_token_count, 6)
        calls = []

        def fake_packed_flex_attention(**kwargs):
            calls.append(kwargs)
            query = kwargs["query"]
            return torch.zeros_like(query)

        with mock.patch.object(mot_module, "packed_flex_attention", side_effect=fake_packed_flex_attention):
            out = module._mixed_attention(qkv, qkv, qkv, sparse)
            with self.assertRaisesRegex(RuntimeError, "attention-probability capture"):
                module._mixed_attention(qkv, qkv, qkv, sparse, return_attn_probs=True)

        self.assertEqual(tuple(out.shape), tuple(qkv.shape))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["mask"], sparse)
        self.assertEqual(tuple(calls[0]["query"].shape), (1, 2, layout.total_token_count, 3))
        self.assertFalse(calls[0]["enable_gqa"])
        clear_packed_topology_cache()


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
