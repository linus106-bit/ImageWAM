import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from imagewam.chunkwise import build_packed_block_sparse_mask, build_packed_chunk_layout, clear_packed_topology_cache
from imagewam.models.backbones import mot as mot_module


if "flux2.model" not in sys.modules:
    flux2_pkg = types.ModuleType("flux2")
    flux2_model = types.ModuleType("flux2.model")

    def apply_rope(q, k, _pe):
        return q, k

    flux2_model.apply_rope = apply_rope
    sys.modules.setdefault("flux2", flux2_pkg)
    sys.modules["flux2.model"] = flux2_model


def _sparse_mask(seq_len=5, batch_size=1):
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
        sparse_packing="batch_padded",
        sparse_block_size=4,
    )
    assert layout.total_token_count == seq_len
    return build_packed_block_sparse_mask(
        layout=layout,
        query_valid=torch.ones(batch_size, seq_len, dtype=torch.bool),
        key_valid=torch.ones(batch_size, seq_len, dtype=torch.bool),
        device="cpu",
        create_block_mask_fn=lambda *args, **kwargs: {"structural": True},
    )


class _VideoDoubleBlock(nn.Module):
    def _prepare_qkv(self, img, txt, _img_pe, _txt_pe, _mod_img, _mod_txt):
        x = torch.cat([txt, img], dim=1)
        batch, seq_len, channels = x.shape
        q = x.view(batch, seq_len, 1, channels).transpose(1, 2)
        return q, q, q, None, int(txt.shape[1]), None

    @staticmethod
    def _apply_residuals(img, txt, img_attn, txt_attn, _mods):
        return img + img_attn, txt + txt_attn


class _VideoSingleBlock(nn.Module):
    def _qkv(self, x, _mod):
        batch, seq_len, channels = x.shape
        q = x.view(batch, seq_len, 1, channels).transpose(1, 2)
        return q, q, q, torch.zeros_like(x), torch.ones(batch, 1, channels, device=x.device, dtype=x.dtype)

    @staticmethod
    def _out(residual, attn, _mlp, _gate):
        return residual + attn


class _ActionBlock(nn.Module):
    @staticmethod
    def prepare_qkv(action, _action_pe, _mod):
        return {"q": action, "k": action, "v": action, "residual": action}

    @staticmethod
    def apply_post(attn, state):
        return state["residual"] + attn


class _Transformer(nn.Module):
    @staticmethod
    def pe_embedder(ids):
        return torch.zeros(ids.shape[0], 1, ids.shape[1], 1, device=ids.device, dtype=ids.dtype)


class _FakeVideoExpert(nn.Module):
    block_protocol = "flux2"
    num_heads = 1
    num_kv_heads = 1
    attn_head_dim = 3
    double_layers = 2
    single_layers = 1

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.double_blocks = nn.ModuleList([_VideoDoubleBlock(), _VideoDoubleBlock()])
        self.single_blocks = nn.ModuleList([_VideoSingleBlock()])
        self.transformer = _Transformer()


class _FakeActionExpert(nn.Module):
    block_protocol = "flux2"
    num_heads = 1
    num_kv_heads = 1
    attn_head_dim = 3
    double_layers = 2
    single_layers = 1

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.double_blocks = nn.ModuleList([_ActionBlock(), _ActionBlock()])
        self.single_blocks = nn.ModuleList([_ActionBlock()])


def _fake_mot():
    return mot_module.MoT({"video": _FakeVideoExpert(), "action": _FakeActionExpert()}, mot_checkpoint_mixed_attn=False)


class MoTPackedAttentionTest(unittest.TestCase):
    def test_flux2_interleaved_calls_one_cross_chunk_attention_per_layer(self):
        module = _fake_mot()
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
                    "observation_token_lengths": (1,),
                    "target_length": 1,
                    "action_length": 1,
                },
            ),
            sparse_packing="interleaved",
            sparse_block_size=4,
        )
        mask = build_packed_block_sparse_mask(
            layout=layout,
            query_valid=torch.ones(1, layout.total_token_count, dtype=torch.bool),
            key_valid=torch.ones(1, layout.total_token_count, dtype=torch.bool),
            device="cpu",
            create_block_mask_fn=lambda *_args, **_kwargs: {"structural": True},
        )

        video_pre_states = []
        action_pre_states = []
        source_shapes = []
        for _index in range(2):
            txt = torch.randn(1, 2, 3)
            img = torch.randn(1, 2, 3)
            action = torch.randn(1, 1, 3)
            source_shapes.append((tuple(txt.shape), tuple(img.shape), tuple(action.shape)))
            video_pre_states.append(
                {
                    "tokens": {"txt": txt, "img": img},
                    "freqs": {
                        "txt": torch.zeros(1, 1, 2, 1),
                        "img": torch.zeros(1, 1, 2, 1),
                    },
                    "t_mod": {"double_img": None, "double_txt": None, "single": None},
                }
            )
            action_pre_states.append(
                {
                    "tokens": action,
                    "ids": torch.zeros(1, 1, 1),
                    "t_mod": {"double_img": None, "single": None},
                }
            )

        calls = []

        def fake_attention(q_cat, _k_cat, _v_cat, attention_mask, return_attn_probs=False):
            self.assertFalse(return_attn_probs)
            calls.append((tuple(q_cat.shape), attention_mask))
            return q_cat

        with mock.patch.object(module, "_mixed_attention", side_effect=fake_attention):
            outputs = module.forward_flux2_interleaved(
                video_pre_states=video_pre_states,
                action_pre_states=action_pre_states,
                attention_mask={"double_joint": mask, "single": mask},
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual([shape for shape, _mask in calls], [(1, 10, 3)] * 3)
        self.assertTrue(all(call_mask is mask for _shape, call_mask in calls))
        self.assertEqual(len(outputs), 2)
        for output, (txt_shape, img_shape, action_shape) in zip(outputs, source_shapes, strict=True):
            self.assertEqual(tuple(output["video"]["txt"].shape), txt_shape)
            self.assertEqual(tuple(output["video"]["img"].shape), img_shape)
            self.assertEqual(tuple(output["action"].shape), action_shape)

    def test_flux2_forward_calls_sparse_attention_once_per_double_and_single_layer(self):
        module = _fake_mot()
        mask = _sparse_mask(seq_len=5, batch_size=2)
        calls = []

        def fake_attention(q_cat, _k_cat, _v_cat, attention_mask, return_attn_probs=False):
            self.assertFalse(return_attn_probs)
            calls.append(attention_mask)
            return q_cat

        txt = torch.randn(2, 2, 3)
        img = torch.randn(2, 2, 3)
        action = torch.randn(2, 1, 3)
        with mock.patch.object(module, "_mixed_attention", side_effect=fake_attention):
            out = module(
                embeds_all={"video": {"txt": txt, "img": img}, "action": action},
                attention_mask={"double_joint": mask, "single": mask},
                freqs_all={"video": {"txt": torch.zeros(2, 1, 2, 1), "img": torch.zeros(2, 1, 2, 1)}},
                context_all={"video": None, "action": {"ids": torch.zeros(2, 1, 1)}},
                t_mod_all={
                    "video": {"double_img": None, "double_txt": None, "single": None},
                    "action": {"double_img": None, "single": None},
                },
            )

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call is mask for call in calls))
        self.assertEqual(tuple(out["video"]["txt"].shape), tuple(txt.shape))
        self.assertEqual(tuple(out["action"].shape), tuple(action.shape))

    def test_flux2_packed_forward_isolates_output_gradients_across_patterns(self):
        module = _fake_mot()
        mask = _sparse_mask(seq_len=5, batch_size=2)
        txt = torch.randn(2, 2, 3, requires_grad=True)
        img = torch.randn(2, 2, 3, requires_grad=True)
        action = torch.randn(2, 1, 3, requires_grad=True)

        with mock.patch.object(module, "_mixed_attention", side_effect=lambda q, _k, _v, _m, return_attn_probs=False: q):
            out = module(
                embeds_all={"video": {"txt": txt, "img": img}, "action": action},
                attention_mask={"double_joint": mask, "single": mask},
                freqs_all={"video": {"txt": torch.zeros(2, 1, 2, 1), "img": torch.zeros(2, 1, 2, 1)}},
                context_all={"video": None, "action": {"ids": torch.zeros(2, 1, 1)}},
                t_mod_all={
                    "video": {"double_img": None, "double_txt": None, "single": None},
                    "action": {"double_img": None, "single": None},
                },
            )
        loss = out["video"]["txt"][0].sum() + out["video"]["img"][0].sum() + out["action"][0].sum()
        loss.backward()

        self.assertGreater(txt.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(img.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(action.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(txt.grad[1].abs().sum().item(), 0.0)
        self.assertEqual(img.grad[1].abs().sum().item(), 0.0)
        self.assertEqual(action.grad[1].abs().sum().item(), 0.0)

    def test_sparse_attention_preserves_repeat_and_native_gqa_modes(self):
        mask = _sparse_mask(seq_len=5)
        for implementation, expected_heads, expected_gqa in (("repeat", 4, False), ("sdpa", 2, True)):
            module = mot_module.MoT.__new__(mot_module.MoT)
            nn.Module.__init__(module)
            module.num_heads = 4
            module.num_kv_heads = 2
            module.attn_head_dim = 3
            module.gqa_implementation = implementation
            module.mot_checkpoint_mixed_attn = False
            module.train(False)
            q = torch.randn(1, 5, 12)
            kv = torch.randn(1, 5, 6)
            calls = []

            def fake_packed_flex_attention(**kwargs):
                calls.append(kwargs)
                return torch.zeros_like(kwargs["query"])

            with mock.patch.object(mot_module, "packed_flex_attention", side_effect=fake_packed_flex_attention):
                module._mixed_attention(q, kv, kv, mask)
            self.assertEqual(tuple(calls[0]["key"].shape), (1, expected_heads, 5, 3))
            self.assertEqual(calls[0]["enable_gqa"], expected_gqa)

    def test_sparse_mixed_attention_checkpoint_uses_non_reentrant(self):
        mask = _sparse_mask(seq_len=5)
        module = mot_module.MoT.__new__(mot_module.MoT)
        nn.Module.__init__(module)
        module.num_heads = 1
        module.num_kv_heads = 1
        module.attn_head_dim = 3
        module.gqa_implementation = "repeat"
        module.mot_checkpoint_mixed_attn = True
        module.train(True)
        qkv = torch.randn(1, 5, 3, requires_grad=True)
        checkpoint_calls = []

        def fake_checkpoint(fn, *args, **kwargs):
            checkpoint_calls.append(kwargs)
            return fn(*args)

        with mock.patch("torch.utils.checkpoint.checkpoint", side_effect=fake_checkpoint):
            with mock.patch.object(mot_module, "packed_flex_attention", side_effect=lambda **kwargs: torch.zeros_like(kwargs["query"])):
                module._mixed_attention(qkv, qkv, qkv, mask)
        self.assertEqual(checkpoint_calls, [{"use_reentrant": False}])


if __name__ == "__main__":
    unittest.main()
