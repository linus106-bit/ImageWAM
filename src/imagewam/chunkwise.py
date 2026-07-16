from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ChunkwiseGeometry:
    num_chunks: int
    actions_per_chunk: int
    total_action_horizon: int
    num_frames: int
    observation_indices: tuple[int, ...]


def resolve_chunkwise_geometry(
    num_chunks: int,
    actions_per_chunk: int,
    *,
    num_frames: int | None,
) -> ChunkwiseGeometry:
    num_chunks = int(num_chunks)
    actions_per_chunk = int(actions_per_chunk)
    if num_chunks < 1:
        raise ValueError(f"`num_chunks` must be positive, got {num_chunks}.")
    if actions_per_chunk < 1:
        raise ValueError(f"`actions_per_chunk` must be positive, got {actions_per_chunk}.")
    total_action_horizon = num_chunks * actions_per_chunk
    expected_num_frames = total_action_horizon + 1
    if num_frames is not None and int(num_frames) != expected_num_frames:
        raise ValueError(
            "Chunkwise trajectory geometry mismatch: "
            f"num_frames={num_frames}, expected {expected_num_frames} for "
            f"num_chunks={num_chunks}, actions_per_chunk={actions_per_chunk}."
        )
    return ChunkwiseGeometry(
        num_chunks=num_chunks,
        actions_per_chunk=actions_per_chunk,
        total_action_horizon=total_action_horizon,
        num_frames=expected_num_frames,
        observation_indices=tuple(i * actions_per_chunk for i in range(num_chunks + 1)),
    )


def _validate_lengths(lengths: Sequence[int], name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in lengths)
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"`{name}` must contain positive lengths, got {values}.")
    return values


@torch.no_grad()
def build_chunkwise_causal_mask(
    *,
    text_attention_mask: torch.Tensor,
    observation_token_lengths: Sequence[int],
    target_length: int,
    action_padding_mask: torch.Tensor,
    state_positions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build the physical FLUX.2 chunk mask for one cumulative clean prefix.

    Sequence order is ``text, clean O0..Oi, noisy target, current action``.
    Future observations/actions cannot leak because they are not represented.
    """

    if text_attention_mask.ndim != 2:
        raise ValueError(
            f"`text_attention_mask` must be [B,L], got {tuple(text_attention_mask.shape)}."
        )
    if action_padding_mask.ndim != 2:
        raise ValueError(
            f"`action_padding_mask` must be [B,A], got {tuple(action_padding_mask.shape)}."
        )
    batch_size, text_length = text_attention_mask.shape
    if action_padding_mask.shape[0] != batch_size:
        raise ValueError("Text and action masks must have the same batch size.")
    observation_lengths = _validate_lengths(observation_token_lengths, "observation_token_lengths")
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError(f"`target_length` must be positive, got {target_length}.")

    action_length = int(action_padding_mask.shape[1])
    observation_total = sum(observation_lengths)
    observation_start = int(text_length)
    target_start = observation_start + observation_total
    action_start = target_start + target_length
    total_length = action_start + action_length
    device = text_attention_mask.device
    mask = torch.zeros(batch_size, total_length, total_length, dtype=torch.bool, device=device)

    text_valid = text_attention_mask.to(device=device, dtype=torch.bool).clone()
    if state_positions is not None:
        if state_positions.ndim != 1 or state_positions.shape[0] != batch_size:
            raise ValueError(
                f"`state_positions` must be [B], got {tuple(state_positions.shape)}."
            )
        state_positions = state_positions.to(device=device, dtype=torch.long)
        if bool(((state_positions < 0) | (state_positions >= text_length)).any()):
            raise ValueError("Every state position must be inside the text sequence.")
        text_valid[torch.arange(batch_size, device=device), state_positions] = False

    # Valid instruction tokens attend only to valid instruction tokens.
    mask[:, :text_length, :text_length] = text_valid[:, :, None] & text_valid[:, None, :]

    # Clean boundary observations are causal at observation granularity.
    cursor = observation_start
    for observation_length in observation_lengths:
        end = cursor + observation_length
        mask[:, cursor:end, :text_length] = text_valid[:, None, :]
        mask[:, cursor:end, observation_start:end] = True
        cursor = end

    # The current noisy target and action form one current-chunk group.
    current_start = target_start
    mask[:, current_start:total_length, :text_length] = text_valid[:, None, :]
    mask[:, current_start:total_length, observation_start:total_length] = True

    if state_positions is not None:
        batch_indices = torch.arange(batch_size, device=device)
        # Current target/action may condition on the state anchor; the state query itself is self-only.
        mask[batch_indices[:, None], torch.arange(current_start, total_length, device=device), state_positions[:, None]] = True
        mask[batch_indices, state_positions] = False
        mask[batch_indices, state_positions, state_positions] = True

    # Padding is a key-side invariant for every query.
    mask[:, :, :text_length] &= text_attention_mask.to(device=device, dtype=torch.bool)[:, None, :]
    action_valid = ~action_padding_mask.to(device=device, dtype=torch.bool)
    mask[:, :, action_start:total_length] &= action_valid[:, None, :]
    return {"double_joint": mask, "single": mask.clone()}


def chunkwise_loss_contribution(
    *,
    video_squared_error: torch.Tensor,
    action_squared_error: torch.Tensor,
    valid_target: torch.Tensor,
    video_weight: torch.Tensor,
    action_weight: torch.Tensor,
    video_denominator: torch.Tensor,
    action_denominator: torch.Tensor,
    lambda_video: float,
    lambda_action: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = int(video_squared_error.shape[0])
    tensors = (
        action_squared_error,
        valid_target,
        video_weight,
        action_weight,
        video_denominator,
        action_denominator,
    )
    if any(int(tensor.shape[0]) != batch_size for tensor in tensors):
        raise ValueError("All chunk loss tensors must have the same batch dimension.")
    video_sum = video_squared_error.flatten(1).sum(dim=1)
    action_sum = action_squared_error.flatten(1).sum(dim=1)
    video_term = (
        valid_target.to(device=video_sum.device, dtype=video_sum.dtype)
        * video_weight.to(device=video_sum.device, dtype=video_sum.dtype)
        * video_sum
        / video_denominator.to(device=video_sum.device, dtype=video_sum.dtype).clamp(min=1.0)
    )
    action_term = (
        action_weight.to(device=action_sum.device, dtype=action_sum.dtype)
        * action_sum
        / action_denominator.to(device=action_sum.device, dtype=action_sum.dtype).clamp(min=1.0)
    )
    loss_video = float(lambda_video) * video_term.mean()
    loss_action = float(lambda_action) * action_term.mean()
    return loss_video + loss_action, {"loss_video": loss_video, "loss_action": loss_action}
