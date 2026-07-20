from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

PACKED_CHUNK_LAYOUT_SCHEMA_VERSION = 1
_PACKED_TOPOLOGY_CACHE_MAX_ENTRIES = 16
_PACKED_TOPOLOGY_CACHE: OrderedDict[tuple[Any, ...], object] = OrderedDict()

ChunkwiseForwardMode = Literal["sequential", "packed_flex"]
SparsePackingKind = Literal["interleaved", "batch_padded"]
SparseAlignment = Literal["none", "segment"]


@dataclass(frozen=True)
class ChunkwiseGeometry:
    num_chunks: int
    actions_per_chunk: int
    total_action_horizon: int
    num_frames: int
    observation_indices: tuple[int, ...]


@dataclass(frozen=True)
class PackedChunkSegment:
    """Immutable token ownership for one packed chunk-conditioning pattern.

    Local physical token order is text, cumulative clean observations, noisy target,
    current action. Ranges are global packed offsets and are half-open.
    """

    pattern_index: int
    chunk_ordinal: int
    text_range: tuple[int, int]
    clean_observation_ranges: tuple[tuple[int, int], ...]
    target_range: tuple[int, int]
    action_range: tuple[int, int]
    padding_range: tuple[int, int] | None = None

    @property
    def local_start(self) -> int:
        return self.text_range[0]

    @property
    def local_end(self) -> int:
        return self.action_range[1] if self.padding_range is None else self.padding_range[1]

    @property
    def physical_end(self) -> int:
        return self.action_range[1]

    @property
    def physical_length(self) -> int:
        return self.physical_end - self.local_start

    @property
    def aligned_length(self) -> int:
        return self.local_end - self.local_start

    @property
    def text_length(self) -> int:
        return self.text_range[1] - self.text_range[0]

    @property
    def target_length(self) -> int:
        return self.target_range[1] - self.target_range[0]

    @property
    def action_length(self) -> int:
        return self.action_range[1] - self.action_range[0]

    @property
    def clean_observation_lengths(self) -> tuple[int, ...]:
        return tuple(end - start for start, end in self.clean_observation_ranges)


@dataclass(frozen=True)
class PackedChunkLayout:
    """Packed sparse-attention layout independent of per-forward validity tensors."""

    segments: tuple[PackedChunkSegment, ...]
    total_token_count: int
    sparse_packing: SparsePackingKind = "interleaved"
    sparse_alignment: SparseAlignment = "none"
    sparse_block_size: int = 128
    schema_version: int = PACKED_CHUNK_LAYOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PACKED_CHUNK_LAYOUT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported packed layout schema version {self.schema_version}; "
                f"expected {PACKED_CHUNK_LAYOUT_SCHEMA_VERSION}."
            )
        if self.sparse_packing not in {"interleaved", "batch_padded"}:
            raise ValueError(f"Unsupported sparse packing {self.sparse_packing!r}.")
        if self.sparse_alignment not in {"none", "segment"}:
            raise ValueError(f"Unsupported sparse alignment {self.sparse_alignment!r}.")
        if int(self.sparse_block_size) <= 0:
            raise ValueError("`sparse_block_size` must be positive.")
        if not self.segments:
            raise ValueError("Packed layouts require at least one segment.")
        expected_start = 0
        owners: list[int] = []
        roles: list[str] = []
        local_indices: list[int] = []
        for segment in self.segments:
            ranges = (
                (segment.text_range, "text"),
                *[(obs_range, "observation") for obs_range in segment.clean_observation_ranges],
                (segment.target_range, "target"),
                (segment.action_range, "action"),
            )
            if segment.local_start != expected_start:
                raise ValueError(
                    "Packed segments must be contiguous and non-overlapping; "
                    f"expected start {expected_start}, got {segment.local_start}."
                )
            cursor = segment.local_start
            local_index = 0
            for (start, end), role in ranges:
                if start != cursor or end <= start:
                    raise ValueError(
                        "Packed segment ranges must be positive and contiguous within a segment."
                    )
                length = end - start
                owners.extend([segment.pattern_index] * length)
                roles.extend([role] * length)
                local_indices.extend(range(local_index, local_index + length))
                cursor = end
                local_index += length
            if segment.padding_range is not None:
                pad_start, pad_end = segment.padding_range
                if pad_start != cursor or pad_end < pad_start:
                    raise ValueError("Padding range must start at the physical segment end.")
                pad_length = pad_end - pad_start
                owners.extend([segment.pattern_index] * pad_length)
                roles.extend(["padding"] * pad_length)
                local_indices.extend(range(local_index, local_index + pad_length))
                cursor = pad_end
            expected_start = cursor
        if expected_start != int(self.total_token_count):
            raise ValueError(
                f"Packed layout total_token_count={self.total_token_count} does not match "
                f"segment extent {expected_start}."
            )
        object.__setattr__(self, "pattern_owner", tuple(owners))
        object.__setattr__(self, "token_role", tuple(roles))
        object.__setattr__(self, "local_token_index", tuple(local_indices))

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.sparse_packing,
            self.sparse_alignment,
            int(self.sparse_block_size),
            tuple(
                (
                    segment.pattern_index,
                    segment.chunk_ordinal,
                    segment.text_range,
                    segment.clean_observation_ranges,
                    segment.target_range,
                    segment.action_range,
                    segment.padding_range,
                )
                for segment in self.segments
            ),
            int(self.total_token_count),
        )

    def global_to_local(self, global_index: int) -> tuple[int, int, str]:
        index = int(global_index)
        if index < 0 or index >= self.total_token_count:
            raise IndexError(f"Global token index {index} outside [0,{self.total_token_count}).")
        return self.pattern_owner[index], self.local_token_index[index], self.token_role[index]

    def local_to_global(self, pattern_index: int, local_index: int) -> int:
        pattern_index = int(pattern_index)
        local_index = int(local_index)
        for global_index, (owner, local) in enumerate(
            zip(self.pattern_owner, self.local_token_index, strict=True)
        ):
            if owner == pattern_index and local == local_index:
                return global_index
        raise IndexError(f"No token for pattern={pattern_index}, local={local_index}.")


def _align_length(length: int, block_size: int, alignment: SparseAlignment) -> int:
    if alignment == "none":
        return int(length)
    remainder = int(length) % int(block_size)
    return int(length) if remainder == 0 else int(length) + int(block_size) - remainder


def build_packed_chunk_layout(
    *,
    chunks: Sequence[dict[str, Any]],
    sparse_packing: SparsePackingKind = "interleaved",
    sparse_alignment: SparseAlignment = "none",
    sparse_block_size: int = 128,
) -> PackedChunkLayout:
    """Build the immutable packed token layout for chunk pattern topology.

    `chunks` entries require text_length, observation_token_lengths, target_length, and
    action_length. This builder deliberately excludes batch validity tensors so the
    topology cache cannot capture current-batch values.
    """

    sparse_block_size = int(sparse_block_size)
    if sparse_block_size <= 0:
        raise ValueError("`sparse_block_size` must be positive.")
    if sparse_packing not in {"interleaved", "batch_padded"}:
        raise ValueError(f"Unsupported sparse_packing={sparse_packing!r}.")
    if sparse_alignment not in {"none", "segment"}:
        raise ValueError(f"Unsupported sparse_alignment={sparse_alignment!r}.")
    segments: list[PackedChunkSegment] = []
    cursor = 0
    for pattern_index, chunk in enumerate(chunks):
        text_length = int(chunk["text_length"])
        observation_lengths = _validate_lengths(
            chunk.get("observation_token_lengths", ()), "observation_token_lengths"
        )
        target_length = int(chunk["target_length"])
        action_length = int(chunk["action_length"])
        if text_length <= 0 or target_length <= 0 or action_length <= 0:
            raise ValueError("text_length, target_length, and action_length must be positive.")
        start = cursor
        text_range = (cursor, cursor + text_length)
        cursor = text_range[1]
        obs_ranges = []
        for observation_length in observation_lengths:
            obs_ranges.append((cursor, cursor + observation_length))
            cursor += observation_length
        target_range = (cursor, cursor + target_length)
        cursor = target_range[1]
        action_range = (cursor, cursor + action_length)
        cursor = action_range[1]
        aligned_end = start + _align_length(cursor - start, sparse_block_size, sparse_alignment)
        padding_range = None if aligned_end == cursor else (cursor, aligned_end)
        cursor = aligned_end
        segments.append(
            PackedChunkSegment(
                pattern_index=pattern_index,
                chunk_ordinal=int(chunk.get("chunk_ordinal", pattern_index)),
                text_range=text_range,
                clean_observation_ranges=tuple(obs_ranges),
                target_range=target_range,
                action_range=action_range,
                padding_range=padding_range,
            )
        )
    return PackedChunkLayout(
        segments=tuple(segments),
        total_token_count=cursor,
        sparse_packing=sparse_packing,
        sparse_alignment=sparse_alignment,
        sparse_block_size=sparse_block_size,
    )


@dataclass(frozen=True)
class PackedBlockSparseMask:
    """ImageWAM-owned wrapper for sparse topology plus immutable dynamic edges.

    `block_mask` may hold PyTorch's prototype BlockMask. Tests may leave it as None
    and use `to_dense_reference`; production code must not use dense materialization.
    """

    layout: PackedChunkLayout
    block_mask: object | None
    query_valid: torch.Tensor | None = None
    key_valid: torch.Tensor | None = None
    state_positions: torch.Tensor | None = None
    sparse_block_size: int = 128
    sparse_alignment: SparseAlignment = "none"

    @property
    def signature(self) -> tuple[Any, ...]:
        return self.layout.signature + (int(self.sparse_block_size), self.sparse_alignment)

    def to_dense_reference(
        self,
        *,
        text_attention_masks: Sequence[torch.Tensor],
        clean_observation_valid: Sequence[torch.Tensor],
        target_valid: Sequence[torch.Tensor],
        action_padding_masks: Sequence[torch.Tensor],
        state_positions: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Diagnostic/test-only dense materializer matching build_chunkwise_causal_mask."""

        batch_size = int(text_attention_masks[0].shape[0])
        dense = torch.zeros(
            batch_size,
            self.layout.total_token_count,
            self.layout.total_token_count,
            dtype=torch.bool,
            device=text_attention_masks[0].device,
        )
        state_positions = state_positions or [None] * len(self.layout.segments)
        for index, segment in enumerate(self.layout.segments):
            local = build_chunkwise_causal_mask(
                text_attention_mask=text_attention_masks[index],
                observation_token_lengths=segment.clean_observation_lengths,
                clean_observation_valid=clean_observation_valid[index],
                target_length=segment.target_length,
                target_valid=target_valid[index],
                action_padding_mask=action_padding_masks[index],
                state_positions=state_positions[index],
            )["double_joint"]
            physical = segment.physical_length
            dense[:, segment.local_start : segment.physical_end, segment.local_start : segment.physical_end] = local[:, :physical, :physical]
        return dense



_ROLE_ID = {"text": 0, "observation": 1, "target": 2, "action": 3, "padding": 4}


def _layout_role_tensor(layout: PackedChunkLayout, *, device: torch.device | str) -> torch.Tensor:
    return torch.tensor([_ROLE_ID[role] for role in layout.token_role], device=device, dtype=torch.long)


def _layout_owner_tensor(layout: PackedChunkLayout, *, device: torch.device | str) -> torch.Tensor:
    return torch.tensor(layout.pattern_owner, device=device, dtype=torch.long)


def _layout_local_index_tensor(layout: PackedChunkLayout, *, device: torch.device | str) -> torch.Tensor:
    return torch.tensor(layout.local_token_index, device=device, dtype=torch.long)


def _build_structural_mask_mod(
    layout: PackedChunkLayout,
    *,
    device: torch.device | str,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the batch-independent packed structural predicate for FlexAttention.

    Dynamic batch validity is intentionally excluded. Partial blocks still call this
    token predicate, so boundaries at block-size +/- 1 remain exact.
    """

    owner = _layout_owner_tensor(layout, device=device)
    role = _layout_role_tensor(layout, device=device)
    local_index = _layout_local_index_tensor(layout, device=device)
    padding_id = _ROLE_ID["padding"]
    text_id = _ROLE_ID["text"]
    observation_id = _ROLE_ID["observation"]
    target_id = _ROLE_ID["target"]
    action_id = _ROLE_ID["action"]
    observation_ordinal = torch.full(
        (layout.total_token_count,), -1, device=device, dtype=torch.long
    )
    for segment in layout.segments:
        for ordinal, (start, end) in enumerate(segment.clean_observation_ranges):
            observation_ordinal[start:end] = ordinal

    def mask_mod(_batch: torch.Tensor, _head: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        same_pattern = owner[query_index] == owner[key_index]
        query_role = role[query_index]
        key_role = role[key_index]
        query_not_padding = query_role != padding_id
        key_not_padding = key_role != padding_id
        text_pair = (query_role == text_id) & (key_role == text_id)
        observation_query = query_role == observation_id
        observation_key = key_role == observation_id
        observation_causal = (
            observation_query
            & observation_key
            & (observation_ordinal[key_index] <= observation_ordinal[query_index])
        )
        observation_to_text = observation_query & (key_role == text_id)
        query_current = (query_role == target_id) | (query_role == action_id)
        key_current = (key_role == target_id) | (key_role == action_id)
        current_joint = query_current & key_current
        current_to_prefix = query_current & ((key_role == text_id) | observation_key)
        # Conservative fallback for unknown future roles while preserving current local order.
        local_causal = local_index[key_index] <= local_index[query_index]
        allowed_local = text_pair | observation_to_text | observation_causal | current_to_prefix | current_joint | local_causal
        return same_pattern & query_not_padding & key_not_padding & allowed_local

    return mask_mod


def packed_block_topology_cache_key(
    layout: PackedChunkLayout,
    *,
    device: torch.device | str,
    block_size: int | tuple[int, int] | None = None,
    num_heads: int | None = None,
) -> tuple[Any, ...]:
    """Cache key for structural topology only; excludes current-batch tensors."""

    head_key = None if num_heads is None else int(num_heads)
    return (
        str(torch.device(device)),
        int(block_size or layout.sparse_block_size),
        head_key,
        layout.signature,
    )


def build_packed_block_sparse_mask(
    *,
    layout: PackedChunkLayout,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    state_positions: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    num_heads: int | None = None,
    create_block_mask_fn: Callable[..., object] | None = None,
) -> PackedBlockSparseMask:
    """Build an ImageWAM-owned packed FlexAttention mask without dense allocation.

    `query_valid`, `key_valid`, and `state_positions` are cloned/detached per forward
    so cached topology cannot retain mutable current-batch state.
    """

    if query_valid.ndim != 2 or key_valid.ndim != 2:
        raise ValueError("`query_valid` and `key_valid` must be [B,L] boolean tensors.")
    if tuple(query_valid.shape) != tuple(key_valid.shape):
        raise ValueError("`query_valid` and `key_valid` must have identical [B,L] shapes.")
    batch_size, seq_len = query_valid.shape
    if int(seq_len) != int(layout.total_token_count):
        raise ValueError(
            "Packed dynamic validity length must match layout.total_token_count: "
            f"got {seq_len}, expected {layout.total_token_count}."
        )
    target_device = torch.device(device) if device is not None else query_valid.device
    query_snapshot = query_valid.to(device=target_device, dtype=torch.bool).clone().detach()
    key_snapshot = key_valid.to(device=target_device, dtype=torch.bool).clone().detach()
    state_snapshot = None
    if state_positions is not None:
        if state_positions.ndim != 1 or int(state_positions.shape[0]) != int(batch_size):
            raise ValueError(f"`state_positions` must be [B], got {tuple(state_positions.shape)}.")
        state_snapshot = state_positions.to(device=target_device, dtype=torch.long).clone().detach()
        if bool(((state_snapshot < -1) | (state_snapshot >= seq_len)).any()):
            raise ValueError("Every packed state position must be -1 or inside the packed sequence.")

    if create_block_mask_fn is None:
        from torch.nn.attention.flex_attention import create_block_mask as create_block_mask_fn

    block_size = int(layout.sparse_block_size)
    key = packed_block_topology_cache_key(
        layout,
        device=target_device,
        block_size=block_size,
        num_heads=num_heads,
    )

    def _create() -> object:
        return create_block_mask_fn(
            _build_structural_mask_mod(layout, device=target_device),
            B=None,
            H=num_heads,
            Q_LEN=int(seq_len),
            KV_LEN=int(seq_len),
            device=str(target_device),
            BLOCK_SIZE=block_size,
        )

    sentinel = object()
    existing = _PACKED_TOPOLOGY_CACHE.get(key, sentinel)
    block_mask = cache_packed_block_topology(key, _create() if existing is sentinel else existing)
    return PackedBlockSparseMask(
        layout=layout,
        block_mask=block_mask,
        query_valid=query_snapshot,
        key_valid=key_snapshot,
        state_positions=state_snapshot,
        sparse_block_size=block_size,
        sparse_alignment=layout.sparse_alignment,
    )


def packed_flex_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: PackedBlockSparseMask,
    scale: float | None = None,
    enable_gqa: bool = False,
    flex_attention_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Call PyTorch FlexAttention through the ImageWAM sparse-mask adapter."""

    if mask.query_valid is None or mask.key_valid is None:
        raise ValueError("Packed FlexAttention requires per-forward query/key validity tensors.")
    if flex_attention_fn is None:
        from torch.nn.attention.flex_attention import flex_attention as flex_attention_fn

    query_valid = mask.query_valid.to(device=query.device, dtype=torch.bool)
    key_valid = mask.key_valid.to(device=query.device, dtype=torch.bool)
    state_positions = None
    if mask.state_positions is not None:
        state_positions = mask.state_positions.to(device=query.device, dtype=torch.long)

    def score_mod(score: torch.Tensor, batch: torch.Tensor, _head: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        allowed = query_valid[batch, query_index] & key_valid[batch, key_index]
        if state_positions is not None:
            query_is_state = query_index == state_positions[batch]
            allowed = torch.where(query_is_state, key_index == query_index, allowed)
        return torch.where(allowed, score, torch.full_like(score, torch.finfo(score.dtype).min))

    return flex_attention_fn(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=mask.block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def cache_packed_block_topology(key: tuple[Any, ...], value: object) -> object:
    """LRU cache for immutable sparse topology only; dynamic validity is excluded."""

    if key in _PACKED_TOPOLOGY_CACHE:
        _PACKED_TOPOLOGY_CACHE.move_to_end(key)
        return _PACKED_TOPOLOGY_CACHE[key]
    _PACKED_TOPOLOGY_CACHE[key] = value
    while len(_PACKED_TOPOLOGY_CACHE) > _PACKED_TOPOLOGY_CACHE_MAX_ENTRIES:
        _PACKED_TOPOLOGY_CACHE.popitem(last=False)
    return value


def packed_topology_cache_size() -> int:
    return len(_PACKED_TOPOLOGY_CACHE)


def clear_packed_topology_cache() -> None:
    _PACKED_TOPOLOGY_CACHE.clear()


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
    clean_observation_valid: torch.Tensor,
    target_length: int,
    target_valid: torch.Tensor,
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
    if clean_observation_valid.ndim != 2 or tuple(clean_observation_valid.shape) != (
        batch_size,
        len(observation_lengths),
    ):
        raise ValueError(
            "`clean_observation_valid` must be [B,N], "
            f"got {tuple(clean_observation_valid.shape)} for B={batch_size}, "
            f"N={len(observation_lengths)}."
        )
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError(f"`target_length` must be positive, got {target_length}.")
    if target_valid.ndim != 1 or tuple(target_valid.shape) != (batch_size,):
        raise ValueError(
            f"`target_valid` must be [B], got {tuple(target_valid.shape)} for B={batch_size}."
        )

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
    clean_valid = clean_observation_valid.to(device=device, dtype=torch.bool)
    cursor = observation_start
    for ordinal, observation_length in enumerate(observation_lengths):
        end = cursor + observation_length
        mask[:, :, cursor:end] &= clean_valid[:, ordinal, None, None]
        cursor = end
    mask[:, :, target_start:action_start] &= target_valid.to(
        device=device, dtype=torch.bool
    )[:, None, None]
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

    def _per_batch_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        if weight.ndim == 0:
            return weight.expand(batch_size)
        if weight.ndim != 1 or int(weight.shape[0]) != batch_size:
            raise ValueError(
                f"`{name}` must be scalar or [B], got {tuple(weight.shape)} for B={batch_size}."
            )
        return weight

    video_weight = _per_batch_weight(video_weight, "video_weight")
    action_weight = _per_batch_weight(action_weight, "action_weight")
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
