# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Vectorized batch-1 PyTorch implementation of Quest decode.

This implementation removes Python loops and CUDA scalar synchronizations
from page scoring, selection, KV gathering, and selected-page attention.

It remains a PyTorch MVP rather than the final fused Triton/CUDA kernel.
"""

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.quest.reference import QuestPageSelection


@dataclass(frozen=True)
class QuestVectorizedResult:
    """Result returned by vectorized batch-1 Quest decode.

    scores contains only valid logical pages and has shape:

        [1, num_query_heads, num_valid_pages]
    """

    output: torch.Tensor
    scores: torch.Tensor
    selection: QuestPageSelection


def vectorized_quest_decode_batch1(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
    block_table: torch.Tensor,
    *,
    sequence_length: int,
    page_budget: int,
    block_size: int,
    scale: float | None = None,
) -> QuestVectorizedResult:
    """Run vectorized Quest decode for exactly one active request.

    Args:
        query:
            Shape [1, num_query_heads, head_size].
        kv_cache:
            Shape [num_physical_pages, 2, block_size,
                   num_kv_heads, head_size].
        page_min/page_max:
            Shape [num_physical_pages, num_kv_heads, head_size].
        block_table:
            Shape [1, max_logical_pages].
        sequence_length:
            Current sequence length including this decode token.
        page_budget:
            Maximum number of pages attended per query head.
        block_size:
            Number of tokens per KV page.
        scale:
            Optional attention scale.

    Returns:
        Vectorized Quest output, scores, and selected page IDs.
    """

    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError(
            "query must have shape "
            "[1, num_query_heads, head_size]"
        )

    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "kv_cache must have shape "
            "[num_pages, 2, block_size, num_kv_heads, head_size]"
        )

    if page_min.shape != page_max.shape:
        raise ValueError(
            "page_min and page_max must have identical shapes"
        )

    if page_min.ndim != 3:
        raise ValueError(
            "page metadata must have shape "
            "[num_pages, num_kv_heads, head_size]"
        )

    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError(
            "block_table must have shape [1, max_logical_pages]"
        )

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if page_budget <= 0:
        raise ValueError("page_budget must be positive")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    (
        num_physical_pages,
        _,
        cache_block_size,
        num_kv_heads,
        cache_head_size,
    ) = kv_cache.shape

    _, num_query_heads, head_size = query.shape

    if cache_block_size != block_size:
        raise ValueError(
            f"KV block size {cache_block_size} does not match "
            f"requested block size {block_size}"
        )

    if cache_head_size != head_size:
        raise ValueError(
            f"KV head size {cache_head_size} does not match "
            f"query head size {head_size}"
        )

    if page_min.shape != (
        num_physical_pages,
        num_kv_heads,
        head_size,
    ):
        raise ValueError(
            "page metadata does not match the KV-cache layout"
        )

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_query_heads} query heads cannot map evenly onto "
            f"{num_kv_heads} KV heads"
        )

    valid_pages = (
        sequence_length + block_size - 1
    ) // block_size

    if valid_pages > block_table.shape[1]:
        raise ValueError(
            "block_table does not contain all valid logical pages"
        )

    selected_count = min(page_budget, valid_pages)
    current_logical_page = valid_pages - 1
    historical_budget = selected_count - 1

    device = query.device
    query_dtype = query.dtype

    physical_pages = block_table[
        0,
        :valid_pages,
    ].to(device=device, dtype=torch.long)

    queries_per_kv_head = (
        num_query_heads // num_kv_heads
    )

    query_to_kv_head = torch.div(
        torch.arange(
            num_query_heads,
            device=device,
            dtype=torch.long,
        ),
        queries_per_kv_head,
        rounding_mode="floor",
    )

    # Gather metadata for this request's physical pages:
    #
    # [P, Hkv, D] -> [P, Hq, D] -> [Hq, P, D].
    request_page_min = page_min.index_select(
        0,
        physical_pages,
    )
    request_page_max = page_max.index_select(
        0,
        physical_pages,
    )

    query_page_min = request_page_min.index_select(
        1,
        query_to_kv_head,
    ).permute(1, 0, 2)

    query_page_max = request_page_max.index_select(
        1,
        query_to_kv_head,
    ).permute(1, 0, 2)

    # Score every query head and valid logical page simultaneously.
    #
    # [Hq, D] -> [Hq, 1, D], broadcast over P pages.
    query_fp32 = query[0].float()
    query_expanded = query_fp32.unsqueeze(1)

    page_bounds = torch.where(
        query_expanded >= 0,
        query_page_max.float(),
        query_page_min.float(),
    )

    scores = (
        query_expanded * page_bounds
    ).sum(dim=-1)

    # Select historical pages for all query heads in one Top-K call.
    if historical_budget == 0:
        historical_pages = torch.empty(
            (num_query_heads, 0),
            device=device,
            dtype=torch.long,
        )
    elif historical_budget >= current_logical_page:
        historical_pages = torch.arange(
            current_logical_page,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(
            num_query_heads,
            -1,
        )
    else:
        historical_pages = torch.topk(
            scores[:, :current_logical_page],
            k=historical_budget,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices

        # Sorting is not required mathematically, but gives deterministic
        # logical-page ordering and matches the reference implementation.
        historical_pages = torch.sort(
            historical_pages,
            dim=-1,
        ).values

    # The current page is mandatory and never competes in historical Top-K.
    current_pages = torch.full(
        (num_query_heads, 1),
        current_logical_page,
        device=device,
        dtype=torch.long,
    )

    selected_logical_pages = torch.cat(
        (historical_pages, current_pages),
        dim=-1,
    )

    # Translate all selected logical IDs to physical IDs simultaneously.
    selected_physical_pages = physical_pages[
        selected_logical_pages
    ]

    # Advanced indexing gathers all selected K/V pages for all query heads:
    #
    # [Hq, K, block_size, D].
    #
    # query_to_kv_head[:, None] broadcasts over K selected pages.
    selected_keys = kv_cache[
        selected_physical_pages,
        0,
        :,
        query_to_kv_head[:, None],
        :,
    ]

    selected_values = kv_cache[
        selected_physical_pages,
        1,
        :,
        query_to_kv_head[:, None],
        :,
    ]

    # Mask token slots beyond the request's actual sequence length.
    token_offsets = torch.arange(
        block_size,
        device=device,
        dtype=torch.long,
    )

    selected_token_positions = (
        selected_logical_pages.unsqueeze(-1) * block_size
        + token_offsets
    )

    valid_token_mask = (
        selected_token_positions < sequence_length
    )

    selected_keys = selected_keys.reshape(
        num_query_heads,
        selected_count * block_size,
        head_size,
    )

    selected_values = selected_values.reshape(
        num_query_heads,
        selected_count * block_size,
        head_size,
    )

    valid_token_mask = valid_token_mask.reshape(
        num_query_heads,
        selected_count * block_size,
    )

    attention_scale = (
        float(scale)
        if scale is not None
        else float(head_size) ** -0.5
    )

    # Batched QK multiplication for all query heads:
    #
    # [Hq, 1, D] @ [Hq, D, S] -> [Hq, 1, S].
    logits = torch.bmm(
        query_fp32.unsqueeze(1),
        selected_keys.float().transpose(1, 2),
    ).squeeze(1)

    logits.mul_(attention_scale)
    logits.masked_fill_(
        ~valid_token_mask,
        float("-inf"),
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    # Batched PV multiplication:
    #
    # [Hq, 1, S] @ [Hq, S, D] -> [Hq, 1, D].
    output = torch.bmm(
        probabilities.unsqueeze(1),
        selected_values.float(),
    ).squeeze(1)

    output = output.to(
        dtype=query_dtype,
    ).unsqueeze(0)

    selection = QuestPageSelection(
        logical_page_ids=(
            selected_logical_pages.unsqueeze(0)
        ),
        physical_page_ids=(
            selected_physical_pages.unsqueeze(0)
        ),
        num_selected_pages=torch.full(
            (1, num_query_heads),
            selected_count,
            device=device,
            dtype=torch.int32,
        ),
    )

    return QuestVectorizedResult(
        output=output,
        scores=scores.unsqueeze(0),
        selection=selection,
    )
