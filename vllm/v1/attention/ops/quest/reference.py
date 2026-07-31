# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuestPageSelection:
    """Reference Top-K page-selection result.

    All page-ID tensors have shape:

        [num_decode_tokens, num_query_heads, max_selected_pages]

    Unused entries are padded with -1.
    """

    logical_page_ids: torch.Tensor
    physical_page_ids: torch.Tensor
    num_selected_pages: torch.Tensor


def _num_logical_pages(
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if seq_lens.ndim != 1:
        raise ValueError(
            f"seq_lens must be one-dimensional, got {seq_lens.shape}"
        )

    if bool(torch.any(seq_lens <= 0).item()):
        raise ValueError("Every decode sequence must contain at least one token")

    return torch.div(
        seq_lens + block_size - 1,
        block_size,
        rounding_mode="floor",
    )


def reference_score_pages(
    query: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Compute Quest page scores in readable PyTorch.

    Args:
        query:
            Decode queries with shape
            [num_decode_tokens, num_query_heads, head_size].
        page_min/page_max:
            Physical-page key metadata with shape
            [num_physical_pages, num_kv_heads, head_size].
        block_table:
            Logical-to-physical page mapping with shape
            [num_decode_tokens, max_logical_pages].
        seq_lens:
            Sequence lengths, including the current decode token.
        block_size:
            Number of KV tokens per physical page.

    Returns:
        FP32 scores with shape
        [num_decode_tokens, num_query_heads, max_logical_pages].

        Logical pages outside each request's sequence are assigned -inf.
    """

    if query.ndim != 3:
        raise ValueError(
            "query must have shape "
            "[num_decode_tokens, num_query_heads, head_size]"
        )

    if page_min.ndim != 3 or page_max.ndim != 3:
        raise ValueError(
            "page_min and page_max must have shape "
            "[num_physical_pages, num_kv_heads, head_size]"
        )

    if page_min.shape != page_max.shape:
        raise ValueError(
            f"page_min shape {page_min.shape} does not match "
            f"page_max shape {page_max.shape}"
        )

    if block_table.ndim != 2:
        raise ValueError(
            "block_table must have shape "
            "[num_decode_tokens, max_logical_pages]"
        )

    num_tokens, num_query_heads, head_size = query.shape
    num_physical_pages, num_kv_heads, metadata_head_size = (
        page_min.shape
    )

    if block_table.shape[0] != num_tokens:
        raise ValueError(
            "block_table row count must match the number of queries"
        )

    if seq_lens.shape != (num_tokens,):
        raise ValueError(
            f"seq_lens must have shape {(num_tokens,)}, "
            f"got {seq_lens.shape}"
        )

    if metadata_head_size != head_size:
        raise ValueError(
            f"Query head size {head_size} does not match "
            f"metadata head size {metadata_head_size}"
        )

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_query_heads} query heads cannot be evenly mapped to "
            f"{num_kv_heads} KV heads"
        )

    if query.device != page_min.device:
        raise ValueError("query and page metadata must be on the same device")

    device = query.device
    physical_ids = block_table.to(device=device, dtype=torch.long)
    seq_lens_device = seq_lens.to(device=device, dtype=torch.long)

    num_pages = _num_logical_pages(seq_lens_device, block_size)
    max_logical_pages = block_table.shape[1]

    if bool(torch.any(num_pages > max_logical_pages).item()):
        raise ValueError(
            "block_table does not contain enough logical-page entries"
        )

    logical_page_indices = torch.arange(
        max_logical_pages,
        device=device,
    )
    valid_page_mask = (
        logical_page_indices.unsqueeze(0) < num_pages.unsqueeze(1)
    )

    valid_physical_ids = physical_ids[valid_page_mask]
    if valid_physical_ids.numel() > 0:
        if bool(torch.any(valid_physical_ids < 0).item()):
            raise ValueError(
                "Valid logical pages must map to non-negative physical IDs"
            )
        if bool(
            torch.any(valid_physical_ids >= num_physical_pages).item()
        ):
            raise ValueError(
                "block_table contains an out-of-range physical page ID"
            )

    # Invalid/padded block-table entries are replaced temporarily so that the
    # metadata gather itself is safe. Their final scores are masked to -inf.
    safe_physical_ids = torch.where(
        valid_page_mask,
        physical_ids,
        torch.zeros_like(physical_ids),
    )

    # [T, P, Hkv, D]
    request_page_min = page_min[safe_physical_ids]
    request_page_max = page_max[safe_physical_ids]

    queries_per_kv_head = num_query_heads // num_kv_heads
    query_to_kv_head = torch.div(
        torch.arange(num_query_heads, device=device),
        queries_per_kv_head,
        rounding_mode="floor",
    )

    # Convert metadata from KV-head layout to query-head layout:
    # [T, P, Hq, D] -> [T, Hq, P, D].
    query_page_min = request_page_min.index_select(
        2,
        query_to_kv_head,
    ).permute(0, 2, 1, 3)

    query_page_max = request_page_max.index_select(
        2,
        query_to_kv_head,
    ).permute(0, 2, 1, 3)

    query_fp32 = query.float().unsqueeze(2)
    min_product = query_fp32 * query_page_min.float()
    max_product = query_fp32 * query_page_max.float()

    scores = torch.maximum(min_product, max_product).sum(dim=-1)

    return scores.masked_fill(
        ~valid_page_mask.unsqueeze(1),
        float("-inf"),
    )


def reference_select_pages(
    scores: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_budget: int,
    block_size: int,
) -> QuestPageSelection:
    """Select pages independently per query head.

    The current, partially filled page is always included. The remaining
    budget is used for Top-K selection over historical pages.
    """

    if scores.ndim != 3:
        raise ValueError(
            "scores must have shape "
            "[num_decode_tokens, num_query_heads, max_logical_pages]"
        )

    if page_budget <= 0:
        raise ValueError("page_budget must be positive")

    num_tokens, num_query_heads, max_logical_pages = scores.shape

    if block_table.shape != (num_tokens, max_logical_pages):
        raise ValueError(
            "block_table shape must match the token and page dimensions "
            "of scores"
        )

    if seq_lens.shape != (num_tokens,):
        raise ValueError(
            f"seq_lens must have shape {(num_tokens,)}, "
            f"got {seq_lens.shape}"
        )

    device = scores.device
    physical_block_table = block_table.to(
        device=device,
        dtype=torch.long,
    )
    seq_lens_device = seq_lens.to(device=device, dtype=torch.long)
    num_pages = _num_logical_pages(seq_lens_device, block_size)

    if bool(torch.any(num_pages > max_logical_pages).item()):
        raise ValueError(
            "block_table does not contain enough logical-page entries"
        )

    max_selected_pages = min(
        page_budget,
        int(num_pages.max().item()),
    )

    logical_page_ids = torch.full(
        (num_tokens, num_query_heads, max_selected_pages),
        -1,
        dtype=torch.long,
        device=device,
    )
    physical_page_ids = torch.full_like(logical_page_ids, -1)
    num_selected_pages = torch.zeros(
        (num_tokens, num_query_heads),
        dtype=torch.int32,
        device=device,
    )

    for token_index in range(num_tokens):
        request_num_pages = int(num_pages[token_index].item())
        current_logical_page = request_num_pages - 1
        request_budget = min(page_budget, request_num_pages)
        historical_budget = request_budget - 1

        for query_head in range(num_query_heads):
            if historical_budget == current_logical_page:
                # Full historical budget: retain logical order. This makes
                # full-budget attention directly comparable with dense
                # attention.
                historical_pages = torch.arange(
                    current_logical_page,
                    device=device,
                )
            elif historical_budget > 0:
                historical_pages = torch.topk(
                    scores[
                        token_index,
                        query_head,
                        :current_logical_page,
                    ],
                    k=historical_budget,
                    largest=True,
                    sorted=False,
                ).indices

                # Page order does not change attention mathematically, but
                # logical ordering produces deterministic reference results.
                historical_pages = torch.sort(historical_pages).values
            else:
                historical_pages = torch.empty(
                    0,
                    dtype=torch.long,
                    device=device,
                )

            selected_logical_pages = torch.cat(
                (
                    historical_pages,
                    torch.tensor(
                        [current_logical_page],
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )

            selected_count = selected_logical_pages.numel()
            selected_physical_pages = physical_block_table[
                token_index
            ].index_select(0, selected_logical_pages)

            logical_page_ids[
                token_index,
                query_head,
                :selected_count,
            ] = selected_logical_pages

            physical_page_ids[
                token_index,
                query_head,
                :selected_count,
            ] = selected_physical_pages

            num_selected_pages[token_index, query_head] = selected_count

    return QuestPageSelection(
        logical_page_ids=logical_page_ids,
        physical_page_ids=physical_page_ids,
        num_selected_pages=num_selected_pages,
    )


def reference_selected_page_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selection: QuestPageSelection,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute exact attention over selected Quest pages.

    This implementation is intentionally slow and readable. It gathers the
    selected K/V tokens into temporary tensors and serves only as a
    correctness reference.

    Args:
        query:
            Decode queries shaped
            [num_decode_tokens, num_query_heads, head_size].
        kv_cache:
            Paged KV cache shaped
            [num_physical_pages, 2, block_size, num_kv_heads, head_size].
        selection:
            Per-query-head logical and physical page selections.
        seq_lens:
            Sequence lengths including the current decode token.
        block_size:
            Number of tokens per physical KV page.
        scale:
            Optional attention scale. Defaults to head_size**-0.5.

    Returns:
        Attention output shaped like query.
    """

    if query.ndim != 3:
        raise ValueError(
            "query must have shape "
            "[num_decode_tokens, num_query_heads, head_size]"
        )

    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "kv_cache must have shape "
            "[num_pages, 2, block_size, num_kv_heads, head_size]"
        )

    num_tokens, num_query_heads, head_size = query.shape
    (
        num_physical_pages,
        _,
        cache_block_size,
        num_kv_heads,
        cache_head_size,
    ) = kv_cache.shape

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

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_query_heads} query heads cannot be evenly mapped to "
            f"{num_kv_heads} KV heads"
        )

    if seq_lens.shape != (num_tokens,):
        raise ValueError(
            f"seq_lens must have shape {(num_tokens,)}, "
            f"got {seq_lens.shape}"
        )

    expected_selection_shape = (
        num_tokens,
        num_query_heads,
        selection.logical_page_ids.shape[-1],
    )

    if selection.logical_page_ids.shape != expected_selection_shape:
        raise ValueError("logical page selection has an invalid shape")

    if selection.physical_page_ids.shape != expected_selection_shape:
        raise ValueError("physical page selection has an invalid shape")

    if selection.num_selected_pages.shape != (
        num_tokens,
        num_query_heads,
    ):
        raise ValueError("num_selected_pages has an invalid shape")

    if query.device != kv_cache.device:
        raise ValueError("query and kv_cache must be on the same device")

    device = query.device
    logical_page_ids = selection.logical_page_ids.to(
        device=device,
        dtype=torch.long,
    )
    physical_page_ids = selection.physical_page_ids.to(
        device=device,
        dtype=torch.long,
    )
    selected_counts = selection.num_selected_pages.to(
        device=device,
        dtype=torch.long,
    )
    seq_lens_device = seq_lens.to(device=device, dtype=torch.long)

    queries_per_kv_head = num_query_heads // num_kv_heads
    attention_scale = (
        float(scale)
        if scale is not None
        else float(head_size) ** -0.5
    )

    output = torch.empty_like(query)

    for token_index in range(num_tokens):
        sequence_length = int(seq_lens_device[token_index].item())

        if sequence_length <= 0:
            raise ValueError("Every decode sequence must contain a token")

        for query_head in range(num_query_heads):
            selected_count = int(
                selected_counts[token_index, query_head].item()
            )

            if selected_count <= 0:
                raise ValueError(
                    "Every query head must select at least one page"
                )

            kv_head = query_head // queries_per_kv_head
            selected_keys: list[torch.Tensor] = []
            selected_values: list[torch.Tensor] = []

            for selected_index in range(selected_count):
                logical_page = int(
                    logical_page_ids[
                        token_index,
                        query_head,
                        selected_index,
                    ].item()
                )
                physical_page = int(
                    physical_page_ids[
                        token_index,
                        query_head,
                        selected_index,
                    ].item()
                )

                if logical_page < 0:
                    raise ValueError("Selected logical page ID is negative")

                if not 0 <= physical_page < num_physical_pages:
                    raise ValueError(
                        "Selected physical page ID is out of range"
                    )

                valid_tokens = min(
                    block_size,
                    sequence_length - logical_page * block_size,
                )

                if valid_tokens <= 0:
                    raise ValueError(
                        "Selected logical page is outside the sequence"
                    )

                selected_keys.append(
                    kv_cache[
                        physical_page,
                        0,
                        :valid_tokens,
                        kv_head,
                    ].float()
                )
                selected_values.append(
                    kv_cache[
                        physical_page,
                        1,
                        :valid_tokens,
                        kv_head,
                    ].float()
                )

            keys = torch.cat(selected_keys, dim=0)
            values = torch.cat(selected_values, dim=0)
            query_vector = query[token_index, query_head].float()

            logits = torch.mv(keys, query_vector) * attention_scale
            probabilities = torch.softmax(logits, dim=0)
            attention_output = torch.matmul(probabilities, values)

            output[token_index, query_head].copy_(
                attention_output.to(dtype=query.dtype)
            )

    return output


@dataclass(frozen=True)
class QuestReferenceResult:
    """Complete output of the PyTorch Quest decode reference."""

    output: torch.Tensor
    scores: torch.Tensor
    selection: QuestPageSelection


def reference_quest_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_budget: int,
    block_size: int,
    scale: float | None = None,
) -> QuestReferenceResult:
    """Run the complete PyTorch Quest decode pipeline.

    This function is a correctness oracle. It is not intended for
    performance-sensitive inference.
    """

    scores = reference_score_pages(
        query=query,
        page_min=page_min,
        page_max=page_max,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
    )

    selection = reference_select_pages(
        scores=scores,
        block_table=block_table,
        seq_lens=seq_lens,
        page_budget=page_budget,
        block_size=block_size,
    )

    output = reference_selected_page_attention(
        query=query,
        kv_cache=kv_cache,
        selection=selection,
        seq_lens=seq_lens,
        block_size=block_size,
        scale=scale,
    )

    return QuestReferenceResult(
        output=output,
        scores=scores,
        selection=selection,
    )
