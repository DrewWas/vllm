# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.quest.reference import (
    reference_score_pages,
    reference_select_pages,
    reference_selected_page_attention,
    reference_quest_decode,
)


def dense_attention_from_paged_cache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Independent dense decode oracle for the reference tests."""

    num_tokens, num_query_heads, head_size = query.shape
    num_kv_heads = kv_cache.shape[3]
    queries_per_kv_head = num_query_heads // num_kv_heads
    scale = head_size**-0.5

    output = torch.empty_like(query)

    for token_index in range(num_tokens):
        seq_len = int(seq_lens[token_index].item())
        num_pages = (seq_len + block_size - 1) // block_size

        page_keys = []
        page_values = []

        for logical_page in range(num_pages):
            physical_page = int(
                block_table[token_index, logical_page].item()
            )
            valid_tokens = min(
                block_size,
                seq_len - logical_page * block_size,
            )

            page_keys.append(
                kv_cache[
                    physical_page,
                    0,
                    :valid_tokens,
                ].float()
            )
            page_values.append(
                kv_cache[
                    physical_page,
                    1,
                    :valid_tokens,
                ].float()
            )

        keys = torch.cat(page_keys, dim=0)
        values = torch.cat(page_values, dim=0)

        for query_head in range(num_query_heads):
            kv_head = query_head // queries_per_kv_head
            query_vector = query[token_index, query_head].float()

            logits = torch.mv(
                keys[:, kv_head],
                query_vector,
            ) * scale

            probabilities = torch.softmax(logits, dim=0)

            output[token_index, query_head].copy_(
                torch.matmul(
                    probabilities,
                    values[:, kv_head],
                ).to(query.dtype)
            )

    return output


def test_full_budget_matches_dense_attention() -> None:
    torch.manual_seed(0)

    block_size = 2
    query = torch.randn(1, 4, 3)

    kv_cache = torch.randn(
        4,
        2,
        block_size,
        2,
        3,
    )

    # Logical pages map to shuffled physical pages.
    block_table = torch.tensor([[3, 0, 2, -1]])
    seq_lens = torch.tensor([5])

    # This is the unused slot in the final logical page. Large values make
    # incorrect masking obvious.
    kv_cache[2, :, 1].fill_(10_000)

    page_min = kv_cache[:, 0].amin(dim=1)
    page_max = kv_cache[:, 0].amax(dim=1)

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
        page_budget=100,
        block_size=block_size,
    )

    quest_output = reference_selected_page_attention(
        query=query,
        kv_cache=kv_cache,
        selection=selection,
        seq_lens=seq_lens,
        block_size=block_size,
    )

    dense_output = dense_attention_from_paged_cache(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
    )

    torch.testing.assert_close(
        quest_output,
        dense_output,
        rtol=1e-5,
        atol=1e-5,
    )


def test_selected_attention_uses_correct_gqa_head() -> None:
    query = torch.ones((1, 4, 1))

    kv_cache = torch.zeros((1, 2, 2, 2, 1))

    # KV head 0 values average to 15.
    kv_cache[0, 1, :, 0, 0] = torch.tensor([10.0, 20.0])

    # KV head 1 values average to 150.
    kv_cache[0, 1, :, 1, 0] = torch.tensor([100.0, 200.0])

    scores = torch.zeros((1, 4, 1))
    block_table = torch.tensor([[0]])
    seq_lens = torch.tensor([2])

    selection = reference_select_pages(
        scores=scores,
        block_table=block_table,
        seq_lens=seq_lens,
        page_budget=1,
        block_size=2,
    )

    output = reference_selected_page_attention(
        query=query,
        kv_cache=kv_cache,
        selection=selection,
        seq_lens=seq_lens,
        block_size=2,
    )

    expected = torch.tensor(
        [[[15.0], [15.0], [150.0], [150.0]]]
    )

    torch.testing.assert_close(output, expected)


def test_sparse_attention_ignores_unselected_pages() -> None:
    query = torch.ones((1, 1, 1))

    kv_cache = torch.zeros((3, 2, 1, 1, 1))

    # Equal keys produce equal logits. Page 1 has a huge value but should
    # not affect output because it is not selected.
    kv_cache[:, 1, 0, 0, 0] = torch.tensor(
        [1.0, 1_000.0, 3.0]
    )

    block_table = torch.tensor([[0, 1, 2]])
    seq_lens = torch.tensor([3])

    # Select historical page 0; current page 2 is appended automatically.
    scores = torch.tensor([[[10.0, 0.0, -1_000.0]]])

    selection = reference_select_pages(
        scores=scores,
        block_table=block_table,
        seq_lens=seq_lens,
        page_budget=2,
        block_size=1,
    )

    output = reference_selected_page_attention(
        query=query,
        kv_cache=kv_cache,
        selection=selection,
        seq_lens=seq_lens,
        block_size=1,
    )

    # Mean of selected values 1 and 3.
    torch.testing.assert_close(
        output,
        torch.tensor([[[2.0]]]),
    )


def test_reference_quest_decode_runs_full_pipeline() -> None:
    torch.manual_seed(1)

    block_size = 2
    query = torch.randn(1, 4, 3)
    kv_cache = torch.randn(4, 2, block_size, 2, 3)

    block_table = torch.tensor([[3, 0, 2, -1]])
    seq_lens = torch.tensor([5])

    # Unused value slot in the partially filled final page.
    kv_cache[2, 1, 1].fill_(10_000)

    page_min = kv_cache[:, 0].amin(dim=1)
    page_max = kv_cache[:, 0].amax(dim=1)

    result = reference_quest_decode(
        query=query,
        kv_cache=kv_cache,
        page_min=page_min,
        page_max=page_max,
        block_table=block_table,
        seq_lens=seq_lens,
        page_budget=100,
        block_size=block_size,
    )

    dense_output = dense_attention_from_paged_cache(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
    )

    torch.testing.assert_close(
        result.output,
        dense_output,
        rtol=1e-5,
        atol=1e-5,
    )

    assert result.scores.shape == (1, 4, 4)

    expected_pages = torch.tensor(
        [[[0, 1, 2]] * 4]
    )
    torch.testing.assert_close(
        result.selection.logical_page_ids,
        expected_pages,
    )
