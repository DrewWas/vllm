# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.quest.reference import (
    reference_score_pages,
    reference_select_pages,
)


def test_score_pages_uses_gqa_and_physical_block_table() -> None:
    # Four query heads map onto two KV heads:
    # Q heads 0-1 -> KV head 0
    # Q heads 2-3 -> KV head 1
    query = torch.tensor(
        [
            [
                [2.0, -1.0],
                [-1.0, 3.0],
                [1.0, 1.0],
                [-2.0, -1.0],
            ]
        ]
    )

    page_min = torch.zeros((3, 2, 2))
    page_max = torch.zeros((3, 2, 2))

    # Logical page 0 maps to physical page 2.
    page_min[2] = torch.tensor(
        [
            [1.0, -2.0],
            [-1.0, 5.0],
        ]
    )
    page_max[2] = torch.tensor(
        [
            [3.0, 4.0],
            [2.0, 6.0],
        ]
    )

    # Logical page 1 maps to physical page 0.
    page_min[0] = torch.tensor(
        [
            [-4.0, 1.0],
            [0.0, -3.0],
        ]
    )
    page_max[0] = torch.tensor(
        [
            [-2.0, 5.0],
            [7.0, -1.0],
        ]
    )

    scores = reference_score_pages(
        query=query,
        page_min=page_min,
        page_max=page_max,
        block_table=torch.tensor([[2, 0, -1]]),
        seq_lens=torch.tensor([4]),
        block_size=2,
    )

    expected = torch.tensor(
        [
            [
                [8.0, -5.0, float("-inf")],
                [11.0, 19.0, float("-inf")],
                [8.0, 6.0, float("-inf")],
                [-3.0, 3.0, float("-inf")],
            ]
        ]
    )

    torch.testing.assert_close(scores, expected)


def test_select_pages_is_independent_per_query_head() -> None:
    scores = torch.tensor(
        [
            [
                [1.0, 9.0, 2.0, -100.0],
                [8.0, 1.0, 7.0, -100.0],
            ]
        ]
    )

    selection = reference_select_pages(
        scores=scores,
        block_table=torch.tensor([[4, 7, 2, 9]]),
        seq_lens=torch.tensor([7]),
        page_budget=2,
        block_size=2,
    )

    # Each query head chooses a different historical page, but both must
    # include logical page 3, the current partially filled page.
    expected_logical = torch.tensor(
        [
            [
                [1, 3],
                [0, 3],
            ]
        ]
    )
    expected_physical = torch.tensor(
        [
            [
                [7, 9],
                [4, 9],
            ]
        ]
    )

    torch.testing.assert_close(
        selection.logical_page_ids,
        expected_logical,
    )
    torch.testing.assert_close(
        selection.physical_page_ids,
        expected_physical,
    )
    torch.testing.assert_close(
        selection.num_selected_pages,
        torch.tensor([[2, 2]], dtype=torch.int32),
    )


def test_select_pages_always_includes_current_page() -> None:
    scores = torch.tensor(
        [
            [
                [10.0, 9.0, 8.0, -1_000_000.0],
            ]
        ]
    )

    selection = reference_select_pages(
        scores=scores,
        block_table=torch.tensor([[5, 6, 7, 8]]),
        seq_lens=torch.tensor([7]),
        page_budget=2,
        block_size=2,
    )

    # Current logical page 3 is selected even though its score is lowest.
    torch.testing.assert_close(
        selection.logical_page_ids,
        torch.tensor([[[0, 3]]]),
    )
    torch.testing.assert_close(
        selection.physical_page_ids,
        torch.tensor([[[5, 8]]]),
    )


def test_full_budget_preserves_all_pages_in_logical_order() -> None:
    scores = torch.tensor(
        [
            [
                [1.0, 100.0, -20.0],
                [50.0, -10.0, 5.0],
            ]
        ]
    )

    selection = reference_select_pages(
        scores=scores,
        block_table=torch.tensor([[8, 3, 6]]),
        seq_lens=torch.tensor([5]),
        page_budget=100,
        block_size=2,
    )

    expected_logical = torch.tensor(
        [
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        ]
    )
    expected_physical = torch.tensor(
        [
            [
                [8, 3, 6],
                [8, 3, 6],
            ]
        ]
    )

    torch.testing.assert_close(
        selection.logical_page_ids,
        expected_logical,
    )
    torch.testing.assert_close(
        selection.physical_page_ids,
        expected_physical,
    )
    torch.testing.assert_close(
        selection.num_selected_pages,
        torch.tensor([[3, 3]], dtype=torch.int32),
    )
