# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.attention.backends.quest import compute_quest_budget


@pytest.mark.parametrize("sequence_length", [1, 8192])
def test_budget_is_full_at_or_below_threshold(
    sequence_length: int,
) -> None:
    result = compute_quest_budget(
        sequence_length=sequence_length,
        block_size=16,
        budget_percent=1.0,
        sparse_start_tokens=8192,
    )

    assert result.selected_pages == result.valid_pages
    assert not result.sparse_active
    assert result.effective_percent == 100.0


@pytest.mark.parametrize(
    ("budget_percent", "expected_pages"),
    [
        (1.0, 6),
        (5.0, 26),
        (10.0, 52),
        (100.0, 513),
    ],
)
def test_budget_above_threshold(
    budget_percent: float,
    expected_pages: int,
) -> None:
    result = compute_quest_budget(
        sequence_length=8193,
        block_size=16,
        budget_percent=budget_percent,
        sparse_start_tokens=8192,
    )

    assert result.valid_pages == 513
    assert result.selected_pages == expected_pages
    assert result.sparse_active == (budget_percent < 100.0)


def test_budget_never_selects_zero_pages() -> None:
    result = compute_quest_budget(
        sequence_length=17,
        block_size=16,
        budget_percent=1.0,
        sparse_start_tokens=0,
    )

    assert result.valid_pages == 2
    assert result.selected_pages == 1
    assert result.sparse_active


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence_length", 0),
        ("block_size", 0),
        ("budget_percent", 0.0),
        ("budget_percent", 101.0),
        ("sparse_start_tokens", -1),
    ],
)
def test_invalid_budget_inputs(
    field: str,
    value: int | float,
) -> None:
    arguments: dict[str, int | float] = {
        "sequence_length": 8193,
        "block_size": 16,
        "budget_percent": 5.0,
        "sparse_start_tokens": 8192,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        compute_quest_budget(**arguments)
