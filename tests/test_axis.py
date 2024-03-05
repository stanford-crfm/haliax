import pytest

from haliax.axis import Axis, eliminate_axes, rearrange_for_partial_order


def test_eliminate_axes():
    H = Axis("H", 3)
    W = Axis("W", 4)
    C = Axis("C", 5)

    assert eliminate_axes((H, W), (H,)) == (W,)
    assert eliminate_axes((H, W), (W,)) == (H,)
    assert eliminate_axes((H, W), (H, W)) == ()

    with pytest.raises(ValueError):
        eliminate_axes((H, W), (C,))

    with pytest.raises(ValueError):
        eliminate_axes((H, W), (H, C))

    # test string references
    assert eliminate_axes((H, W), ("H",)) == (W,)
    assert eliminate_axes(("H", W), (H,)) == (W,)
    assert eliminate_axes(("H", W), ("H",)) == (W,)
    assert eliminate_axes(("H", W), ("H", "W")) == ()


def assert_partial_order_respected(partial_order, output):
    positions = {el: i for i, el in enumerate(output)}

    last_pos = -1
    for el in partial_order:
        if el is ...:
            # Reset last_pos for flexible positions
            last_pos = -1
        else:
            # Check if the element is in the correct order
            assert el in positions, f"{el} is missing in the output"
            assert positions[el] > last_pos, f"Partial order not respected for {el}"
            last_pos = positions[el]


def test_basic_order():
    partial_order = ("apple", ..., "banana")
    candidates = ("banana", "apple", "cherry")
    expected_output = ("apple", "cherry", "banana")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert actual_output == expected_output
    assert_partial_order_respected(partial_order, actual_output)


def test_start_with_ellipsis():
    partial_order = (..., "apple", "banana")
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("cherry", "apple", "banana")


def test_end_with_ellipsis():
    partial_order = ("apple", ..., "banana", ...)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)

    # this one could be either but we'll assert the order so we catch changes
    assert actual_output == ("apple", "banana", "cherry")


def test_full_flexibility():
    partial_order = (...,)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)


def test_no_flexibility():
    partial_order = ("apple", "banana")
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)


def test_final_ellipsis():
    partial_order = ("apple", "banana", ...)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "cherry")


def test_lots_of_ellipsis():
    partial_order = ("apple", ..., "banana", ..., "cherry", ...)
    candidates = ("banana", "orange", "cherry", "apple", "grape")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "orange", "cherry", "grape")


def test_no_ellipsis():
    partial_order = ("apple", "banana", "cherry")
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "cherry")


def test_no_elements():
    partial_order = (...,)
    candidates = ()
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ()


def test_missing_elements_errors():
    partial_order = ("qux", ...)
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)


def test_duplicate_elements_errors():
    partial_order: tuple = ("apple", "apple", ...)
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)

    candidates = ("banana", "apple", "apple")

    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)

    partial_order = ("apple", "banana", "grape", ...)

    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)
