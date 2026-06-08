from __future__ import annotations


def assert_shape(actual, expected, name: str) -> None:
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple != expected_tuple:
        raise AssertionError(f"{name} shape {actual_tuple} != {expected_tuple}")
