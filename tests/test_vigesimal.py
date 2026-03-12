"""Tests for the vigesimal number system core functions."""

import numpy as np
import pytest

from maya_encoding.core.vigesimal import (
    auto_n_levels,
    from_vigesimal,
    maya_decompose,
    maya_encode_array,
    to_bars_dots,
    to_vigesimal,
)


class TestAutoNLevels:
    def test_zero(self):
        assert auto_n_levels(0) == 1

    def test_small(self):
        assert auto_n_levels(19) == 1

    def test_boundary_20(self):
        assert auto_n_levels(20) == 2

    def test_boundary_400(self):
        assert auto_n_levels(399) == 2
        assert auto_n_levels(400) == 3

    def test_large(self):
        assert auto_n_levels(160000) >= 4


class TestToVigesimal:
    def test_zero(self):
        assert to_vigesimal(0) == [0]

    def test_single_digit(self):
        assert to_vigesimal(7) == [7]
        assert to_vigesimal(19) == [19]

    def test_two_digits(self):
        assert to_vigesimal(20) == [0, 1]
        assert to_vigesimal(347) == [7, 17]
        assert to_vigesimal(399) == [19, 19]

    def test_three_digits(self):
        assert to_vigesimal(400) == [0, 0, 1]
        assert to_vigesimal(8000) == [0, 0, 0, 1]

    def test_fixed_levels(self):
        assert to_vigesimal(7, n_levels=3) == [7, 0, 0]
        assert to_vigesimal(347, n_levels=4) == [7, 17, 0, 0]

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            to_vigesimal(-1)


class TestFromVigesimal:
    def test_zero(self):
        assert from_vigesimal([0]) == 0

    def test_single(self):
        assert from_vigesimal([7]) == 7
        assert from_vigesimal([19]) == 19

    def test_multi(self):
        assert from_vigesimal([7, 17]) == 347
        assert from_vigesimal([0, 1]) == 20
        assert from_vigesimal([19, 19]) == 399

    def test_invalid_digit(self):
        with pytest.raises(ValueError, match="\\[0, 19\\]"):
            from_vigesimal([20])

    @pytest.mark.parametrize("n", [0, 1, 19, 20, 100, 347, 399, 400, 7999, 8000, 160000])
    def test_roundtrip(self, n):
        """Encode -> decode should be identity for all valid inputs."""
        levels = auto_n_levels(n)
        digits = to_vigesimal(n, levels)
        assert from_vigesimal(digits) == n


class TestToBarsDots:
    @pytest.mark.parametrize(
        "digit, expected_bars, expected_dots",
        [
            (0, 0, 0),
            (1, 0, 1),
            (4, 0, 4),
            (5, 1, 0),
            (7, 1, 2),
            (10, 2, 0),
            (15, 3, 0),
            (19, 3, 4),
        ],
    )
    def test_known_values(self, digit, expected_bars, expected_dots):
        bars, dots = to_bars_dots(digit)
        assert bars == expected_bars
        assert dots == expected_dots

    def test_all_digits_exhaustive(self):
        """Every digit 0-19 should produce valid bars and dots."""
        for d in range(20):
            bars, dots = to_bars_dots(d)
            assert 0 <= bars <= 3
            assert 0 <= dots <= 4
            assert bars * 5 + dots == d

    def test_invalid(self):
        with pytest.raises(ValueError):
            to_bars_dots(-1)
        with pytest.raises(ValueError):
            to_bars_dots(20)


class TestMayaDecompose:
    def test_zero(self):
        result = maya_decompose(0)
        assert result["digits"] == [0]
        assert result["bars"] == [0]
        assert result["dots"] == [0]
        assert result["n_levels"] == 1

    def test_347(self):
        result = maya_decompose(347)
        assert result["digits"] == [7, 17]
        assert result["bars"] == [1, 3]
        assert result["dots"] == [2, 2]
        assert result["n_levels"] == 2

    def test_with_levels(self):
        result = maya_decompose(7, n_levels=3)
        assert result["n_levels"] == 3
        assert len(result["digits"]) == 3


class TestMayaEncodeArray:
    def test_basic(self):
        values = np.array([0, 7, 20, 347])
        result = maya_encode_array(values, n_levels=2, components="full", normalize=False)
        assert result.shape == (4, 6)  # 2 levels * 3 components

        # Check 347: digits=[7,17], bars=[1,3], dots=[2,2]
        np.testing.assert_array_equal(result[3], [7, 1, 2, 17, 3, 2])

    def test_normalized(self):
        values = np.array([19])
        result = maya_encode_array(values, n_levels=1, components="full", normalize=True)
        np.testing.assert_allclose(result[0], [19 / 19, 3 / 3, 4 / 4])

    def test_lite(self):
        values = np.array([347])
        result = maya_encode_array(values, n_levels=2, components="lite", normalize=False)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], [7, 17])

    def test_bars_dots(self):
        values = np.array([347])
        result = maya_encode_array(values, n_levels=2, components="bars_dots", normalize=False)
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 2])

    def test_vectorized_matches_scalar(self):
        """Vectorized results should match scalar decomposition."""
        test_values = [0, 1, 19, 20, 100, 347, 399, 400, 7999]
        arr = np.array(test_values)
        result = maya_encode_array(arr, n_levels=3, components="full", normalize=False)

        for i, n in enumerate(test_values):
            decomp = maya_decompose(n, n_levels=3)
            expected = []
            for level in range(3):
                expected.extend([
                    decomp["digits"][level],
                    decomp["bars"][level],
                    decomp["dots"][level],
                ])
            np.testing.assert_array_equal(result[i], expected)
