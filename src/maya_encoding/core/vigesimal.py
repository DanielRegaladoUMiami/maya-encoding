"""
Core vigesimal (base-20) number system functions.

The Maya vigesimal system decomposes numbers into positions of value 1, 20, 400, 8000, ...
Each position (digit 0-19) further decomposes into bars (value 5, max 3) and dots (value 1, max 4).

This module provides pure functions for conversion, with numpy vectorization for performance.
"""

from __future__ import annotations

import math

import numpy as np


def auto_n_levels(max_value: int | float) -> int:
    """Calculate the minimum number of vigesimal levels needed to represent a value.

    Parameters
    ----------
    max_value : int or float
        The maximum absolute value to represent.

    Returns
    -------
    int
        Number of vigesimal levels needed (minimum 1).

    Examples
    --------
    >>> auto_n_levels(19)
    1
    >>> auto_n_levels(20)
    2
    >>> auto_n_levels(399)
    2
    >>> auto_n_levels(400)
    3
    """
    if max_value <= 0:
        return 1
    abs_val = abs(int(max_value))
    if abs_val == 0:
        return 1
    return max(1, math.floor(math.log(abs_val) / math.log(20)) + 1)


def to_vigesimal(n: int, n_levels: int | None = None) -> list[int]:
    """Convert a non-negative integer to a list of vigesimal digits (LSB first).

    Parameters
    ----------
    n : int
        Non-negative integer to convert.
    n_levels : int or None
        Number of vigesimal levels. If None, auto-detected.
        Result is zero-padded or truncated to this length.

    Returns
    -------
    list[int]
        List of vigesimal digits, least significant first.
        Each digit is in range [0, 19].

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> to_vigesimal(0)
    [0]
    >>> to_vigesimal(19)
    [19]
    >>> to_vigesimal(20)
    [0, 1]
    >>> to_vigesimal(347)
    [7, 17]
    >>> to_vigesimal(347, n_levels=4)
    [7, 17, 0, 0]
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}. Use utils.handle_negatives first.")

    if n_levels is None:
        n_levels = auto_n_levels(n)

    digits = []
    remaining = int(n)
    for _ in range(n_levels):
        digits.append(remaining % 20)
        remaining //= 20

    return digits


def from_vigesimal(digits: list[int]) -> int:
    """Convert a list of vigesimal digits (LSB first) back to an integer.

    Parameters
    ----------
    digits : list[int]
        Vigesimal digits, least significant first. Each must be in [0, 19].

    Returns
    -------
    int
        The reconstructed integer.

    Raises
    ------
    ValueError
        If any digit is outside [0, 19].

    Examples
    --------
    >>> from_vigesimal([7, 17])
    347
    >>> from_vigesimal([0, 1])
    20
    >>> from_vigesimal([0])
    0
    """
    result = 0
    for i, d in enumerate(digits):
        if not 0 <= d <= 19:
            raise ValueError(f"Vigesimal digit must be in [0, 19], got {d} at position {i}.")
        result += d * (20 ** i)
    return result


def to_bars_dots(digit: int) -> tuple[int, int]:
    """Decompose a vigesimal digit (0-19) into bars and dots.

    In the Maya system:
    - Each bar represents 5
    - Each dot represents 1
    - Maximum: 3 bars (15) + 4 dots (4) = 19

    Parameters
    ----------
    digit : int
        A vigesimal digit in range [0, 19].

    Returns
    -------
    tuple[int, int]
        (bars, dots) where bars in [0, 3] and dots in [0, 4].

    Raises
    ------
    ValueError
        If digit is outside [0, 19].

    Examples
    --------
    >>> to_bars_dots(0)
    (0, 0)
    >>> to_bars_dots(7)
    (1, 2)
    >>> to_bars_dots(19)
    (3, 4)
    """
    if not 0 <= digit <= 19:
        raise ValueError(f"Vigesimal digit must be in [0, 19], got {digit}.")
    return digit // 5, digit % 5


def maya_decompose(n: int, n_levels: int | None = None) -> dict:
    """Full Maya decomposition of a non-negative integer.

    Returns the vigesimal digits, bars, and dots for each level.

    Parameters
    ----------
    n : int
        Non-negative integer to decompose.
    n_levels : int or None
        Number of vigesimal levels. If None, auto-detected.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'digits': list[int] — vigesimal digits (LSB first)
        - 'bars': list[int] — bars component per level
        - 'dots': list[int] — dots component per level
        - 'n_levels': int — number of levels used

    Examples
    --------
    >>> maya_decompose(347)
    {'digits': [7, 17], 'bars': [1, 3], 'dots': [2, 2], 'n_levels': 2}
    >>> maya_decompose(0)
    {'digits': [0], 'bars': [0], 'dots': [0], 'n_levels': 1}
    """
    digits = to_vigesimal(n, n_levels)
    bars = []
    dots = []
    for d in digits:
        b, dt = to_bars_dots(d)
        bars.append(b)
        dots.append(dt)

    return {
        "digits": digits,
        "bars": bars,
        "dots": dots,
        "n_levels": len(digits),
    }


def maya_encode_array(
    values: np.ndarray,
    n_levels: int,
    components: str = "full",
    normalize: bool = True,
) -> np.ndarray:
    """Vectorized Maya encoding for a 1D array of non-negative integers.

    Parameters
    ----------
    values : np.ndarray
        1D array of non-negative integers.
    n_levels : int
        Number of vigesimal levels to use.
    components : str
        Which components to include:
        - 'full': digits, bars, and dots (3 features per level)
        - 'lite': digits only (1 feature per level)
        - 'bars_dots': bars and dots only (2 features per level)
    normalize : bool
        If True, normalize each component to [0, 1].

    Returns
    -------
    np.ndarray
        2D array of shape (len(values), n_features).
        n_features = n_levels * features_per_level.
    """
    values = np.asarray(values, dtype=np.int64).ravel()
    n = len(values)

    # Compute vigesimal digits for all levels (vectorized)
    all_digits = np.zeros((n, n_levels), dtype=np.int64)
    remaining = values.copy()
    for level in range(n_levels):
        all_digits[:, level] = remaining % 20
        remaining //= 20

    # Compute bars and dots (vectorized)
    all_bars = all_digits // 5
    all_dots = all_digits % 5

    # Assemble output based on components
    if components == "lite":
        result = all_digits.astype(np.float64)
        if normalize:
            result /= 19.0
    elif components == "bars_dots":
        result = np.zeros((n, n_levels * 2), dtype=np.float64)
        for level in range(n_levels):
            result[:, level * 2] = all_bars[:, level]
            result[:, level * 2 + 1] = all_dots[:, level]
        if normalize:
            result[:, 0::2] /= 3.0  # bars: [0, 3]
            result[:, 1::2] /= 4.0  # dots: [0, 4]
    else:  # 'full'
        result = np.zeros((n, n_levels * 3), dtype=np.float64)
        for level in range(n_levels):
            result[:, level * 3] = all_digits[:, level]
            result[:, level * 3 + 1] = all_bars[:, level]
            result[:, level * 3 + 2] = all_dots[:, level]
        if normalize:
            result[:, 0::3] /= 19.0  # digits: [0, 19]
            result[:, 1::3] /= 3.0   # bars: [0, 3]
            result[:, 2::3] /= 4.0   # dots: [0, 4]

    return result
