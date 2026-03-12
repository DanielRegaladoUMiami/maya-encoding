"""
Utility functions for input validation and preprocessing.

Handles conversion of negative numbers, floating-point values, and input validation
for the Maya encoding pipeline.
"""

from __future__ import annotations

import numpy as np


def validate_input(X) -> np.ndarray:
    """Convert input to a 2D numpy array and validate.

    Accepts numpy arrays, pandas DataFrames/Series, and lists.

    Parameters
    ----------
    X : array-like
        Input data. Can be 1D (single feature) or 2D (multiple features).

    Returns
    -------
    np.ndarray
        2D numpy float64 array of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If input contains NaN or infinity values.
    """
    # Handle pandas objects
    try:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
    except ImportError:
        pass

    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"Input must be 1D or 2D, got {X.ndim}D array.")

    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values. Please handle missing data first.")

    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinity values. Please handle infinite values first.")

    return X


def handle_negatives(
    values: np.ndarray, strategy: str = "abs_sign"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Handle negative values according to the specified strategy.

    Parameters
    ----------
    values : np.ndarray
        1D array of numeric values (may contain negatives).
    strategy : str
        How to handle negatives:
        - 'abs_sign': Return absolute values and a sign array (0=positive, 1=negative)
        - 'shift': Shift all values so minimum is 0
        - 'error': Raise ValueError if negatives found

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        (processed_values, sign_array)
        sign_array is None unless strategy='abs_sign' and negatives exist.

    Raises
    ------
    ValueError
        If strategy='error' and negative values are found.
    """
    has_negatives = np.any(values < 0)

    if not has_negatives:
        return values, None

    if strategy == "error":
        raise ValueError(
            "Negative values found but handle_negative='error'. "
            "Use 'abs_sign' or 'shift' to handle negatives."
        )
    elif strategy == "shift":
        min_val = np.min(values)
        return values - min_val, None
    elif strategy == "abs_sign":
        signs = (values < 0).astype(np.float64)
        return np.abs(values), signs
    else:
        raise ValueError(f"Unknown negative strategy: '{strategy}'. Use 'abs_sign', 'shift', 'error'.")


def handle_floats(
    values: np.ndarray, strategy: str = "scale", scale_factor: int | str = "auto"
) -> tuple[np.ndarray, int]:
    """Convert floating-point values to integers for vigesimal encoding.

    Parameters
    ----------
    values : np.ndarray
        1D array of non-negative float values.
    strategy : str
        How to handle floats:
        - 'scale': Multiply by scale_factor and round to integer
        - 'round': Round to nearest integer
        - 'integer_part': Take floor (discard fractional part)
    scale_factor : int or 'auto'
        Factor to multiply by when strategy='scale'.
        If 'auto', detected from decimal precision of the data.

    Returns
    -------
    tuple[np.ndarray, int]
        (integer_values, scale_factor_used)
    """
    if strategy == "round":
        return np.round(values).astype(np.int64), 1
    elif strategy == "integer_part":
        return np.floor(values).astype(np.int64), 1
    elif strategy == "scale":
        if scale_factor == "auto":
            scale_factor = auto_scale_factor(values)
        scale_factor = int(scale_factor)
        scaled = np.round(values * scale_factor).astype(np.int64)
        return scaled, scale_factor
    else:
        raise ValueError(
            f"Unknown float strategy: '{strategy}'. Use 'scale', 'round', 'integer_part'."
        )


def auto_scale_factor(values: np.ndarray) -> int:
    """Detect the optimal scale factor based on decimal precision in data.

    Examines the data to find the maximum number of decimal places used,
    then returns 10^(decimal_places) as the scale factor.

    Parameters
    ----------
    values : np.ndarray
        1D array of float values.

    Returns
    -------
    int
        Scale factor (power of 10). Minimum 1, maximum 10000.

    Examples
    --------
    >>> auto_scale_factor(np.array([1.5, 2.0, 3.25]))
    100
    >>> auto_scale_factor(np.array([1.0, 2.0, 3.0]))
    1
    >>> auto_scale_factor(np.array([1.123, 2.456]))
    1000
    """
    # Check if all values are effectively integers
    if np.allclose(values, np.round(values)):
        return 1

    # Sample up to 1000 values for efficiency
    sample = values[:1000] if len(values) > 1000 else values

    max_decimals = 0
    for val in sample:
        if val == 0 or np.isnan(val):
            continue
        # Convert to string and count decimal places
        s = f"{val:.10f}".rstrip("0")
        if "." in s:
            decimals = len(s.split(".")[1])
            max_decimals = max(max_decimals, decimals)

    # Cap at 4 decimal places (scale factor 10000)
    max_decimals = min(max_decimals, 4)
    return max(1, 10 ** max_decimals)


def get_feature_names(
    col_name: str, n_levels: int, components: str, has_sign: bool = False
) -> list[str]:
    """Generate descriptive feature names for a VFD-encoded column.

    Parameters
    ----------
    col_name : str
        Original column name.
    n_levels : int
        Number of vigesimal levels.
    components : str
        Component mode: 'full', 'lite', or 'bars_dots'.
    has_sign : bool
        Whether a sign feature is included.

    Returns
    -------
    list[str]
        List of feature names.
    """
    names = []

    if has_sign:
        names.append(f"{col_name}_sign")

    for level in range(n_levels):
        if components == "lite":
            names.append(f"{col_name}_L{level}_digit")
        elif components == "bars_dots":
            names.append(f"{col_name}_L{level}_bars")
            names.append(f"{col_name}_L{level}_dots")
        else:  # full
            names.append(f"{col_name}_L{level}_digit")
            names.append(f"{col_name}_L{level}_bars")
            names.append(f"{col_name}_L{level}_dots")

    return names
