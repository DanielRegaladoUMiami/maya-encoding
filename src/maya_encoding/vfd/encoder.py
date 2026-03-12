"""VFDEncoder: Vigesimal Feature Decomposition transformer for scikit-learn.

Converts numeric features into multi-scale Maya vigesimal representations,
providing models with hierarchical structure (digits, bars, dots) at no
information loss.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from maya_encoding.core.utils import (
    get_feature_names,
    handle_floats,
    validate_input,
)
from maya_encoding.core.vigesimal import auto_n_levels, maya_encode_array


class VFDEncoder(BaseEstimator, TransformerMixin):
    """Vigesimal Feature Decomposition encoder for numeric features.

    Decomposes numbers into the Maya vigesimal (base-20) system, extracting
    hierarchical features at three granularities:
    - **Digits** (0-19): the vigesimal position value
    - **Bars** (0-3): groups of 5 within each digit
    - **Dots** (0-4): remainder within each group of 5

    This gives the model multi-scale information about each number "for free",
    potentially improving performance on tasks where numerical magnitude and
    grouping patterns matter.

    Parameters
    ----------
    n_levels : int or 'auto', default='auto'
        Number of vigesimal levels (positions). Each level represents a power
        of 20. If 'auto', determined from the maximum value seen during fit().
        - 1 level covers [0, 19]
        - 2 levels cover [0, 399]
        - 3 levels cover [0, 7999]
        - 4 levels cover [0, 159999]

    components : str, default='full'
        Which components to extract per level:
        - 'full': digit + bars + dots (3 features per level)
        - 'lite': digit only (1 feature per level)
        - 'bars_dots': bars + dots only (2 features per level)

    normalize : bool, default=True
        If True, normalize each component to [0, 1]:
        - digits: /19, bars: /3, dots: /4

    handle_negative : str, default='abs_sign'
        How to handle negative input values:
        - 'abs_sign': Encode |value| and add a binary sign feature (0=pos, 1=neg)
        - 'shift': Shift all values so minimum is 0 (learned during fit)
        - 'error': Raise ValueError if negatives encountered

    handle_float : str, default='scale'
        How to convert floats to integers for vigesimal encoding:
        - 'scale': Multiply by scale_factor, then round
        - 'round': Round to nearest integer
        - 'integer_part': Take floor (discard fractional part)

    scale_factor : int or 'auto', default='auto'
        Multiplier for float-to-int conversion when handle_float='scale'.
        If 'auto', determined from decimal precision observed during fit().

    Attributes
    ----------
    n_levels_ : int
        Actual number of vigesimal levels used (resolved from 'auto').
    scale_factor_ : int
        Actual scale factor used (resolved from 'auto').
    n_features_in_ : int
        Number of input features seen during fit.
    shift_values_ : np.ndarray or None
        Per-feature shift values when handle_negative='shift'.
    has_negatives_ : np.ndarray
        Boolean array indicating which features had negative values.
    feature_names_in_ : list[str]
        Input feature names (from DataFrame columns or generated).

    Examples
    --------
    >>> import numpy as np
    >>> from maya_encoding import VFDEncoder
    >>> X = np.array([[347], [20], [0], [399]])
    >>> enc = VFDEncoder(n_levels=2, components='full', normalize=False)
    >>> enc.fit_transform(X)
    array([[ 7.,  1.,  2., 17.,  3.,  2.],
           [ 0.,  0.,  0.,  1.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [19.,  3.,  4., 19.,  3.,  4.]])

    """

    def __init__(
        self,
        n_levels: int | str = "auto",
        components: str = "full",
        normalize: bool = True,
        handle_negative: str = "abs_sign",
        handle_float: str = "scale",
        scale_factor: int | str = "auto",
    ):
        self.n_levels = n_levels
        self.components = components
        self.normalize = normalize
        self.handle_negative = handle_negative
        self.handle_float = handle_float
        self.scale_factor = scale_factor

    def fit(self, X, y=None):
        """Fit the encoder by learning parameters from the data.

        Determines n_levels, scale_factor, and shift values as needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used; present for sklearn compatibility.

        Returns
        -------
        self

        """
        X = validate_input(X)
        self.n_features_in_ = X.shape[1]

        # Store input feature names
        self.feature_names_in_ = self._get_input_names(X)

        # Per-feature: handle negatives
        self.has_negatives_ = np.any(X < 0, axis=0)
        self.shift_values_ = None

        if self.handle_negative == "shift" and np.any(self.has_negatives_):
            self.shift_values_ = np.where(
                self.has_negatives_, np.min(X, axis=0), 0.0
            )

        # Preprocess to get max values for auto-detection
        X_processed = self._preprocess(X, fitting=True)

        # Determine scale_factor
        has_floats = not np.allclose(X_processed, np.round(X_processed))
        if has_floats and self.handle_float == "scale":
            if self.scale_factor == "auto":
                from maya_encoding.core.utils import auto_scale_factor

                self.scale_factor_ = int(
                    max(auto_scale_factor(X_processed[:, i]) for i in range(X_processed.shape[1]))
                )
            else:
                self.scale_factor_ = int(self.scale_factor)
        else:
            self.scale_factor_ = int(self.scale_factor) if self.scale_factor != "auto" else 1

        # Convert to integers
        X_int = self._to_integers(X_processed)

        # Determine n_levels
        if self.n_levels == "auto":
            max_val = np.max(np.abs(X_int))
            self.n_levels_ = auto_n_levels(max_val)
        else:
            self.n_levels_ = int(self.n_levels)

        return self

    def transform(self, X):
        """Transform numeric features to VFD encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Transformed array of shape (n_samples, n_output_features).

        """
        check_is_fitted(self)
        X = validate_input(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_processed = self._preprocess(X, fitting=False)
        X_int = self._to_integers(X_processed)

        # Encode each feature column
        encoded_parts = []

        for i in range(self.n_features_in_):
            col_values = X_int[:, i]

            # Add sign feature if needed
            if self.handle_negative == "abs_sign" and self.has_negatives_[i]:
                signs = (X[:, i] < 0).astype(np.float64).reshape(-1, 1)
                encoded_parts.append(signs)

            # VFD encode
            col_encoded = maya_encode_array(
                col_values,
                n_levels=self.n_levels_,
                components=self.components,
                normalize=self.normalize,
            )
            encoded_parts.append(col_encoded)

        return np.hstack(encoded_parts)

    def inverse_transform(self, X_encoded):
        """Reconstruct original values from VFD encoding (approximate).

        Note: Reconstruction is exact only when normalize=False and no
        information was lost during float handling.

        Parameters
        ----------
        X_encoded : np.ndarray
            VFD-encoded data.

        Returns
        -------
        np.ndarray
            Reconstructed values of shape (n_samples, n_features_in).

        """
        check_is_fitted(self)
        X_encoded = np.asarray(X_encoded, dtype=np.float64)

        n_samples = X_encoded.shape[0]
        result = np.zeros((n_samples, self.n_features_in_))

        col_idx = 0
        features_per_level = {"full": 3, "lite": 1, "bars_dots": 2}[self.components]
        features_per_col = self.n_levels_ * features_per_level

        for i in range(self.n_features_in_):
            sign_offset = 0
            signs = None

            if self.handle_negative == "abs_sign" and self.has_negatives_[i]:
                signs = X_encoded[:, col_idx]
                sign_offset = 1

            end_idx = col_idx + sign_offset + features_per_col
            col_data = X_encoded[:, col_idx + sign_offset : end_idx]

            # Reconstruct from digits
            if self.components == "full":
                # Use digit components (every 3rd starting at 0)
                digit_features = col_data[:, 0::3]
                if self.normalize:
                    digit_features = digit_features * 19.0
            elif self.components == "lite":
                digit_features = col_data
                if self.normalize:
                    digit_features = digit_features * 19.0
            else:  # bars_dots
                bars = col_data[:, 0::2]
                dots = col_data[:, 1::2]
                if self.normalize:
                    bars = bars * 3.0
                    dots = dots * 4.0
                digit_features = np.round(bars) * 5 + np.round(dots)

            # Reconstruct integer value
            digits = np.round(digit_features).astype(np.int64)
            values = np.zeros(n_samples, dtype=np.float64)
            for level in range(self.n_levels_):
                values += digits[:, level] * (20 ** level)

            # Undo scaling
            if self.scale_factor_ > 1:
                values = values / self.scale_factor_

            # Undo sign
            if signs is not None:
                values = np.where(signs > 0.5, -values, values)

            # Undo shift
            if self.shift_values_ is not None and self.shift_values_[i] != 0:
                values += self.shift_values_[i]

            result[:, i] = values
            col_idx += sign_offset + features_per_col

        return result

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : ignored
            Not used; present for sklearn compatibility.

        Returns
        -------
        list[str]
            Output feature names.

        """
        check_is_fitted(self)
        names = []
        for i in range(self.n_features_in_):
            col_name = self.feature_names_in_[i]
            has_sign = self.handle_negative == "abs_sign" and self.has_negatives_[i]
            col_names = get_feature_names(
                col_name, self.n_levels_, self.components, has_sign
            )
            names.extend(col_names)
        return names

    @property
    def feature_names_out_(self) -> list[str]:
        """Alias for get_feature_names_out()."""
        return self.get_feature_names_out()

    def _get_input_names(self, X) -> list[str]:
        """Extract or generate input feature names."""
        try:
            import pandas  # noqa: F401

            if hasattr(X, "columns"):
                return list(X.columns)
        except ImportError:
            pass

        return [f"f{i}" for i in range(X.shape[1] if X.ndim == 2 else 1)]

    def _preprocess(self, X: np.ndarray, fitting: bool) -> np.ndarray:
        """Handle negatives according to strategy."""
        X_out = X.copy()

        if self.handle_negative == "shift" and fitting:
            # During fit, shift values are computed
            pass
        elif self.handle_negative == "shift" and self.shift_values_ is not None:
            X_out = X_out - self.shift_values_
        elif self.handle_negative == "abs_sign":
            X_out = np.abs(X_out)
        elif self.handle_negative == "error":
            if np.any(X_out < 0):
                raise ValueError("Negative values found with handle_negative='error'.")

        if fitting and self.handle_negative == "shift":
            if self.shift_values_ is not None:
                X_out = X_out - self.shift_values_

        return X_out

    def _to_integers(self, X: np.ndarray) -> np.ndarray:
        """Convert floats to integers for encoding."""
        result = np.zeros_like(X, dtype=np.int64)
        sf = getattr(self, "scale_factor_", 1)

        for i in range(X.shape[1]):
            col = X[:, i]
            if not np.allclose(col, np.round(col)):
                int_col, _ = handle_floats(col, self.handle_float, sf)
            else:
                int_col = np.round(col).astype(np.int64)
            result[:, i] = int_col

        return result

    def __repr__(self):
        params = []
        if self.n_levels != "auto":
            params.append(f"n_levels={self.n_levels}")
        if self.components != "full":
            params.append(f"components='{self.components}'")
        if not self.normalize:
            params.append("normalize=False")
        if self.handle_negative != "abs_sign":
            params.append(f"handle_negative='{self.handle_negative}'")
        if self.handle_float != "scale":
            params.append(f"handle_float='{self.handle_float}'")
        if self.scale_factor != "auto":
            params.append(f"scale_factor={self.scale_factor}")
        return f"VFDEncoder({', '.join(params)})"
