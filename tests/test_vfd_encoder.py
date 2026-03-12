"""Tests for VFDEncoder sklearn transformer."""

import numpy as np
import pytest

from maya_encoding.vfd.encoder import VFDEncoder


class TestVFDEncoderBasic:
    def test_fit_transform(self):
        X = np.array([[347], [20], [0], [399]])
        enc = VFDEncoder(n_levels=2, components="full", normalize=False)
        result = enc.fit_transform(X)
        assert result.shape == (4, 6)

        # 347 = [7, 17] -> bars=[1,3], dots=[2,2]
        np.testing.assert_array_equal(result[0], [7, 1, 2, 17, 3, 2])

    def test_auto_levels(self):
        X = np.array([[0], [399]])
        enc = VFDEncoder(n_levels="auto", components="lite", normalize=False)
        enc.fit(X)
        assert enc.n_levels_ == 2

    def test_lite_components(self):
        X = np.array([[347]])
        enc = VFDEncoder(n_levels=2, components="lite", normalize=False)
        result = enc.fit_transform(X)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], [7, 17])

    def test_bars_dots_components(self):
        X = np.array([[347]])
        enc = VFDEncoder(n_levels=2, components="bars_dots", normalize=False)
        result = enc.fit_transform(X)
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 2])

    def test_normalization(self):
        X = np.array([[19]])
        enc = VFDEncoder(n_levels=1, components="full", normalize=True)
        result = enc.fit_transform(X)
        np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0])

    def test_zero_normalization(self):
        X = np.array([[0]])
        enc = VFDEncoder(n_levels=1, components="full", normalize=True)
        result = enc.fit_transform(X)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0])


class TestVFDEncoderMultiColumn:
    def test_two_columns(self):
        X = np.array([[10, 5], [20, 15]])
        enc = VFDEncoder(n_levels=2, components="lite", normalize=False)
        result = enc.fit_transform(X)
        # 2 columns * 2 levels = 4 features
        assert result.shape == (2, 4)

    def test_feature_names(self):
        X = np.array([[10, 5], [20, 15]])
        enc = VFDEncoder(n_levels=2, components="full")
        enc.fit(X)
        names = enc.get_feature_names_out()
        # 2 columns * 2 levels * 3 components = 12
        assert len(names) == 12
        assert "f0_L0_digit" in names
        assert "f1_L1_bars" in names


class TestVFDEncoderNegatives:
    def test_abs_sign(self):
        X = np.array([[-5], [5]])
        enc = VFDEncoder(n_levels=1, components="lite", normalize=False, handle_negative="abs_sign")
        result = enc.fit_transform(X)
        # sign feature + 1 digit = 2 features
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0  # negative sign
        assert result[1, 0] == 0.0  # positive sign
        assert result[0, 1] == 5.0  # |−5|
        assert result[1, 1] == 5.0  # |5|

    def test_shift(self):
        X = np.array([[-10], [0], [10]])
        enc = VFDEncoder(n_levels=2, components="lite", normalize=False, handle_negative="shift")
        result = enc.fit_transform(X)
        # After shift: [0, 10, 20]
        assert result.shape == (3, 2)

    def test_error(self):
        X = np.array([[-1], [1]])
        enc = VFDEncoder(handle_negative="error")
        with pytest.raises(ValueError, match="Negative"):
            enc.fit_transform(X)

    def test_no_negatives_no_sign(self):
        X = np.array([[5], [10]])
        enc = VFDEncoder(n_levels=1, components="lite", normalize=False, handle_negative="abs_sign")
        result = enc.fit_transform(X)
        # No negatives, so no sign feature
        assert result.shape == (2, 1)


class TestVFDEncoderFloats:
    def test_scale_auto(self):
        X = np.array([[1.5], [2.75]])
        enc = VFDEncoder(
            n_levels="auto",
            components="lite",
            normalize=False,
            handle_float="scale",
        )
        enc.fit_transform(X)
        assert enc.scale_factor_ >= 100  # Should detect 2 decimal places

    def test_round(self):
        X = np.array([[1.6], [2.4]])
        enc = VFDEncoder(n_levels=1, components="lite", normalize=False, handle_float="round")
        result = enc.fit_transform(X)
        np.testing.assert_array_equal(result.ravel(), [2, 2])

    def test_integer_part(self):
        X = np.array([[1.9], [2.1]])
        enc = VFDEncoder(
            n_levels=1,
            components="lite",
            normalize=False,
            handle_float="integer_part",
        )
        result = enc.fit_transform(X)
        np.testing.assert_array_equal(result.ravel(), [1, 2])


class TestVFDEncoderInverse:
    def test_roundtrip_integers(self):
        X = np.array([[0], [7], [20], [347], [399]])
        enc = VFDEncoder(n_levels=2, normalize=False, components="full")
        encoded = enc.fit_transform(X)
        decoded = enc.inverse_transform(encoded)
        np.testing.assert_array_almost_equal(decoded, X, decimal=0)

    def test_roundtrip_lite(self):
        X = np.array([[0], [7], [20], [347]])
        enc = VFDEncoder(n_levels=2, normalize=False, components="lite")
        encoded = enc.fit_transform(X)
        decoded = enc.inverse_transform(encoded)
        np.testing.assert_array_almost_equal(decoded, X, decimal=0)


class TestVFDEncoderEdgeCases:
    def test_single_sample(self):
        X = np.array([[100]])
        enc = VFDEncoder()
        result = enc.fit_transform(X)
        assert result.shape[0] == 1

    def test_all_zeros(self):
        X = np.zeros((5, 2))
        enc = VFDEncoder(n_levels=1, normalize=False, components="full")
        result = enc.fit_transform(X)
        np.testing.assert_array_equal(result, np.zeros((5, 6)))

    def test_nan_raises(self):
        X = np.array([[1.0], [np.nan]])
        enc = VFDEncoder()
        with pytest.raises(ValueError, match="NaN"):
            enc.fit(X)

    def test_inf_raises(self):
        X = np.array([[1.0], [np.inf]])
        enc = VFDEncoder()
        with pytest.raises(ValueError, match="infinity"):
            enc.fit(X)

    def test_transform_before_fit_raises(self):
        enc = VFDEncoder()
        with pytest.raises(Exception):  # NotFittedError
            enc.transform(np.array([[1]]))


class TestVFDEncoderRepr:
    def test_default_repr(self):
        enc = VFDEncoder()
        assert "VFDEncoder" in repr(enc)

    def test_custom_repr(self):
        enc = VFDEncoder(n_levels=3, components="lite")
        r = repr(enc)
        assert "n_levels=3" in r
        assert "lite" in r


class TestVFDPassthrough:
    def test_passthrough_shape(self):
        X = np.array([[100, 200], [300, 400]])
        enc = VFDEncoder(n_levels=2, components="full", normalize=False, passthrough=True)
        result = enc.fit_transform(X)
        # 2 original + 2 features * 6 VFD features each = 2 + 12 = 14
        assert result.shape == (2, 14)

    def test_passthrough_preserves_original(self):
        X = np.array([[100.0, 200.0], [300.0, 400.0]])
        enc = VFDEncoder(n_levels=2, components="full", normalize=False, passthrough=True)
        result = enc.fit_transform(X)
        # First 2 columns should be original features
        np.testing.assert_array_equal(result[:, :2], X)

    def test_passthrough_false_default(self):
        X = np.array([[100], [200]])
        enc = VFDEncoder(n_levels=2, components="full", normalize=False)
        result = enc.fit_transform(X)
        # Without passthrough: only VFD features
        assert result.shape == (2, 6)

    def test_passthrough_feature_names(self):
        X = np.array([[100, 200]])
        enc = VFDEncoder(n_levels=1, components="lite", normalize=False, passthrough=True)
        enc.fit(X)
        names = enc.get_feature_names_out()
        # 2 original names + 2 VFD names
        assert names[0] == "f0"
        assert names[1] == "f1"
        assert names[2] == "f0_L0_digit"
        assert names[3] == "f1_L0_digit"
        assert len(names) == 4

    def test_passthrough_inverse_transform(self):
        X = np.array([[100.0], [200.0], [300.0]])
        enc = VFDEncoder(n_levels=2, components="full", normalize=False, passthrough=True)
        encoded = enc.fit_transform(X)
        reconstructed = enc.inverse_transform(encoded)
        np.testing.assert_array_almost_equal(reconstructed, X)

    def test_passthrough_pipeline(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        np.random.seed(42)
        X = np.random.randint(0, 100, size=(50, 3)).astype(float)
        y = X[:, 0] * 2 + X[:, 1] + np.random.normal(0, 1, 50)

        pipe = Pipeline([
            ("vfd", VFDEncoder(passthrough=True)),
            ("lr", LinearRegression()),
        ])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert score > 0.5  # Should fit well since target depends on raw features
