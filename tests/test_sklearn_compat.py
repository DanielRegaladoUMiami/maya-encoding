"""Tests for scikit-learn compatibility."""

import numpy as np

from maya_encoding.vfd.encoder import VFDEncoder


class TestSklearnPipeline:
    """Test VFDEncoder works in sklearn pipelines."""

    def test_pipeline_regression(self):
        """VFDEncoder should work in a regression pipeline."""
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        X = np.random.randint(0, 400, size=(100, 3)).astype(float)
        y = np.random.randn(100)

        pipe = Pipeline([
            ("encode", VFDEncoder(n_levels=2, components="full")),
            ("model", LinearRegression()),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)

    def test_pipeline_with_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        X = np.random.randint(0, 100, size=(50, 2)).astype(float)
        y = np.random.randn(50)

        pipe = Pipeline([
            ("encode", VFDEncoder(components="lite")),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ])

        pipe.fit(X, y)
        score = pipe.score(X, y)
        assert isinstance(score, float)

    def test_get_params(self):
        """get_params should work for GridSearchCV compatibility."""
        enc = VFDEncoder(n_levels=3, components="lite")
        params = enc.get_params()
        assert params["n_levels"] == 3
        assert params["components"] == "lite"

    def test_set_params(self):
        enc = VFDEncoder()
        enc.set_params(n_levels=5, components="bars_dots")
        assert enc.n_levels == 5
        assert enc.components == "bars_dots"

    def test_clone(self):
        from sklearn.base import clone

        enc = VFDEncoder(n_levels=3, components="lite", normalize=False)
        enc2 = clone(enc)
        assert enc2.n_levels == 3
        assert enc2.components == "lite"
        assert enc2.normalize is False
