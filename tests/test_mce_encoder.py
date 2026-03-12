"""Tests for MayaCalendarEncoder sklearn transformer."""

import numpy as np
import pytest

from maya_encoding.mce.encoder import MayaCalendarEncoder


class TestMCEBasic:
    def test_fit_transform_strings(self):
        dates = np.array(["2012-12-21", "2024-01-01", "2024-06-15"])
        enc = MayaCalendarEncoder()
        result = enc.fit_transform(dates)
        assert result.shape[0] == 3
        assert result.shape[1] > 0

    def test_single_component_tzolkin(self):
        dates = np.array(["2024-01-01"])
        enc = MayaCalendarEncoder(
            components=["tzolkin"],
            tzolkin_encoding="separate",
            cyclical=False,
        )
        result = enc.fit_transform(dates)
        # separate: number + day_name = 2 features
        assert result.shape == (1, 2)

    def test_single_component_haab(self):
        dates = np.array(["2024-01-01"])
        enc = MayaCalendarEncoder(
            components=["haab"],
            haab_encoding="hierarchical",
            cyclical=False,
            wayeb_flag=True,
        )
        result = enc.fit_transform(dates)
        # hierarchical: month + day + bars + dots + wayeb = 5 features
        assert result.shape == (1, 5)

    def test_single_component_long_count(self):
        dates = np.array(["2024-01-01"])
        enc = MayaCalendarEncoder(
            components=["long_count"],
            long_count_levels=3,
            cyclical=False,
        )
        result = enc.fit_transform(dates)
        # 3 levels = 3 features
        assert result.shape == (1, 3)


class TestMCEFeatureNames:
    def test_names_match_output(self):
        dates = np.array(["2024-01-01", "2024-06-15"])
        enc = MayaCalendarEncoder(
            components=["tzolkin", "haab"],
            cyclical=True,
            wayeb_flag=True,
        )
        result = enc.fit_transform(dates)
        names = enc.get_feature_names_out()
        assert len(names) == result.shape[1]

    def test_tzolkin_separate_names(self):
        enc = MayaCalendarEncoder(
            components=["tzolkin"],
            tzolkin_encoding="separate",
            cyclical=True,
        )
        enc.fit(np.array(["2024-01-01"]))
        names = enc.get_feature_names_out()
        assert "tzolkin_number" in names
        assert "tzolkin_day_name" in names
        assert "tzolkin_number_sin" in names
        assert "tzolkin_number_cos" in names

    def test_tzolkin_combined_names(self):
        enc = MayaCalendarEncoder(
            components=["tzolkin"],
            tzolkin_encoding="combined",
            cyclical=True,
        )
        enc.fit(np.array(["2024-01-01"]))
        names = enc.get_feature_names_out()
        assert "tzolkin_position" in names

    def test_long_count_names(self):
        enc = MayaCalendarEncoder(
            components=["long_count"],
            long_count_levels=3,
            cyclical=False,
        )
        enc.fit(np.array(["2024-01-01"]))
        names = enc.get_feature_names_out()
        assert "long_count_kin" in names
        assert "long_count_uinal" in names
        assert "long_count_tun" in names


class TestMCECyclical:
    def test_cyclical_adds_sincos(self):
        dates = np.array(["2024-01-01"])
        enc_no = MayaCalendarEncoder(components=["tzolkin"], cyclical=False, tzolkin_encoding="separate")
        enc_yes = MayaCalendarEncoder(components=["tzolkin"], cyclical=True, tzolkin_encoding="separate")

        r_no = enc_no.fit_transform(dates)
        r_yes = enc_yes.fit_transform(dates)

        # With cyclical: 2 raw + 4 sin/cos = 6
        # Without: 2 raw
        assert r_yes.shape[1] > r_no.shape[1]

    def test_sincos_range(self):
        """Sin/cos values should be in [-1, 1]."""
        dates = np.array([f"2024-{m:02d}-15" for m in range(1, 13)])
        enc = MayaCalendarEncoder(cyclical=True)
        result = enc.fit_transform(dates)
        assert np.all(result >= -1.01)
        assert np.all(result <= 1.01)


class TestMCENormalization:
    def test_normalized_range(self):
        """All normalized features should be in [0, 1] (except sin/cos)."""
        dates = np.array([f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in [1, 15]])
        enc = MayaCalendarEncoder(cyclical=False, normalize=True)
        result = enc.fit_transform(dates)
        assert np.all(result >= -0.01)
        assert np.all(result <= 1.01)


class TestMCEHaabEncoding:
    def test_flat_encoding(self):
        dates = np.array(["2024-01-01"])
        enc = MayaCalendarEncoder(
            components=["haab"],
            haab_encoding="flat",
            cyclical=False,
            wayeb_flag=False,
        )
        result = enc.fit_transform(dates)
        assert result.shape == (1, 1)

    def test_wayeb_flag(self):
        # Generate a full year to ensure some Wayeb' days exist
        import datetime as dt

        dates = np.array([
            (dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat()
            for i in range(365)
        ])
        enc = MayaCalendarEncoder(
            components=["haab"],
            haab_encoding="hierarchical",
            cyclical=False,
            wayeb_flag=True,
        )
        result = enc.fit_transform(dates)
        # Last column is wayeb flag
        wayeb_col = result[:, -1]
        assert np.sum(wayeb_col) == 5  # Exactly 5 Wayeb' days in a Haab' year


class TestMCEEpoch:
    def test_gmt_default(self):
        enc = MayaCalendarEncoder()
        enc.fit(np.array(["2024-01-01"]))
        assert enc.epoch_jdn_ == 584283

    def test_custom_epoch(self):
        enc = MayaCalendarEncoder(epoch=500000)
        enc.fit(np.array(["2024-01-01"]))
        assert enc.epoch_jdn_ == 500000

    def test_invalid_epoch(self):
        with pytest.raises(ValueError, match="Unknown epoch"):
            enc = MayaCalendarEncoder(epoch="invalid")
            enc.fit(np.array(["2024-01-01"]))


class TestMCEEdgeCases:
    def test_transform_before_fit(self):
        enc = MayaCalendarEncoder()
        with pytest.raises(Exception):
            enc.transform(np.array(["2024-01-01"]))

    def test_invalid_component(self):
        with pytest.raises(ValueError, match="Unknown component"):
            enc = MayaCalendarEncoder(components=["invalid"])
            enc.fit(np.array(["2024-01-01"]))

    def test_repr(self):
        enc = MayaCalendarEncoder(components=["tzolkin"], cyclical=False)
        r = repr(enc)
        assert "MayaCalendarEncoder" in r


class TestMCEPandas:
    def test_pandas_series(self):
        """Should accept pandas Series of dates."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        dates = pd.Series(["2024-01-01", "2024-06-15", "2024-12-31"])
        enc = MayaCalendarEncoder(cyclical=False)
        result = enc.fit_transform(dates)
        assert result.shape[0] == 3

    def test_pandas_datetime_series(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        dates = pd.to_datetime(["2024-01-01", "2024-06-15"])
        enc = MayaCalendarEncoder(cyclical=False)
        result = enc.fit_transform(dates)
        assert result.shape[0] == 2
