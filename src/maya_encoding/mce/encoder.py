"""
MayaCalendarEncoder: Maya Calendar Encoding transformer for scikit-learn.

Converts date/datetime features into cyclical features based on the three
Maya calendar systems (Tzolk'in, Haab', Long Count), providing models with
multi-scale temporal information from interlocking cycles of coprime periods.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from maya_encoding.core.calendar import (
    GMT_EPOCH_JDN,
    SPINDEN_EPOCH_JDN,
    dates_to_jdn_array,
    jdn_array_to_haab,
    jdn_array_to_long_count,
    jdn_array_to_tzolkin,
)


class MayaCalendarEncoder(BaseEstimator, TransformerMixin):
    """Maya Calendar Encoding for temporal features.

    Converts dates into features derived from the three Maya calendar systems:

    - **Tzolk'in** (260-day cycle): Two interlocking sub-cycles of period 13
      and 20. Since gcd(13,20)=1, every (number, day_name) combination is
      unique within the 260-day cycle. Captures bi-weekly and tri-weekly patterns.

    - **Haab'** (365-day cycle): 18 months of 20 days + 5-day Wayeb' period.
      Each day within a month is further decomposed into Maya bars (÷5) and
      dots (%5), giving sub-monthly structure.

    - **Long Count** (linear mixed-radix): Kin (days), uinal (20 days),
      tun (360 days ≈ 1 year). Captures longer-term trends. Uses the Maya
      calendar exception (18×20=360 for the tun level).

    Parameters
    ----------
    components : list[str], default=['tzolkin', 'haab', 'long_count']
        Which calendar systems to include. Any subset of
        ['tzolkin', 'haab', 'long_count'].

    tzolkin_encoding : str, default='separate'
        How to encode the Tzolk'in:
        - 'separate': Two features (number 1-13, day_name 0-19)
        - 'combined': Single feature (position 0-259 in the cycle)

    haab_encoding : str, default='hierarchical'
        How to encode the Haab':
        - 'hierarchical': month + day + bars + dots (4 features)
        - 'flat': Single feature (position 0-364 in the cycle)

    long_count_levels : int, default=3
        Number of Long Count levels to include:
        - 1: kin only (mod 20)
        - 2: kin + uinal (mod 18)
        - 3: kin + uinal + tun (mod 20) — captures ~yearly cycles
        - 4: + katun (~20 year cycles)
        - 5: + baktun (~394 year cycles)

    cyclical : bool, default=True
        If True, add sine/cosine pairs for each cyclical component.
        This creates a smooth representation where values near the cycle
        boundary are close in feature space.

    epoch : str or int, default='gmt'
        Maya epoch to use:
        - 'gmt': GMT correlation (JDN 584283, the academic standard)
        - 'spinden': Spinden correlation (JDN 489384)
        - int: Custom Julian Day Number for the epoch

    wayeb_flag : bool, default=True
        If True, add a binary feature indicating whether the date falls
        in the 5-day Wayeb' period of the Haab' calendar.

    normalize : bool, default=True
        If True, normalize raw components to [0, 1] range.

    Attributes
    ----------
    n_features_in_ : int
        Always 1 (single date column).
    epoch_jdn_ : int
        Resolved Julian Day Number of the Maya epoch.

    Examples
    --------
    >>> import pandas as pd
    >>> from maya_encoding import MayaCalendarEncoder
    >>> dates = pd.Series(['2012-12-21', '2024-01-01', '2024-06-15'])
    >>> enc = MayaCalendarEncoder(components=['tzolkin', 'haab'])
    >>> enc.fit_transform(dates)  # doctest: +SKIP
    """

    def __init__(
        self,
        components: list[str] | None = None,
        tzolkin_encoding: str = "separate",
        haab_encoding: str = "hierarchical",
        long_count_levels: int = 3,
        cyclical: bool = True,
        epoch: str | int = "gmt",
        wayeb_flag: bool = True,
        normalize: bool = True,
    ):
        self.components = components if components is not None else ["tzolkin", "haab", "long_count"]
        self.tzolkin_encoding = tzolkin_encoding
        self.haab_encoding = haab_encoding
        self.long_count_levels = long_count_levels
        self.cyclical = cyclical
        self.epoch = epoch
        self.wayeb_flag = wayeb_flag
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit the encoder (validates input and resolves parameters).

        Parameters
        ----------
        X : array-like
            Date column (strings, datetime, datetime64, or timestamps).
        y : ignored

        Returns
        -------
        self
        """
        # Resolve epoch
        if self.epoch == "gmt":
            self.epoch_jdn_ = GMT_EPOCH_JDN
        elif self.epoch == "spinden":
            self.epoch_jdn_ = SPINDEN_EPOCH_JDN
        elif isinstance(self.epoch, int):
            self.epoch_jdn_ = self.epoch
        else:
            raise ValueError(
                f"Unknown epoch '{self.epoch}'. Use 'gmt', 'spinden', or an integer JDN."
            )

        # Validate components
        valid_components = {"tzolkin", "haab", "long_count"}
        for c in self.components:
            if c not in valid_components:
                raise ValueError(f"Unknown component '{c}'. Use: {valid_components}")

        self.n_features_in_ = 1

        # Validate that we can parse the dates
        _dates = self._to_array(X)
        _ = dates_to_jdn_array(_dates[:min(5, len(_dates))])

        return self

    def transform(self, X):
        """Transform dates to Maya calendar features.

        Parameters
        ----------
        X : array-like
            Date column.

        Returns
        -------
        np.ndarray
            2D array of shape (n_dates, n_output_features).
        """
        check_is_fitted(self)

        dates = self._to_array(X)
        jdn = dates_to_jdn_array(dates)
        n = len(jdn)

        feature_arrays = []

        # --- Tzolk'in ---
        if "tzolkin" in self.components:
            tz_numbers, tz_names = jdn_array_to_tzolkin(jdn, self.epoch_jdn_)

            if self.tzolkin_encoding == "separate":
                tz_num = tz_numbers.astype(np.float64)
                tz_name = tz_names.astype(np.float64)

                if self.normalize:
                    tz_num = (tz_num - 1) / 12.0  # [1,13] -> [0,1]
                    tz_name = tz_name / 19.0       # [0,19] -> [0,1]

                feature_arrays.append(tz_num.reshape(-1, 1))
                feature_arrays.append(tz_name.reshape(-1, 1))

                if self.cyclical:
                    # Sine/cosine for number (period 13)
                    angle_num = 2 * np.pi * (tz_numbers - 1) / 13.0
                    feature_arrays.append(np.sin(angle_num).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle_num).reshape(-1, 1))

                    # Sine/cosine for day name (period 20)
                    angle_name = 2 * np.pi * tz_names / 20.0
                    feature_arrays.append(np.sin(angle_name).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle_name).reshape(-1, 1))

            elif self.tzolkin_encoding == "combined":
                # Combined position in 260-day cycle
                # Use CRT: pos = (40*(num-1) + 221*name) mod 260
                tz_pos = (40 * (tz_numbers - 1) + 221 * tz_names) % 260
                tz_pos_f = tz_pos.astype(np.float64)

                if self.normalize:
                    tz_pos_f = tz_pos_f / 259.0

                feature_arrays.append(tz_pos_f.reshape(-1, 1))

                if self.cyclical:
                    angle = 2 * np.pi * tz_pos / 260.0
                    feature_arrays.append(np.sin(angle).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle).reshape(-1, 1))

        # --- Haab' ---
        if "haab" in self.components:
            hb_months, hb_days = jdn_array_to_haab(jdn, self.epoch_jdn_)

            if self.haab_encoding == "hierarchical":
                hb_month_f = hb_months.astype(np.float64)
                hb_day_f = hb_days.astype(np.float64)

                # Bars and dots decomposition of the day within month
                hb_bars = (hb_days // 5).astype(np.float64)
                hb_dots = (hb_days % 5).astype(np.float64)

                if self.normalize:
                    hb_month_f = hb_month_f / 18.0    # [0,18] -> [0,1]
                    hb_day_f = hb_day_f / 19.0         # [0,19] -> [0,1]
                    hb_bars = hb_bars / 3.0             # [0,3] -> [0,1]
                    hb_dots = hb_dots / 4.0             # [0,4] -> [0,1]

                feature_arrays.append(hb_month_f.reshape(-1, 1))
                feature_arrays.append(hb_day_f.reshape(-1, 1))
                feature_arrays.append(hb_bars.reshape(-1, 1))
                feature_arrays.append(hb_dots.reshape(-1, 1))

                if self.cyclical:
                    # Sine/cosine for month (period 19: 18 months + Wayeb')
                    angle_month = 2 * np.pi * hb_months / 19.0
                    feature_arrays.append(np.sin(angle_month).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle_month).reshape(-1, 1))

                    # Sine/cosine for day within month (period 20)
                    angle_day = 2 * np.pi * hb_days / 20.0
                    feature_arrays.append(np.sin(angle_day).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle_day).reshape(-1, 1))

            elif self.haab_encoding == "flat":
                # Flat position in 365-day cycle
                hb_pos = (hb_months * 20 + hb_days).astype(np.float64)
                # Clamp Wayeb': month 18, days 0-4 -> positions 360-364
                hb_pos = np.where(hb_months == 18, 360 + hb_days, hb_pos).astype(np.float64)

                if self.normalize:
                    hb_pos = hb_pos / 364.0

                feature_arrays.append(hb_pos.reshape(-1, 1))

                if self.cyclical:
                    raw_pos = np.where(hb_months == 18, 360 + hb_days, hb_months * 20 + hb_days)
                    angle = 2 * np.pi * raw_pos / 365.0
                    feature_arrays.append(np.sin(angle).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle).reshape(-1, 1))

            # Wayeb' flag
            if self.wayeb_flag:
                is_wayeb = (hb_months == 18).astype(np.float64)
                feature_arrays.append(is_wayeb.reshape(-1, 1))

        # --- Long Count ---
        if "long_count" in self.components:
            lc = jdn_array_to_long_count(jdn, self.long_count_levels, self.epoch_jdn_)
            # lc shape: (n, long_count_levels), LSB first: [kin, uinal, tun, ...]

            # Max values for each level for normalization
            lc_max = [19, 17, 19, 19, 19]  # kin:0-19, uinal:0-17, tun/katun/baktun:0-19
            lc_periods = [20, 18, 20, 20, 20]

            for level in range(self.long_count_levels):
                lc_val = lc[:, level].astype(np.float64)

                if self.normalize:
                    lc_val_norm = lc_val / lc_max[level]
                    feature_arrays.append(lc_val_norm.reshape(-1, 1))
                else:
                    feature_arrays.append(lc_val.reshape(-1, 1))

                if self.cyclical:
                    angle = 2 * np.pi * lc[:, level] / lc_periods[level]
                    feature_arrays.append(np.sin(angle).reshape(-1, 1))
                    feature_arrays.append(np.cos(angle).reshape(-1, 1))

        if not feature_arrays:
            raise ValueError("No features generated. Check 'components' parameter.")

        return np.hstack(feature_arrays)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names for transformation.

        Returns
        -------
        list[str]
            Descriptive feature names.
        """
        check_is_fitted(self)
        names = []

        if "tzolkin" in self.components:
            if self.tzolkin_encoding == "separate":
                names.append("tzolkin_number")
                names.append("tzolkin_day_name")
                if self.cyclical:
                    names.extend(["tzolkin_number_sin", "tzolkin_number_cos"])
                    names.extend(["tzolkin_day_name_sin", "tzolkin_day_name_cos"])
            elif self.tzolkin_encoding == "combined":
                names.append("tzolkin_position")
                if self.cyclical:
                    names.extend(["tzolkin_position_sin", "tzolkin_position_cos"])

        if "haab" in self.components:
            if self.haab_encoding == "hierarchical":
                names.extend(["haab_month", "haab_day", "haab_day_bars", "haab_day_dots"])
                if self.cyclical:
                    names.extend(["haab_month_sin", "haab_month_cos"])
                    names.extend(["haab_day_sin", "haab_day_cos"])
            elif self.haab_encoding == "flat":
                names.append("haab_position")
                if self.cyclical:
                    names.extend(["haab_position_sin", "haab_position_cos"])
            if self.wayeb_flag:
                names.append("is_wayeb")

        if "long_count" in self.components:
            lc_names = ["long_count_kin", "long_count_uinal", "long_count_tun",
                        "long_count_katun", "long_count_baktun"]
            for level in range(self.long_count_levels):
                names.append(lc_names[level])
                if self.cyclical:
                    names.extend([f"{lc_names[level]}_sin", f"{lc_names[level]}_cos"])

        return names

    @property
    def feature_names_out_(self) -> list[str]:
        """Alias for get_feature_names_out()."""
        return self.get_feature_names_out()

    def _to_array(self, X):
        """Convert input to a flat array-like of dates."""
        try:
            import pandas as pd

            if isinstance(X, pd.DataFrame):
                if X.shape[1] != 1:
                    raise ValueError(
                        f"MayaCalendarEncoder expects a single date column, got {X.shape[1]} columns."
                    )
                return X.iloc[:, 0]
            if isinstance(X, pd.Series):
                return X
        except ImportError:
            pass

        X = np.asarray(X)
        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError(
                    f"MayaCalendarEncoder expects a single date column, got {X.shape[1]} columns."
                )
            X = X.ravel()
        return X

    def __repr__(self):
        parts = []
        if self.components != ["tzolkin", "haab", "long_count"]:
            parts.append(f"components={self.components}")
        if self.tzolkin_encoding != "separate":
            parts.append(f"tzolkin_encoding='{self.tzolkin_encoding}'")
        if self.haab_encoding != "hierarchical":
            parts.append(f"haab_encoding='{self.haab_encoding}'")
        if self.long_count_levels != 3:
            parts.append(f"long_count_levels={self.long_count_levels}")
        if not self.cyclical:
            parts.append("cyclical=False")
        if self.epoch != "gmt":
            parts.append(f"epoch='{self.epoch}'")
        return f"MayaCalendarEncoder({', '.join(parts)})"
