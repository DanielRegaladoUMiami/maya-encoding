"""Tests for Maya calendar conversion functions."""

import datetime

import numpy as np
import pytest

from maya_encoding.core.calendar import (
    GMT_EPOCH_JDN,
    HAAB_MONTH_NAMES,
    TZOLKIN_DAY_NAMES,
    gregorian_to_jdn,
    is_wayeb,
    jdn_array_to_haab,
    jdn_array_to_long_count,
    jdn_array_to_tzolkin,
    jdn_to_gregorian,
    jdn_to_haab,
    jdn_to_long_count,
    jdn_to_tzolkin,
)


class TestGregorianJDN:
    def test_known_date_2012_12_21(self):
        """December 21, 2012 is JDN 2456283."""
        assert gregorian_to_jdn("2012-12-21") == 2456283

    def test_known_date_2000_01_01(self):
        """January 1, 2000 is JDN 2451545."""
        assert gregorian_to_jdn("2000-01-01") == 2451545

    def test_roundtrip(self):
        """JDN -> Gregorian -> JDN should be identity."""
        for jdn in [2451545, 2456283, 2440588, 2460000]:
            date = jdn_to_gregorian(jdn)
            assert gregorian_to_jdn(date) == jdn

    def test_accepts_datetime(self):
        d = datetime.date(2012, 12, 21)
        assert gregorian_to_jdn(d) == 2456283

    def test_accepts_string(self):
        assert gregorian_to_jdn("2012-12-21") == 2456283

    def test_unix_epoch(self):
        """1970-01-01 = JDN 2440588."""
        assert gregorian_to_jdn("1970-01-01") == 2440588


class TestTzolkin:
    def test_2012_12_21_is_4_ajaw(self):
        """December 21, 2012 = 13.0.0.0.0 = 4 Ajaw."""
        jdn = gregorian_to_jdn("2012-12-21")
        number, day_name = jdn_to_tzolkin(jdn)
        assert number == 4
        assert day_name == 19  # Ajaw
        assert TZOLKIN_DAY_NAMES[day_name] == "Ajaw"

    def test_cycle_length_260(self):
        """After 260 days, Tzolk'in should repeat exactly."""
        jdn = gregorian_to_jdn("2024-01-01")
        num1, name1 = jdn_to_tzolkin(jdn)
        num2, name2 = jdn_to_tzolkin(jdn + 260)
        assert num1 == num2
        assert name1 == name2

    def test_number_range(self):
        """Tzolk'in number should always be 1-13."""
        for offset in range(260):
            jdn = GMT_EPOCH_JDN + offset
            number, day_name = jdn_to_tzolkin(jdn)
            assert 1 <= number <= 13
            assert 0 <= day_name <= 19

    def test_all_260_combinations_unique(self):
        """In a 260-day span, all (number, day_name) pairs should be unique."""
        seen = set()
        for offset in range(260):
            jdn = GMT_EPOCH_JDN + offset
            pair = jdn_to_tzolkin(jdn)
            assert pair not in seen, f"Duplicate Tzolk'in date at offset {offset}: {pair}"
            seen.add(pair)
        assert len(seen) == 260

    def test_vectorized_matches_scalar(self):
        jdn_arr = np.array([gregorian_to_jdn(f"2024-01-{d:02d}") for d in range(1, 29)])
        nums_v, names_v = jdn_array_to_tzolkin(jdn_arr)
        for i, jdn in enumerate(jdn_arr):
            num_s, name_s = jdn_to_tzolkin(int(jdn))
            assert nums_v[i] == num_s
            assert names_v[i] == name_s


class TestHaab:
    def test_2012_12_21_is_3_kankin(self):
        """December 21, 2012 = 3 K'ank'in."""
        jdn = gregorian_to_jdn("2012-12-21")
        month, day = jdn_to_haab(jdn)
        assert month == 13  # K'ank'in
        assert day == 3
        assert HAAB_MONTH_NAMES[month] == "K'ank'in"

    def test_cycle_length_365(self):
        """After 365 days, Haab' should repeat."""
        jdn = gregorian_to_jdn("2024-01-01")
        m1, d1 = jdn_to_haab(jdn)
        m2, d2 = jdn_to_haab(jdn + 365)
        assert m1 == m2
        assert d1 == d2

    def test_month_day_ranges(self):
        """All Haab' dates in a cycle should have valid ranges."""
        for offset in range(365):
            jdn = GMT_EPOCH_JDN + offset
            month, day = jdn_to_haab(jdn)
            assert 0 <= month <= 18
            if month < 18:
                assert 0 <= day <= 19, f"Regular month {month} day out of range: {day}"
            else:
                assert 0 <= day <= 4, f"Wayeb' day out of range: {day}"

    def test_wayeb_count(self):
        """Exactly 5 days in a 365-day cycle should be Wayeb'."""
        wayeb_count = sum(
            1 for offset in range(365) if is_wayeb(GMT_EPOCH_JDN + offset)
        )
        assert wayeb_count == 5

    def test_vectorized_matches_scalar(self):
        jdn_arr = np.array([gregorian_to_jdn(f"2024-01-{d:02d}") for d in range(1, 29)])
        months_v, days_v = jdn_array_to_haab(jdn_arr)
        for i, jdn in enumerate(jdn_arr):
            m_s, d_s = jdn_to_haab(int(jdn))
            assert months_v[i] == m_s
            assert days_v[i] == d_s


class TestLongCount:
    def test_2012_12_21_is_13_0_0_0_0(self):
        """December 21, 2012 = 13.0.0.0.0."""
        jdn = gregorian_to_jdn("2012-12-21")
        lc = jdn_to_long_count(jdn)
        assert lc == (13, 0, 0, 0, 0)

    def test_epoch_is_zero(self):
        """The Maya epoch should be 0.0.0.0.0 (or 13.0.0.0.0)."""
        lc = jdn_to_long_count(GMT_EPOCH_JDN)
        # At the epoch, the Long Count is 0.0.0.0.0
        # (some sources say 13.0.0.0.0 but computationally it's 0)
        assert lc[1:] == (0, 0, 0, 0)  # katun through kin should all be 0

    def test_one_day_after_epoch(self):
        """Day after epoch = 0.0.0.0.1."""
        lc = jdn_to_long_count(GMT_EPOCH_JDN + 1, n_levels=5)
        assert lc == (0, 0, 0, 0, 1)

    def test_uinal_boundary(self):
        """After 20 kin, uinal increments."""
        lc = jdn_to_long_count(GMT_EPOCH_JDN + 20, n_levels=5)
        assert lc == (0, 0, 0, 1, 0)

    def test_tun_boundary(self):
        """After 360 kin (18 uinals), tun increments."""
        lc = jdn_to_long_count(GMT_EPOCH_JDN + 360, n_levels=5)
        assert lc == (0, 0, 1, 0, 0)

    def test_fewer_levels(self):
        lc = jdn_to_long_count(GMT_EPOCH_JDN + 1, n_levels=3)
        assert lc == (0, 0, 1)  # (tun, uinal, kin)

    def test_vectorized_shape(self):
        jdn_arr = np.array([GMT_EPOCH_JDN, GMT_EPOCH_JDN + 1, GMT_EPOCH_JDN + 20])
        result = jdn_array_to_long_count(jdn_arr, n_levels=3)
        assert result.shape == (3, 3)

    def test_vectorized_matches_scalar(self):
        test_jdns = [GMT_EPOCH_JDN + d for d in [0, 1, 19, 20, 359, 360, 7200]]
        jdn_arr = np.array(test_jdns)
        result = jdn_array_to_long_count(jdn_arr, n_levels=5)

        for i, jdn in enumerate(test_jdns):
            scalar = jdn_to_long_count(jdn, n_levels=5)
            # Vectorized returns LSB first, scalar returns MSB first
            # So reverse the vectorized to compare
            np.testing.assert_array_equal(
                list(reversed(result[i])),
                list(scalar),
                err_msg=f"Mismatch at JDN {jdn}",
            )
