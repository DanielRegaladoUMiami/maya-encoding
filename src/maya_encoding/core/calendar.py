"""
Maya calendar conversion functions.

Implements conversions from Gregorian dates to the three Maya calendar systems:
- Tzolk'in (260-day sacred calendar): 13 numbers x 20 day names
- Haab' (365-day solar calendar): 18 months of 20 days + 5-day Wayeb'
- Long Count (linear day count): mixed-radix system (20, 20, 18, 20, 20...)

All conversions go through Julian Day Number (JDN) as intermediate representation.
Uses the GMT (Goodman-Martinez-Thompson) correlation constant (JDN 584283) by default.
"""

from __future__ import annotations

import datetime
from typing import Union

import numpy as np

# --- Constants ---

# GMT correlation constant: Julian Day Number of the Maya epoch (0.0.0.0.0)
# This is the most widely accepted correlation in Mayanist scholarship.
# Corresponds to August 11, 3114 BCE in the proleptic Gregorian calendar.
GMT_EPOCH_JDN = 584283

# Alternative correlations (available but not default)
SPINDEN_EPOCH_JDN = 489384

# The 20 Tzolk'in day names (in Yucatec Maya orthography)
TZOLKIN_DAY_NAMES = [
    "Imix", "Ik'", "Ak'bal", "K'an", "Chikchan",
    "Kimi", "Manik'", "Lamat", "Muluk", "Ok",
    "Chuwen", "Eb", "Ben", "Ix", "Men",
    "Kib", "Kaban", "Etz'nab", "Kawak", "Ajaw",
]

# The 19 Haab' month names (18 regular + Wayeb')
HAAB_MONTH_NAMES = [
    "Pop", "Wo", "Sip", "Sotz'", "Sek",
    "Xul", "Yaxk'in", "Mol", "Ch'en", "Yax",
    "Sak", "Keh", "Mak", "K'ank'in", "Muwan",
    "Pax", "K'ayab", "Kumk'u",
    "Wayeb'",  # The 5 "nameless" days
]

# Long Count level multipliers (in kins/days)
# kin=1, uinal=20, tun=360, katun=7200, baktun=144000
LONG_COUNT_MULTIPLIERS = [1, 20, 360, 7200, 144000]
LONG_COUNT_LEVEL_NAMES = ["kin", "uinal", "tun", "katun", "baktun"]

# Radix for each Long Count level (how many of level i fit in level i+1)
# Note: uinal->tun is 18 (not 20) — the calendar exception
LONG_COUNT_RADIX = [20, 18, 20, 20]


# --- Date Parsing ---

DateLike = Union[str, datetime.date, datetime.datetime, np.datetime64, int, float]


def _parse_date(date: DateLike) -> datetime.date:
    """Convert various date representations to datetime.date.

    Parameters
    ----------
    date : DateLike
        A date as string ('YYYY-MM-DD'), datetime, numpy datetime64,
        or Unix timestamp (int/float, seconds since 1970-01-01).

    Returns
    -------
    datetime.date
    """
    if isinstance(date, datetime.datetime):
        return date.date()
    if isinstance(date, datetime.date):
        return date
    if isinstance(date, str):
        return datetime.date.fromisoformat(date)
    if isinstance(date, np.datetime64):
        # Convert to Python datetime
        ts = (date - np.datetime64("1970-01-01", "D")) / np.timedelta64(1, "D")
        return datetime.date(1970, 1, 1) + datetime.timedelta(days=int(ts))
    if isinstance(date, (int, float)):
        # Unix timestamp
        return datetime.date.fromtimestamp(date)
    raise TypeError(f"Cannot parse date of type {type(date)}. Use str, datetime, or timestamp.")


# --- Julian Day Number ---

def gregorian_to_jdn(date: DateLike) -> int:
    """Convert a Gregorian date to Julian Day Number (JDN).

    Uses the standard algorithm for proleptic Gregorian calendar.

    Parameters
    ----------
    date : DateLike
        The date to convert.

    Returns
    -------
    int
        Julian Day Number.

    Examples
    --------
    >>> gregorian_to_jdn('2012-12-21')
    2456283
    """
    d = _parse_date(date)
    y = d.year
    m = d.month
    day = d.day

    # Standard JDN algorithm
    a = (14 - m) // 12
    y_adj = y + 4800 - a
    m_adj = m + 12 * a - 3

    jdn = day + (153 * m_adj + 2) // 5 + 365 * y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 - 32045
    return jdn


def jdn_to_gregorian(jdn: int) -> datetime.date:
    """Convert a Julian Day Number back to Gregorian date.

    Parameters
    ----------
    jdn : int
        Julian Day Number.

    Returns
    -------
    datetime.date
    """
    # Standard inverse JDN algorithm
    a = jdn + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153

    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10

    return datetime.date(year, month, day)


# --- Tzolk'in ---

def jdn_to_tzolkin(jdn: int, epoch_jdn: int = GMT_EPOCH_JDN) -> tuple[int, int]:
    """Convert a Julian Day Number to Tzolk'in date.

    The Tzolk'in is a 260-day cycle composed of two interlocking sub-cycles:
    - A number from 1 to 13
    - A day name from 0 to 19 (index into TZOLKIN_DAY_NAMES)

    Since gcd(13, 20) = 1, every combination occurs exactly once per 260 days.

    Parameters
    ----------
    jdn : int
        Julian Day Number.
    epoch_jdn : int
        JDN of the Maya epoch (default: GMT correlation).

    Returns
    -------
    tuple[int, int]
        (number, day_name_index) where number in [1, 13] and day_name_index in [0, 19].

    Examples
    --------
    >>> jdn_to_tzolkin(2456283)  # 2012-12-21 = 4 Ajaw
    (4, 19)
    """
    # Days since epoch
    day_count = jdn - epoch_jdn

    # The epoch 0.0.0.0.0 corresponds to 4 Ajaw in the Tzolk'in
    # Ajaw = index 19, number = 4
    epoch_number = 4    # 1-indexed
    epoch_name = 19     # 0-indexed (Ajaw)

    # Tzolk'in number cycles through 1-13
    number = ((day_count + epoch_number - 1) % 13) + 1

    # Tzolk'in day name cycles through 0-19
    day_name = (day_count + epoch_name) % 20

    return number, day_name


def tzolkin_to_day_in_cycle(number: int, day_name: int) -> int:
    """Convert Tzolk'in (number, day_name) to position in the 260-day cycle (0-259).

    Uses the Chinese Remainder Theorem since gcd(13, 20) = 1.

    Parameters
    ----------
    number : int
        Tzolk'in number (1-13).
    day_name : int
        Tzolk'in day name index (0-19).

    Returns
    -------
    int
        Day position in the 260-day cycle (0-259).
    """
    # CRT: find x such that x ≡ (number-1) mod 13 and x ≡ day_name mod 20
    # Solution: x = (40*(number-1) + 221*day_name) mod 260
    return (40 * (number - 1) + 221 * day_name) % 260


# --- Haab' ---

def jdn_to_haab(jdn: int, epoch_jdn: int = GMT_EPOCH_JDN) -> tuple[int, int]:
    """Convert a Julian Day Number to Haab' date.

    The Haab' is a 365-day cycle:
    - 18 months of 20 days each (months 0-17)
    - 1 short month "Wayeb'" of 5 days (month 18)

    Parameters
    ----------
    jdn : int
        Julian Day Number.
    epoch_jdn : int
        JDN of the Maya epoch (default: GMT correlation).

    Returns
    -------
    tuple[int, int]
        (month_index, day) where month_index in [0, 18] and day in [0, 19]
        (or [0, 4] for Wayeb' month 18).

    Examples
    --------
    >>> jdn_to_haab(2456283)  # 2012-12-21 = 3 K'ank'in
    (13, 3)
    """
    day_count = jdn - epoch_jdn

    # The epoch 0.0.0.0.0 corresponds to 8 Kumk'u in the Haab'
    # Kumk'u = month index 17, day 8
    # Day-in-year for 8 Kumk'u: 17*20 + 8 = 348
    epoch_haab_day = 17 * 20 + 8  # = 348

    # Position in the 365-day Haab' cycle
    haab_pos = (day_count + epoch_haab_day) % 365

    if haab_pos < 0:
        haab_pos += 365

    # First 360 days: 18 months of 20 days
    if haab_pos < 360:
        month = haab_pos // 20
        day = haab_pos % 20
    else:
        # Last 5 days: Wayeb'
        month = 18
        day = haab_pos - 360

    return month, day


def is_wayeb(jdn: int, epoch_jdn: int = GMT_EPOCH_JDN) -> bool:
    """Check if a Julian Day Number falls in the 5-day Wayeb' period.

    Parameters
    ----------
    jdn : int
        Julian Day Number.
    epoch_jdn : int
        JDN of the Maya epoch.

    Returns
    -------
    bool
        True if the date is in Wayeb'.
    """
    month, _ = jdn_to_haab(jdn, epoch_jdn)
    return month == 18


# --- Long Count ---

def jdn_to_long_count(
    jdn: int, n_levels: int = 5, epoch_jdn: int = GMT_EPOCH_JDN
) -> tuple[int, ...]:
    """Convert a Julian Day Number to Maya Long Count.

    The Long Count is a mixed-radix system:
    - Level 0: kin (1 day)
    - Level 1: uinal (20 kin)
    - Level 2: tun (18 uinal = 360 days) ← calendar exception
    - Level 3: katun (20 tun = 7,200 days)
    - Level 4: baktun (20 katun = 144,000 days)

    Parameters
    ----------
    jdn : int
        Julian Day Number.
    n_levels : int
        Number of Long Count levels to return (1-5). Default 5 (full).
    epoch_jdn : int
        JDN of the Maya epoch.

    Returns
    -------
    tuple[int, ...]
        Long Count digits from highest to lowest level.
        For n_levels=5: (baktun, katun, tun, uinal, kin).

    Examples
    --------
    >>> jdn_to_long_count(2456283)  # 2012-12-21 = 13.0.0.0.0
    (13, 0, 0, 0, 0)
    """
    day_count = jdn - epoch_jdn

    # Decompose using mixed radix: kin, uinal, tun, katun, baktun
    digits = []
    remaining = day_count

    # Level 0: kin (mod 20)
    digits.append(remaining % 20)
    remaining //= 20

    # Level 1: uinal (mod 18) — the calendar exception
    digits.append(remaining % 18)
    remaining //= 18

    # Level 2+: tun, katun, baktun (all mod 20)
    for _ in range(n_levels - 2):
        digits.append(remaining % 20)
        remaining //= 20

    # Pad if needed
    while len(digits) < n_levels:
        digits.append(0)

    # Return in traditional order: highest level first
    return tuple(reversed(digits[:n_levels]))


def long_count_to_kin(long_count: tuple[int, ...]) -> int:
    """Convert Long Count digits to total number of kin (days).

    Parameters
    ----------
    long_count : tuple[int, ...]
        Long Count digits, highest level first.
        E.g., (13, 0, 0, 0, 0) for 13.0.0.0.0.

    Returns
    -------
    int
        Total kin (days) since Maya epoch.
    """
    # Reverse to get LSB first: kin, uinal, tun, katun, baktun
    digits = list(reversed(long_count))
    total = 0
    for i, d in enumerate(digits):
        if i < len(LONG_COUNT_MULTIPLIERS):
            total += d * LONG_COUNT_MULTIPLIERS[i]
    return total


# --- Vectorized conversions ---

def dates_to_jdn_array(dates) -> np.ndarray:
    """Convert an array of dates to Julian Day Numbers.

    Parameters
    ----------
    dates : array-like
        Array of dates (strings, datetime, timestamps, or numpy datetime64).

    Returns
    -------
    np.ndarray
        1D array of JDN integers.
    """
    # Handle numpy datetime64 arrays efficiently
    if isinstance(dates, np.ndarray) and np.issubdtype(dates.dtype, np.datetime64):
        # Convert to days since epoch, then to JDN
        # JDN of 1970-01-01 = 2440588
        days_since_unix = (dates - np.datetime64("1970-01-01", "D")) / np.timedelta64(1, "D")
        return (days_since_unix + 2440588).astype(np.int64)

    # Handle pandas datetime
    try:
        import pandas as pd

        if isinstance(dates, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(dates):
                days = (dates - pd.Timestamp("1970-01-01")).dt.days
                return (days.values + 2440588).astype(np.int64)
            else:
                dates = pd.to_datetime(dates)
                days = (dates - pd.Timestamp("1970-01-01")).dt.days
                return (days.values + 2440588).astype(np.int64)
    except ImportError:
        pass

    # Fallback: parse one by one
    result = np.array([gregorian_to_jdn(d) for d in dates], dtype=np.int64)
    return result


def jdn_array_to_tzolkin(jdn_array: np.ndarray, epoch_jdn: int = GMT_EPOCH_JDN) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Tzolk'in conversion.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (numbers_array, day_names_array)
    """
    day_count = jdn_array - epoch_jdn
    numbers = ((day_count + 4 - 1) % 13) + 1  # epoch number = 4
    day_names = (day_count + 19) % 20           # epoch name = 19 (Ajaw)
    return numbers.astype(np.int64), day_names.astype(np.int64)


def jdn_array_to_haab(jdn_array: np.ndarray, epoch_jdn: int = GMT_EPOCH_JDN) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Haab' conversion.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (months_array, days_array)
    """
    day_count = jdn_array - epoch_jdn
    epoch_haab_day = 17 * 20 + 8  # 8 Kumk'u = day 348 in Haab'
    haab_pos = (day_count + epoch_haab_day) % 365

    # Handle negatives
    haab_pos = np.where(haab_pos < 0, haab_pos + 365, haab_pos)

    months = np.where(haab_pos < 360, haab_pos // 20, 18)
    days = np.where(haab_pos < 360, haab_pos % 20, haab_pos - 360)

    return months.astype(np.int64), days.astype(np.int64)


def jdn_array_to_long_count(
    jdn_array: np.ndarray, n_levels: int = 3, epoch_jdn: int = GMT_EPOCH_JDN
) -> np.ndarray:
    """Vectorized Long Count conversion.

    Parameters
    ----------
    jdn_array : np.ndarray
        1D array of JDN values.
    n_levels : int
        Number of Long Count levels (1=kin, 2=+uinal, 3=+tun, etc.)
    epoch_jdn : int
        Maya epoch JDN.

    Returns
    -------
    np.ndarray
        2D array of shape (n_dates, n_levels), LSB first: [kin, uinal, tun, ...].
    """
    day_count = jdn_array - epoch_jdn
    n = len(jdn_array)
    result = np.zeros((n, n_levels), dtype=np.int64)
    remaining = day_count.copy()

    # Level 0: kin (mod 20)
    if n_levels >= 1:
        result[:, 0] = remaining % 20
        remaining = remaining // 20

    # Level 1: uinal (mod 18) — calendar exception
    if n_levels >= 2:
        result[:, 1] = remaining % 18
        remaining = remaining // 18

    # Levels 2+: tun, katun, baktun (mod 20)
    for level in range(2, n_levels):
        result[:, level] = remaining % 20
        remaining = remaining // 20

    return result
