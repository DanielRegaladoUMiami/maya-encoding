# Maya Calendar Encoder Guide

## The Maya Calendar Systems

The ancient Maya developed some of the most sophisticated calendar systems in history. The `MayaCalendarEncoder` leverages three of these systems to create temporal features with non-standard periodicities.

### Tzolk'in (Sacred Calendar)

The Tzolk'in is a 260-day cycle formed by the interlocking of two sub-cycles:

- **13 numbers** (1-13)
- **20 day names** (Imix', Ik', Ak'bal, K'an, Chikchan, Kimi, Manik', Lamat, Muluk, Ok, Chuwen, Eb', B'en, Ix, Men, Kib', Kab'an, Etz'nab', Kawak, Ajaw)

Since 13 and 20 are coprime, the Tzolk'in produces 260 unique date combinations before repeating.

### Haab' (Solar Calendar)

The Haab' is a 365-day cycle consisting of:

- **18 months** of 20 days each (numbered 0-19)
- **5 Wayeb' days** (considered unlucky)

Total: 18 × 20 + 5 = 365 days.

### Long Count

The Long Count is a linear day count from the Maya epoch (August 11, 3114 BCE in the GMT correlation). It uses a mixed-radix system:

| Unit | Days | Multiplier |
|------|------|------------|
| K'in | 1 | 1 |
| Uinal | 20 | 20 |
| Tun | 360 | 18 × 20 |
| K'atun | 7,200 | 20 × 360 |
| B'ak'tun | 144,000 | 20 × 7,200 |

Note the exception at the tun level (18 instead of 20) to approximate the solar year.

## Encoding Strategies

### Tzolk'in Encoding

Two modes for Tzolk'in features:

- **separate**: Number (1-13) and name index (0-19) as separate features
- **combined**: Single position index (0-259)

### Haab' Encoding

Two modes for Haab' features:

- **hierarchical**: Month index and day-of-month encoded with vigesimal bars/dots
- **flat**: Single day-of-year (0-364)

### Cyclical Encoding

When `cyclical=True`, numeric features are transformed into sin/cos pairs:

$$\text{sin\_feat} = \sin\left(\frac{2\pi \cdot \text{value}}{\text{period}}\right)$$
$$\text{cos\_feat} = \cos\left(\frac{2\pi \cdot \text{value}}{\text{period}}\right)$$

This ensures smooth transitions at cycle boundaries (e.g., day 259 → day 0 in Tzolk'in).

### Wayeb' Flag

When `wayeb_flag=True`, a binary feature indicates whether the date falls in the 5-day Wayeb' period.

## Usage Examples

```python
from maya_encoding import MayaCalendarEncoder
import numpy as np

dates = np.array(["2024-01-01", "2024-06-15"])

# Tzolk'in with cyclical encoding
enc = MayaCalendarEncoder(
    components=["tzolkin"],
    cyclical=True,
    tzolkin_encoding="separate",
)
features = enc.fit_transform(dates)

# Full encoding
enc_full = MayaCalendarEncoder(
    components=["tzolkin", "haab", "long_count"],
    cyclical=True,
    wayeb_flag=True,
    long_count_levels=3,
)
features_full = enc_full.fit_transform(dates)
```

## When to Use MCE

MCE is most useful when:

- Your time series has non-standard periodicities (not aligned with weeks/months)
- You want cycle features that don't depend on Gregorian calendar structure
- You're looking for features with coprime relationships (13 and 20)
- You want to enrich standard temporal features with orthogonal cycle information
