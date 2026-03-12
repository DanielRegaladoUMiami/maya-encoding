# Mathematical Background

## Vigesimal Number System

### Base-20 Decomposition

Any non-negative integer $n$ can be represented in base 20 as:

$$n = \sum_{i=0}^{L-1} d_i \cdot 20^i$$

where $d_i \in \{0, 1, \ldots, 19\}$ are the vigesimal digits and $L$ is the number of levels.

### Bars and Dots

Each vigesimal digit $d$ (0-19) is further decomposed:

$$\text{bars}(d) = \lfloor d / 5 \rfloor \in \{0, 1, 2, 3\}$$

$$\text{dots}(d) = d \mod 5 \in \{0, 1, 2, 3, 4\}$$

This creates a two-level hierarchy within each digit position.

### Feature Vector

For `components="full"` with $L$ levels, each scalar produces a $3L$-dimensional feature vector:

$$\mathbf{f}(n) = [d_0, b_0, \dot{d}_0, d_1, b_1, \dot{d}_1, \ldots, d_{L-1}, b_{L-1}, \dot{d}_{L-1}]$$

## Maya Calendar Mathematics

### Tzolk'in Cycle

The Tzolk'in position is determined by two independent cycles:

$$\text{number} = ((J - J_0) \mod 13) + 1$$

$$\text{name\_index} = (J - J_0) \mod 20$$

where $J$ is the Julian Day Number and $J_0 = 584283$ is the GMT correlation epoch.

Since $\gcd(13, 20) = 1$, the combined cycle length is $13 \times 20 = 260$ days (by the Chinese Remainder Theorem).

### Haab' Calendar

The Haab' position within its 365-day cycle:

$$\text{day\_of\_year} = (J - J_0) \mod 365$$

$$\text{month} = \lfloor \text{day\_of\_year} / 20 \rfloor$$

$$\text{day\_of\_month} = \text{day\_of\_year} \mod 20$$

The 5 Wayeb' days occur when month = 18 (the 19th "month").

### Long Count (Mixed Radix)

The Long Count uses a modified vigesimal system:

$$J - J_0 = k_0 + 20 \cdot k_1 + 360 \cdot k_2 + 7200 \cdot k_3 + 144000 \cdot k_4$$

The multipliers are:

| Level | Name | Radix | Cumulative |
|-------|------|-------|------------|
| 0 | K'in | 20 | 1 |
| 1 | Uinal | 18 | 20 |
| 2 | Tun | 20 | 360 |
| 3 | K'atun | 20 | 7,200 |
| 4 | B'ak'tun | 20 | 144,000 |

Note the exception at level 1 → 2: the multiplier is 18 (not 20) so that a Tun ≈ 360 days, approximating the solar year.

### Cyclical Encoding

For a component with period $P$ and value $v$:

$$\text{sin} = \sin\left(\frac{2\pi v}{P}\right), \quad \text{cos} = \cos\left(\frac{2\pi v}{P}\right)$$

This maps the cyclic variable to a unit circle, ensuring $v = 0$ and $v = P$ map to the same point, which is important for smooth feature transitions at cycle boundaries.

### Calendar Round

The Calendar Round combines Tzolk'in (260 days) and Haab' (365 days):

$$\text{CR period} = \text{lcm}(260, 365) = 18{,}980 \text{ days} \approx 52 \text{ years}$$

This 52-year cycle was the longest period the Maya could reference without the Long Count.
