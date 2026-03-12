# VFD Encoder Guide

## How Vigesimal Feature Decomposition Works

The Maya vigesimal (base-20) number system represents any non-negative integer as a sequence of digits from 0 to 19. Each digit is further decomposed into **bars** (groups of 5) and **dots** (remainder), creating a natural hierarchical structure.

### Decomposition Example

The number **347** in vigesimal:

```
347 = 7 × 1 + 17 × 20
```

Level 0 (ones digit = 7): bars = 1, dots = 2
Level 1 (twenties digit = 17): bars = 3, dots = 2

This gives us 6 features: `[7, 1, 2, 17, 3, 2]`

### Why This Is Useful for ML

Standard numeric features encode magnitude on a single scale. VFD decomposes numbers into multiple scales, where:

- **Digits** capture the position within each base-20 level
- **Bars** capture groups of 5 within each level
- **Dots** capture the remainder within each group

This is analogous to how hour/minute/second decomposition helps time features, but generalized to the base-20 system.

## Component Modes

### Full Mode (default)

Produces `3 × n_levels` features per input column: digit, bars, dots at each level.

```python
enc = VFDEncoder(n_levels=2, components="full")
# Features: x0_L0_digit, x0_L0_bars, x0_L0_dots,
#           x0_L1_digit, x0_L1_bars, x0_L1_dots
```

### Lite Mode

Produces `n_levels` features per input column: only digits.

```python
enc = VFDEncoder(n_levels=2, components="lite")
# Features: x0_L0_digit, x0_L1_digit
```

### Bars-Dots Mode

Produces `2 × n_levels` features per input column: bars and dots only.

```python
enc = VFDEncoder(n_levels=2, components="bars_dots")
# Features: x0_L0_bars, x0_L0_dots, x0_L1_bars, x0_L1_dots
```

## Auto-Detection

When `n_levels="auto"` (default), the encoder automatically determines the number of vigesimal levels needed to represent the maximum value in the training data.

## Normalization

When `normalize=True`, features are scaled to [0, 1]:

- Digits: divided by 19
- Bars: divided by 3
- Dots: divided by 4

## Inverse Transform

The encoder supports inverse transformation for `full` and `lite` modes:

```python
enc = VFDEncoder(n_levels=2, normalize=False)
encoded = enc.fit_transform(X)
reconstructed = enc.inverse_transform(encoded)
# reconstructed ≈ X (exact for integers)
```
