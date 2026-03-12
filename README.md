# maya-encoding

[![CI](https://github.com/danielregalado/maya-encoding/actions/workflows/ci.yml/badge.svg)](https://github.com/danielregalado/maya-encoding/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/maya-encoding.svg)](https://badge.fury.io/py/maya-encoding)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Maya-inspired numerical encodings for machine learning.**

Two sklearn-compatible transformers that use the mathematical structure of the ancient Maya number system and calendar to create richer feature representations:

- **VFDEncoder** (Vigesimal Feature Decomposition) — Decomposes numbers into the Maya base-20 system with bars (÷5) and dots (%5), giving models multi-scale numerical structure for free.
- **MayaCalendarEncoder** (Maya Calendar Encoding) — Converts dates into features from the Tzolk'in (260-day), Haab' (365-day), and Long Count calendars, providing interlocking cyclical patterns at multiple time scales.

## Installation

```bash
pip install maya-encoding
```

With visualization support:
```bash
pip install maya-encoding[viz]
```

## Quick Start

### VFD: Numeric Feature Encoding

```python
from maya_encoding import VFDEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# VFD decomposes numbers into vigesimal digits, bars, and dots
encoder = VFDEncoder(components='full')

# Works seamlessly in sklearn pipelines
pipe = Pipeline([
    ('encode', VFDEncoder()),
    ('model', RandomForestRegressor())
])
pipe.fit(X_train, y_train)
```

How it works: the number **347** becomes:
```
347 = 17×20 + 7

Level 0 (ones):   digit=7,  bars=1, dots=2
Level 1 (twenties): digit=17, bars=3, dots=2

Feature vector: [7, 1, 2, 17, 3, 2] (or normalized to [0,1])
```

This gives the model three "zoom levels" per number — coarse magnitude (digits), medium grouping (bars), and fine residual (dots).

### MCE: Temporal Feature Encoding

```python
from maya_encoding import MayaCalendarEncoder

# Encode dates using Maya calendar cycles
encoder = MayaCalendarEncoder(
    components=['tzolkin', 'haab', 'long_count'],
    cyclical=True,  # Add sine/cosine for smooth cycle boundaries
)

X_temporal = encoder.fit_transform(df['date'])
```

The Maya calendar provides interlocking cycles of coprime periods (13, 20, 260, 365, 360), capturing multi-scale temporal patterns that standard sine/cosine encoding requires manual period selection to achieve.

## Why Maya Encoding?

**The problem:** When you feed a number like "347" to a model, it knows nothing about its structure. It has to learn from scratch that 347 is close to 350, "large" compared to 5, and divisible in certain ways.

**The solution:** The Maya vigesimal system naturally decomposes numbers into a hierarchy:
- **Vigesimal digits** (×20): coarse magnitude
- **Bars** (×5): medium grouping
- **Dots** (×1): fine residual

This is a strict information superset — the model can ignore the extra features via regularization if they're not useful, but gets multi-scale structure for free if they are.

For temporal data, the Maya calendar's three interlocking cycles (Tzolk'in 260-day, Haab' 365-day, Long Count) provide coprime-period features that capture patterns standard time encodings miss.

## API Reference

### VFDEncoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_levels` | `'auto'` | Vigesimal levels (auto-detected from data) |
| `components` | `'full'` | `'full'`, `'lite'` (digits only), `'bars_dots'` |
| `normalize` | `True` | Normalize to [0,1] |
| `handle_negative` | `'abs_sign'` | `'abs_sign'`, `'shift'`, `'error'` |
| `handle_float` | `'scale'` | `'scale'`, `'round'`, `'integer_part'` |
| `scale_factor` | `'auto'` | Auto-detected from decimal precision |

### MayaCalendarEncoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `components` | `['tzolkin', 'haab', 'long_count']` | Calendar systems to use |
| `tzolkin_encoding` | `'separate'` | `'separate'` (2 features) or `'combined'` (1 feature) |
| `haab_encoding` | `'hierarchical'` | `'hierarchical'` (with bars/dots) or `'flat'` |
| `long_count_levels` | `3` | 1-5: kin, uinal, tun, katun, baktun |
| `cyclical` | `True` | Add sine/cosine pairs |
| `epoch` | `'gmt'` | `'gmt'` (standard), `'spinden'`, or custom JDN |
| `wayeb_flag` | `True` | Binary feature for the 5-day Wayeb' period |

## Visualization

```python
from maya_encoding.visualization.glyphs import plot_maya_number, render_maya_text

# Text rendering
print(render_maya_text(347))

# Matplotlib rendering
plot_maya_number(347)
```

## Development

```bash
git clone https://github.com/danielregalado/maya-encoding.git
cd maya-encoding
pip install -e ".[dev]"
pytest
```

Run benchmarks:
```bash
python benchmarks/run_vfd_benchmarks.py
python benchmarks/run_mce_benchmarks.py
```

## Citation

If you use maya-encoding in your research, please cite:

```bibtex
@software{regalado2026maya,
  author = {Regalado, Daniel},
  title = {maya-encoding: Maya-Inspired Numerical Encodings for Machine Learning},
  year = {2026},
  url = {https://github.com/danielregalado/maya-encoding}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
