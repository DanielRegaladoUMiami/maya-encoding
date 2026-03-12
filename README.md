# maya-encoding

[![CI](https://github.com/DanielRegaladoUMiami/maya-encoding/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielRegaladoUMiami/maya-encoding/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/maya-encoding)](https://pypi.org/project/maya-encoding/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/maya-encoding)](https://pypi.org/project/maya-encoding/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://danielregaladoumiami.github.io/maya-encoding/)

**Maya-inspired numerical encodings for machine learning.**

> **[Documentation](https://danielregaladoumiami.github.io/maya-encoding/)** · **[PyPI](https://pypi.org/project/maya-encoding/)** · **[Examples](examples/)**

Two scikit-learn compatible transformers that use the mathematical structure of the ancient Maya number system and calendar to create richer feature representations.

## Overview

| Encoder | Input | What it does | Use case |
|---------|-------|-------------|----------|
| **VFDEncoder** | Numeric features | Decomposes into base-20 digits, bars (÷5), dots (%5) | Multi-scale numeric patterns |
| **MayaCalendarEncoder** | Dates | Extracts Tzolk'in (260d), Haab' (365d), Long Count cycles | Temporal feature engineering |

## Installation

```bash
pip install maya-encoding
```

With optional dependencies:

```bash
pip install maya-encoding[viz]         # matplotlib visualization
pip install maya-encoding[benchmarks]  # xgboost, seaborn for benchmarks
pip install maya-encoding[dev]         # development tools (ruff, pytest)
```

## Quick Start

### VFD: Numeric Feature Encoding

```python
import numpy as np
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

How it works — the number **347** becomes:

```
347 = 17×20 + 7

Level 0 (ones):     digit=7,  bars=1, dots=2
Level 1 (twenties): digit=17, bars=3, dots=2

Feature vector: [7, 1, 2, 17, 3, 2]  →  normalized: [0.37, 0.33, 0.50, 0.89, 1.00, 0.50]
```

Three "zoom levels" per number: coarse magnitude (digits), medium grouping (bars), and fine residual (dots).

### Passthrough Mode: Best of Both Worlds

Use `passthrough=True` to keep original features alongside VFD features — ideal for tree-based models:

```python
# Original features + VFD features combined
pipe = Pipeline([
    ('encode', VFDEncoder(passthrough=True)),
    ('model', GradientBoostingRegressor())
])
```

### MCE: Temporal Feature Encoding

```python
import numpy as np
from maya_encoding import MayaCalendarEncoder

# Encode dates using Maya calendar cycles
encoder = MayaCalendarEncoder(
    components=['tzolkin', 'haab', 'long_count'],
    cyclical=True,  # sine/cosine for smooth cycle boundaries
)

dates = np.array(["2024-01-01", "2024-06-15", "2024-12-21"])
features = encoder.fit_transform(dates)
```

The Maya calendar provides interlocking cycles of coprime periods (13, 20, 260, 365, 360), capturing multi-scale temporal patterns that standard encoding requires manual period selection to achieve.

### Explore Maya Numbers

```python
from maya_encoding import maya_decompose, to_vigesimal, to_bars_dots

# Convert to vigesimal
digits = to_vigesimal(347)  # [7, 17] (LSB first)

# Full decomposition
info = maya_decompose(347)
# {'digits': [7, 17], 'bars': [1, 3], 'dots': [2, 2], 'n_levels': 2}

# Visualize
from maya_encoding.visualization.glyphs import render_maya_text
print(render_maya_text(347))
```

### Explore Maya Calendar

```python
from maya_encoding.core.calendar import (
    gregorian_to_jdn, jdn_to_tzolkin, jdn_to_haab, jdn_to_long_count
)

# December 21, 2012 — end of the 13th b'ak'tun
jdn = gregorian_to_jdn("2012-12-21")
print(jdn_to_tzolkin(jdn))     # (4, 19) → 4 Ajaw
print(jdn_to_haab(jdn))        # (13, 3) → month 13, day 3
print(jdn_to_long_count(jdn))  # (13, 0, 0, 0, 0) → 13.0.0.0.0
```

## When to Use Maya Encoding

| Encoder | Strong Fit | Acceptable Fit |
|---------|-----------|----------------|
| **VFDEncoder** | Discrete/count data (retail, events, scores), linear models | Continuous features with `passthrough=True` for tree models |
| **MayaCalendarEncoder** | Tropical/biological time series (agriculture, epidemiology, climate) | General time series with unexplained seasonal variance |

**VFD** decomposes numbers into a natural hierarchy — digits (×20), bars (×5), dots (×1). This is a *strict information superset*: the model gets multi-scale structure for free. Linear models see +3–4% R²; tree-based models benefit with `passthrough=True`.

**MCE** provides *orthogonal* cycles with coprime periods (13, 20, 260, 365) that capture patterns Gregorian features miss. The 260-day Tzolk'in correlates with human gestation, maize growing cycles, and tropical astronomical events.

→ **[Full guide: When to Use Maya Encoding](https://danielregaladoumiami.github.io/maya-encoding/guide/when-to-use/)**

## API Reference

### VFDEncoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_levels` | `'auto'` | Vigesimal levels (auto-detected from data) |
| `components` | `'full'` | `'full'`, `'lite'` (digits only), `'bars_dots'` |
| `normalize` | `True` | Normalize features to [0, 1] |
| `handle_negative` | `'abs_sign'` | `'abs_sign'`, `'shift'`, `'error'` |
| `handle_float` | `'scale'` | `'scale'`, `'round'`, `'integer_part'` |
| `passthrough` | `False` | Keep original features alongside VFD output |
| `scale_factor` | `'auto'` | Decimal precision auto-detection |

### MayaCalendarEncoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `components` | `['tzolkin', 'haab', 'long_count']` | Calendar systems to use |
| `tzolkin_encoding` | `'separate'` | `'separate'` (number + name) or `'combined'` (position 0-259) |
| `haab_encoding` | `'hierarchical'` | `'hierarchical'` (with bars/dots) or `'flat'` (day 0-364) |
| `long_count_levels` | `3` | 1–5: k'in, uinal, tun, k'atun, b'ak'tun |
| `cyclical` | `True` | Add sine/cosine pairs for smooth cycle boundaries |
| `epoch` | `'gmt'` | `'gmt'` (standard), `'spinden'`, or custom JDN |
| `wayeb_flag` | `True` | Binary flag for the 5-day Wayeb' period |

## Examples

See the [`examples/`](examples/) directory:

- [`01_quickstart.ipynb`](examples/01_quickstart.ipynb) — Basic VFD and MCE usage
- [`02_vfd_deep_dive.ipynb`](examples/02_vfd_deep_dive.ipynb) — Components, visualization, performance
- [`03_mce_temporal.ipynb`](examples/03_mce_temporal.ipynb) — Calendar systems and time series
- [`04_benchmark_results.ipynb`](examples/04_benchmark_results.ipynb) — Performance comparisons

## Development

```bash
git clone https://github.com/DanielRegaladoUMiami/maya-encoding.git
cd maya-encoding
pip install -e ".[dev]"
pytest          # Run 124 tests
ruff check .    # Lint
```

Run benchmarks:

```bash
pip install -e ".[benchmarks]"
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
  url = {https://github.com/DanielRegaladoUMiami/maya-encoding}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
