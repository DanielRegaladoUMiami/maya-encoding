# Maya Encoding

**Maya-inspired numerical encodings for machine learning.**

`maya-encoding` provides two scikit-learn compatible transformers that leverage the mathematical elegance of ancient Maya numerical and calendar systems for modern feature engineering.

## Encoders

### VFDEncoder (Vigesimal Feature Decomposition)

Transforms numeric features into hierarchical base-20 representations. Each number is decomposed into vigesimal digits, bars (÷5), and dots (%5) at multiple positional levels — creating multi-scale features that capture periodic and modular patterns.

```python
from maya_encoding import VFDEncoder
import numpy as np

X = np.array([[0], [7], [20], [347]])
enc = VFDEncoder(n_levels=2, components="full")
X_encoded = enc.fit_transform(X)
```

### MayaCalendarEncoder (Maya Calendar Encoding)

Transforms dates into features derived from three Maya calendar systems:

- **Tzolk'in** (260-day sacred calendar): 13 numbers × 20 day names
- **Haab'** (365-day solar calendar): 18 months × 20 days + 5 Wayeb'
- **Long Count**: Linear day count from the Maya epoch

```python
from maya_encoding import MayaCalendarEncoder
import numpy as np

dates = np.array(["2024-01-01", "2024-06-15", "2024-12-21"])
mce = MayaCalendarEncoder(components=["tzolkin", "haab"], cyclical=True)
features = mce.fit_transform(dates)
```

## Installation

```bash
pip install maya-encoding
```

## Results at a Glance

### VFD — California Housing Regression (R², 5-fold CV)

| Encoding | Linear Regression | Ridge | Random Forest | Gradient Boosting |
|----------|:-:|:-:|:-:|:-:|
| Raw + Scaled | 0.5530 | 0.5530 | 0.6561 | 0.6852 |
| VFD-lite | 0.5832 | 0.5812 | 0.5445 | 0.5742 |
| VFD-full | 0.5742 | 0.5723 | 0.5891 | 0.6184 |
| **VFD-lite + passthrough** | **0.5985** | **0.5968** | 0.6588 | 0.6899 |
| **VFD-full + passthrough** | 0.5908 | 0.5881 | **0.6615** | **0.6937** |

### MCE — Temporal Cycle Detection (R², synthetic data)

| Configuration | Train R² | Test R² |
|--------------|:-:|:-:|
| All components + cyclical | 0.9875 | **0.9146** |
| Tzolk'in only | 0.3656 | 0.0707 |
| Haab' only | 0.6212 | 0.5891 |

### Fraud Detection (F1, 5-fold stratified CV)

| Pipeline | Logistic Regression | Random Forest | Gradient Boosting |
|----------|:-:|:-:|:-:|
| Baseline (PCA) | 0.7082 | 0.8961 | 0.8729 |
| VFD (replace amount) | 0.6876 | 0.8971 | 0.8816 |
| **VFD + passthrough** | 0.6903 | **0.8993** | **0.8816** |

!!! tip "Rule of thumb"
    **Linear models** → use VFD directly. **Tree-based models** → always use `passthrough=True`.

## Quick Links

- [Getting Started](getting-started.md) — installation and first steps
- [When to Use](guide/when-to-use.md) — choosing the right encoder for your data
- [VFD Guide](guide/vfd.md) — understanding vigesimal feature decomposition
- [MCE Guide](guide/mce.md) — temporal encoding with Maya calendars
- [API Reference](api/vfd.md) — full API documentation
- [Mathematical Background](math-background.md) — the math behind the encoders
