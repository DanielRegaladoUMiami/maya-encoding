# Getting Started

## Installation

Install from PyPI:

```bash
pip install maya-encoding
```

For visualization support:

```bash
pip install maya-encoding[viz]
```

For running benchmarks:

```bash
pip install maya-encoding[benchmarks]
```

## Basic Usage

### VFD Encoding

The VFDEncoder transforms numeric features into multi-scale vigesimal representations:

```python
import numpy as np
from maya_encoding import VFDEncoder

# Numeric features
X = np.array([[100], [250], [347], [0]])

# Encode with 2 vigesimal levels
enc = VFDEncoder(n_levels=2, components="full", normalize=True)
X_encoded = enc.fit_transform(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {X_encoded.shape}")
print(f"Feature names: {list(enc.get_feature_names_out())}")
```

### Maya Calendar Encoding

The MayaCalendarEncoder transforms dates into Maya calendar cycle features:

```python
import numpy as np
from maya_encoding import MayaCalendarEncoder

# Date strings
dates = np.array(["2024-01-01", "2024-06-15", "2024-12-21"])

# Encode with Tzolk'in and Haab' components
mce = MayaCalendarEncoder(
    components=["tzolkin", "haab"],
    cyclical=True,
)
features = mce.fit_transform(dates)

print(f"Input shape: {dates.shape}")
print(f"Output shape: {features.shape}")
```

### Using in scikit-learn Pipelines

Both encoders work seamlessly in scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
    ("vfd", VFDEncoder(components="full")),
    ("rf", RandomForestRegressor(n_estimators=100)),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

## Handling Special Values

### Negative Numbers

VFDEncoder supports three strategies for negative numbers:

```python
# abs_sign: separate sign feature + absolute value encoding
enc = VFDEncoder(handle_negative="abs_sign")

# shift: shift all values to be non-negative
enc = VFDEncoder(handle_negative="shift")

# error: raise ValueError if negatives found
enc = VFDEncoder(handle_negative="error")
```

### Float Values

Three strategies for handling floating-point numbers:

```python
# scale: multiply by auto-detected scale factor
enc = VFDEncoder(handle_float="scale")

# round: round to nearest integer
enc = VFDEncoder(handle_float="round")

# integer_part: use only integer part
enc = VFDEncoder(handle_float="integer_part")
```

## Next Steps

- [VFD Guide](guide/vfd.md) for in-depth VFD documentation
- [MCE Guide](guide/mce.md) for calendar encoding details
- [API Reference](api/vfd.md) for complete API docs
