# When to Use Maya Encoding

Choosing the right encoder depends on your data type and what patterns you want to capture. This guide helps you decide when Maya Encoding adds value — and when it doesn't.

## VFDEncoder — Numeric Feature Decomposition

The Maya number system was built for counting tangible things: days, people, tribute, cacao beans. The VFDEncoder inherits that strength — it decomposes numbers into a multi-scale hierarchy of digits (×20), bars (×5), and dots (×1), giving your model structure that raw numbers don't provide.

### Strong Fit

**Discrete, count-based data** is where VFD shines brightest:

- **Retail / inventory**: units sold, stock counts, order quantities
- **Event counts**: website visits, disease cases, incidents per day
- **Scores and ratings**: test scores, survey responses, rankings
- **Financial data with natural rounding**: prices ending in 0 or 5, quantities in multiples of 5 or 20

In these domains, the bars/dots decomposition often captures meaningful groupings — a count of 347 items becomes `17×20 + 7`, revealing structure at multiple scales.

**Any numeric feature in a linear model** benefits from VFD. Our benchmarks show consistent +3–4% R² improvement for Linear Regression, Ridge, and Lasso. The multi-scale features give linear models access to non-linear patterns they can't learn from raw values alone.

### Acceptable Fit

**Continuous measurements** (temperature, weight, distance) can still benefit, but the advantage is smaller. The encoder handles floats via automatic scaling to integers before decomposition, but the bars/dots split may not capture as meaningful structure in continuous data.

**Tree-based models** (Random Forest, Gradient Boosting, XGBoost) already learn numeric thresholds natively. Using VFD as a *replacement* for original features can hurt performance (−7 to −10% R² in benchmarks). The solution: use `passthrough=True` to keep original features alongside VFD output.

```python
from maya_encoding import VFDEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# For tree-based models, always use passthrough
pipe = Pipeline([
    ('encode', VFDEncoder(passthrough=True)),
    ('model', GradientBoostingRegressor())
])
```

### Not a Good Fit

- **High-cardinality categorical data** — VFD is for numeric features, not categories
- **Very small datasets** (< 100 samples) — the extra features may cause overfitting
- **Already well-engineered features** — if your features are already multi-scale (e.g., you have both `price` and `price_bucket`), VFD adds redundancy

### Handling Edge Cases

The ancient Maya had no negative numbers, fractions, or decimals. The VFDEncoder handles all of these through preprocessing:

| Input type | Strategy | What it does |
|-----------|----------|-------------|
| Negatives | `handle_negative='abs_sign'` | Adds binary sign feature + encodes absolute value |
| Negatives | `handle_negative='shift'` | Shifts all values to be non-negative |
| Floats | `handle_float='scale'` | Auto-detects decimal precision, scales to integers |
| Floats | `handle_float='round'` | Rounds to nearest integer |

These are not limitations — they are design choices that expand the original system while preserving its mathematical structure.

---

## MayaCalendarEncoder — Temporal Feature Engineering

The Maya calendar cycles (13d, 20d, 260d, 365d) were empirically calibrated over millennia to track agricultural, biological, and astronomical patterns in tropical Mesoamerica.

### Strong Fit

**Tropical and biological time series** is the home turf:

- **Agricultural forecasting**: crop yields, commodity prices, planting/harvest cycles
- **Epidemiological data**: disease outbreaks with ~13-day vector reproduction cycles
- **Climate and weather data**: especially in tropical latitudes (±20°)
- **Biological rhythms**: any process aligned with ~260-day or ~365-day cycles

**Any time series with unexplained seasonal variance** — if standard Gregorian features (day-of-week, month) leave residual patterns, Maya calendar cycles may capture what they miss because they operate on *different* frequencies.

### Why These Cycles Matter

The 260-day Tzolk'in simultaneously correlates with:

- **Human gestation** (~266 days average, with the Maya approximation of 260)
- **Maize growing cycle** in Mesoamerica (planting to harvest)
- **Nine lunar months** (9 × 29.5 ≈ 265.5 days)
- **Zenith sun passages** at 14.8°N latitude (the interval between the two annual events)

These are not arbitrary numbers — they are frequencies refined by over 3,000 years of empirical observation of tropical biological and astronomical cycles.

### The Mathematical Argument

Beyond the domain-specific correlations, there is a pure mathematical reason to use Maya calendar features: **coprime periods produce orthogonal cycles**.

The Gregorian calendar's common features share factors:

- Day-of-week (7), month (12), year (365) — `gcd(7, 364) = 7`

The Maya periods are coprime:

- 13 and 20: `gcd(13, 20) = 1` → 260 unique Tzolk'in combinations
- 260 and 365: `gcd(260, 365) = 5` → 18,980-day Calendar Round

This means Maya features capture variation along *different axes* than Gregorian features. They're not replacements — they're complements that expand the feature space with orthogonal information.

```python
from maya_encoding import MayaCalendarEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Combine Maya cycles with standard temporal features
mce = MayaCalendarEncoder(
    components=['tzolkin', 'haab'],
    cyclical=True,   # sin/cos for smooth boundaries
    wayeb_flag=True,  # binary flag for 5-day Wayeb' period
)
```

### Acceptable Fit

**General time series** in non-tropical domains — the coprime period structure provides orthogonal features regardless of geography, but the advantage may be marginal if your data doesn't have cycles near 13, 20, or 260 days.

### Not a Good Fit

- **Sub-daily data** — Maya calendars are day-resolution; for hourly/minute patterns, use other encoders
- **Very short time series** (< 260 days) — not enough data to capture a full Tzolk'in cycle
- **Purely trend-based data** — if your signal is monotonic with no cyclical component, calendar features won't help

---

## Decision Flowchart

```
Is your data numeric (not temporal)?
├── Yes → Is it discrete / count-based?
│   ├── Yes → VFDEncoder ✓ (strong fit)
│   └── No → VFDEncoder with passthrough=True (acceptable fit)
└── No → Is it date/timestamp data?
    ├── Yes → Does it span > 260 days?
    │   ├── Yes → Is it tropical/biological?
    │   │   ├── Yes → MayaCalendarEncoder ✓ (strong fit)
    │   │   └── No → MayaCalendarEncoder (acceptable fit, try it)
    │   └── No → Standard temporal encoding may be sufficient
    └── No → Maya Encoding is not applicable
```

## Combining Both Encoders

For datasets with both numeric and temporal features, you can use both encoders via `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from maya_encoding import VFDEncoder, MayaCalendarEncoder

preprocessor = ColumnTransformer([
    ('vfd', VFDEncoder(passthrough=True), numeric_cols),
    ('mce', MayaCalendarEncoder(cyclical=True), date_cols),
])
```

See the [examples](https://github.com/DanielRegaladoUMiami/maya-encoding/tree/main/examples) for complete working pipelines.

---

## Applied Use Cases

### Fraud Detection

Transaction amounts carry structural information that raw values hide. Legitimate transactions tend to be multiples of 20 (ATM withdrawals), round numbers (wire transfers), or charm-priced (retail). Fraudulent transactions often break these patterns.

VFD decomposes amounts into multi-scale features that capture this structure automatically. In our benchmark on the [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset, **isolating only the Amount feature**:

| Encoding | F1 | ROC AUC |
|----------|-----|---------|
| Raw Amount | 0.076 | 0.441 |
| VFD Amount (passthrough) | 0.096 | 0.642 |

VFD nearly triples the AUC from a single feature by separating magnitude tiers from residue patterns — without any manual feature engineering.

→ [Full notebook: Fraud Detection with VFD](https://github.com/DanielRegaladoUMiami/maya-encoding/blob/main/examples/05_fraud_detection.ipynb)

### Pricing & Demand Prediction

Retail prices are not random. They cluster at psychological thresholds ($49.99 vs $50.00), use charm pricing (.99, .95), and create non-linear demand responses at round-number boundaries. VFD's level structure naturally captures these thresholds:

- **L0 (ones)**: within-bracket position — captures charm pricing (.99 vs .00)
- **L1 (twenties)**: which $20-bracket — steps at each price tier boundary
- **L2 (four-hundreds)**: major price tier

In our demand prediction benchmark, VFD with passthrough improves Ridge regression from R² 0.978 → 0.980, giving linear models access to threshold effects they can't learn from raw prices alone.

→ [Full notebook: Pricing Analysis with VFD](https://github.com/DanielRegaladoUMiami/maya-encoding/blob/main/examples/06_pricing_analysis.ipynb)
