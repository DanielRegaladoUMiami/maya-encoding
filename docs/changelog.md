# Changelog

## v0.2.1 (2026-03-21)

### Fixed

- Benchmarks now include `passthrough=True` variants — tree-based models show proper performance
- Binary encoding data leakage fixed in benchmark script (MinMaxScaler was fitting on test data)
- Visualization module `__init__.py` now properly exports `plot_maya_number`, `plot_maya_grid`, `render_maya_text`
- Fixed `plot_maya_number` / `plot_maya_grid` return type handling in notebook examples

### Added

- "Results at a Glance" section in README and docs homepage with benchmark tables
- Synthetic data fallback in fraud detection notebook (works without Kaggle credentials)
- Matplotlib visualization demos in VFD deep dive notebook
- Benchmark notebook rewritten with 5 encoding strategies × 4 models + MCE temporal analysis
- All 6 example notebooks listed in README

---

## v0.2.0 (2026-03-12)

### Features

- **VFDEncoder: passthrough mode** — New `passthrough=True` parameter keeps original
  input features alongside VFD-encoded features. This improves performance with
  tree-based models (RandomForest, GradientBoosting, XGBoost) that benefit from
  both the raw signal and multi-scale VFD decomposition.

### Fixes

- Fixed `gregorian_to_jdn()` usage in example notebooks (was passing 3 args instead of 1)
- Fixed Haab' and Long Count output format in notebook examples
- Fixed docs URL in notebook 01
- Removed compromised polyfill.io CDN from MkDocs config

---

## v0.1.0 (2026-03-12)

Initial release.

### Features

- **VFDEncoder**: Vigesimal Feature Decomposition transformer
  - Three component modes: full, lite, bars_dots
  - Auto-detection of vigesimal levels
  - Normalization support
  - Negative number handling (abs_sign, shift, error)
  - Float handling (scale, round, integer_part)
  - Inverse transform support
- **MayaCalendarEncoder**: Maya Calendar temporal encoding
  - Tzolk'in (260-day sacred calendar) features
  - Haab' (365-day solar calendar) features
  - Long Count (linear day count) features
  - Cyclical sin/cos encoding
  - Wayeb' binary flag
  - Configurable epoch
- **Core functions**: to_vigesimal, from_vigesimal, to_bars_dots, maya_decompose
- **Calendar functions**: Gregorian ↔ JDN ↔ Maya calendar conversions
- **Visualization**: Text and matplotlib rendering of Maya numbers
- Full scikit-learn compatibility (Pipeline, clone, get_params)
- 118 tests passing across Python 3.9-3.12
