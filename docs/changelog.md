# Changelog

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
