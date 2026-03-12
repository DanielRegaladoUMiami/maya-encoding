# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-12

### Added
- `VFDEncoder`: Vigesimal Feature Decomposition for numeric features
  - Full, lite, and bars_dots component modes
  - Automatic level detection and scale factor inference
  - Negative and float handling strategies
  - sklearn pipeline compatible (fit/transform/inverse_transform)
- `MayaCalendarEncoder`: Maya Calendar Encoding for temporal features
  - Tzolk'in (260-day), Haab' (365-day), and Long Count components
  - Hierarchical and flat encoding modes
  - Optional sine/cosine cyclical encoding
  - GMT and custom epoch support
  - Wayeb' period detection
- Core mathematical functions
  - Vigesimal number system (encode/decode/decompose)
  - Maya calendar conversions (Gregorian to Tzolk'in, Haab', Long Count)
  - Vectorized operations with numpy
- Visualization
  - `plot_maya_number()`: Render numbers as Maya glyphs
  - `render_maya_text()`: Text-based Maya numeral representation
- Benchmark suite comparing VFD and MCE against baseline encodings
- Full test suite with >90% coverage
