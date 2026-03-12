#!/bin/bash
# ==============================================================================
# Maya Encoding — GitHub Repository Setup Script
# ==============================================================================
# This script:
# 1. Initializes git in this directory
# 2. Creates a GitHub repository
# 3. Pushes the initial code
# 4. Creates all 17 GitHub issues organized by milestones
#
# Prerequisites:
# - gh CLI installed and authenticated (https://cli.github.com/)
# - git installed
#
# Usage:
#   cd maya-encoding
#   bash setup_github.sh
# ==============================================================================

set -e

# --- Configuration ---
REPO_NAME="maya-encoding"
REPO_DESC="Maya-inspired numerical encodings for machine learning: Vigesimal Feature Decomposition (VFD) and Maya Calendar Encoding (MCE)"
VISIBILITY="public"  # Change to "private" if desired

echo "=================================================="
echo "  Maya Encoding — GitHub Setup"
echo "=================================================="

# --- Step 1: Initialize git ---
echo ""
echo "[1/3] Initializing git repository..."
git init
git add -A
git commit -m "Initial commit: maya-encoding v0.1.0

Complete library implementing two sklearn-compatible encoders:
- VFDEncoder: Vigesimal Feature Decomposition for numeric features
- MayaCalendarEncoder: Maya Calendar Encoding for temporal features

Includes core engine, tests, benchmarks, CI/CD, and documentation."

# --- Step 2: Create GitHub repo and push ---
echo ""
echo "[2/3] Creating GitHub repository..."
gh repo create "$REPO_NAME" --description "$REPO_DESC" --"$VISIBILITY" --source=. --remote=origin --push

echo ""
echo "Repository created and code pushed!"

# --- Step 3: Create milestones ---
echo ""
echo "[3/3] Creating issues..."

# Helper function to create issues
create_issue() {
    local title="$1"
    local body="$2"
    local labels="$3"
    gh issue create --title "$title" --body "$body" --label "$labels" 2>/dev/null || \
    gh issue create --title "$title" --body "$body"
    echo "  Created: $title"
}

# Create labels
echo "Creating labels..."
gh label create "core" --color "0075ca" --description "Core mathematical functions" 2>/dev/null || true
gh label create "vfd" --color "7057ff" --description "Vigesimal Feature Decomposition" 2>/dev/null || true
gh label create "mce" --color "008672" --description "Maya Calendar Encoding" 2>/dev/null || true
gh label create "benchmark" --color "e4e669" --description "Benchmark experiments" 2>/dev/null || true
gh label create "infra" --color "d73a4a" --description "Infrastructure and CI/CD" 2>/dev/null || true
gh label create "docs" --color "0e8a16" --description "Documentation" 2>/dev/null || true
gh label create "viz" --color "fbca04" --description "Visualization" 2>/dev/null || true
gh label create "tests" --color "c5def5" --description "Test suite" 2>/dev/null || true
gh label create "release" --color "b60205" --description "Release management" 2>/dev/null || true
gh label create "paper" --color "d4c5f9" --description "Academic paper" 2>/dev/null || true
gh label create "size:S" --color "c2e0c6" --description "Small task (< 1 day)" 2>/dev/null || true
gh label create "size:M" --color "fef2c0" --description "Medium task (1-3 days)" 2>/dev/null || true
gh label create "size:L" --color "f9d0c4" --description "Large task (3-5 days)" 2>/dev/null || true

echo ""
echo "Creating Milestone 1: Core Engine issues..."

create_issue \
  "#1: Implement vigesimal number system core" \
  "## Description
Implement the core vigesimal (base-20) number system functions.

## File
\`src/maya_encoding/core/vigesimal.py\`

## Functions
- \`to_vigesimal(n: int, n_levels: int) -> list[int]\` — convert integer to vigesimal digits (LSB first)
- \`from_vigesimal(digits: list[int]) -> int\` — inverse (exact reconstruction)
- \`to_bars_dots(digit: int) -> tuple[int, int]\` — decompose digit 0-19 into (bars, dots)
- \`maya_decompose(n: int, n_levels: int) -> dict\` — full decomposition {digits, bars, dots}
- \`auto_n_levels(max_value: int) -> int\` — calculate needed levels
- \`maya_encode_array(values, n_levels, components, normalize)\` — vectorized numpy encoding

## Acceptance Criteria
- [ ] \`from_vigesimal(to_vigesimal(n, L)) == n\` for all n in [0, 160000]
- [ ] \`to_bars_dots\` correct for all digits 0-19 (exhaustive test)
- [ ] \`bars * 5 + dots == digit\` always holds
- [ ] Vectorized results match scalar decomposition
- [ ] All edge cases handled: 0, 1, 19, 20, 399, 400" \
  "core,size:M"

create_issue \
  "#2: Implement utility functions for negatives and floats" \
  "## Description
Input validation and preprocessing utilities.

## File
\`src/maya_encoding/core/utils.py\`

## Functions
- \`validate_input(X)\` — convert DataFrame/array to numpy 2D, validate for NaN/inf
- \`handle_negatives(values, strategy)\` — abs_sign, shift, or error
- \`handle_floats(values, strategy, scale_factor)\` — scale, round, or integer_part
- \`auto_scale_factor(values)\` — detect decimal precision, return 10^n
- \`get_feature_names(col_name, n_levels, components, has_sign)\` — descriptive names

## Acceptance Criteria
- [ ] \`auto_scale_factor([1.5, 2.25, 3.125])\` returns 1000
- [ ] \`auto_scale_factor([1.0, 2.0])\` returns 1
- [ ] abs_sign: [-5, 3] produces signs=[1,0], values=[5,3]
- [ ] NaN and inf inputs raise clear errors
- [ ] Feature names are descriptive and unique" \
  "core,size:S"

create_issue \
  "#3: Implement Maya calendar conversions" \
  "## Description
Convert Gregorian dates to the three Maya calendar systems.

## File
\`src/maya_encoding/core/calendar.py\`

## Functions
- \`gregorian_to_jdn(date)\` — Gregorian to Julian Day Number
- \`jdn_to_long_count(jdn, n_levels, epoch_jdn)\` — JDN to Long Count
- \`jdn_to_tzolkin(jdn)\` — JDN to (number 1-13, day_name 0-19)
- \`jdn_to_haab(jdn)\` — JDN to (month 0-18, day 0-19)
- \`is_wayeb(jdn)\` — check if date is in 5-day Wayeb period
- Vectorized versions: \`dates_to_jdn_array\`, \`jdn_array_to_tzolkin\`, etc.

## Key References
- GMT correlation: JDN 584283 (August 11, 3114 BCE)
- 2012-12-21 = 13.0.0.0.0 = 4 Ajaw 3 K'ank'in

## Acceptance Criteria
- [ ] 2012-12-21 correctly converts to 13.0.0.0.0, 4 Ajaw, 3 K'ank'in
- [ ] All 260 Tzolk'in combinations unique in a cycle
- [ ] Exactly 5 Wayeb' days per 365-day Haab' cycle
- [ ] Uinal->Tun boundary at 18 (not 20) — the calendar exception
- [ ] Accepts datetime, string, numpy datetime64, Unix timestamp" \
  "core,size:L"

create_issue \
  "#4: Package scaffolding and project config" \
  "## Description
Project configuration, packaging, and import structure.

## Files
- \`pyproject.toml\` — build config with hatchling, metadata, optional deps
- \`src/maya_encoding/__init__.py\` — public API exports
- \`src/maya_encoding/_version.py\` — version string
- All \`__init__.py\` files for subpackages

## Acceptance Criteria
- [ ] \`pip install -e .\` works
- [ ] \`from maya_encoding import VFDEncoder, MayaCalendarEncoder\` imports correctly
- [ ] \`maya_encoding.__version__\` returns '0.1.0'
- [ ] Optional deps install: \`pip install -e '.[viz]'\`, \`pip install -e '.[dev]'\`" \
  "infra,size:S"

echo ""
echo "Creating Milestone 2: Sklearn Transformers issues..."

create_issue \
  "#5: Implement VFDEncoder sklearn transformer" \
  "## Description
Full sklearn-compatible transformer for Vigesimal Feature Decomposition.

## File
\`src/maya_encoding/vfd/encoder.py\`

## Class: VFDEncoder(BaseEstimator, TransformerMixin)
- \`fit(X)\`: detect n_levels, scale_factor, validate input
- \`transform(X)\`: apply VFD to each column
- \`inverse_transform(X_encoded)\`: reconstruct original values
- \`get_feature_names_out()\`: descriptive names per column and component

## Parameters
n_levels, components (full/lite/bars_dots), normalize, handle_negative, handle_float, scale_factor

## Acceptance Criteria
- [ ] Pipeline with RandomForest works end-to-end
- [ ] DataFrame input → feature names include original column names
- [ ] Numpy array input → feature names use indices (f0, f1, ...)
- [ ] inverse_transform recovers original values (for normalize=False)
- [ ] Auto-detect n_levels and scale_factor in fit()
- [ ] Handles negatives and floats correctly" \
  "vfd,size:L"

create_issue \
  "#6: Implement MayaCalendarEncoder sklearn transformer" \
  "## Description
Sklearn-compatible transformer for Maya Calendar Encoding.

## File
\`src/maya_encoding/mce/encoder.py\`

## Class: MayaCalendarEncoder(BaseEstimator, TransformerMixin)
- \`fit(X)\`: validate dates, resolve epoch
- \`transform(X)\`: convert dates to Maya calendar features
- \`get_feature_names_out()\`: descriptive names

## Parameters
components, tzolkin_encoding, haab_encoding, long_count_levels, cyclical, epoch, wayeb_flag, normalize

## Acceptance Criteria
- [ ] Accepts datetime64, strings, Unix timestamps
- [ ] cyclical=True generates sin/cos pairs
- [ ] Wayeb flag correctly identifies the 5-day period
- [ ] Pipeline with XGBoost works end-to-end
- [ ] Feature names match output column count" \
  "mce,size:L"

create_issue \
  "#7: Complete test suite" \
  "## Description
Comprehensive tests for all modules.

## Files
- \`tests/test_vigesimal.py\` — vigesimal core functions
- \`tests/test_calendar.py\` — calendar conversions
- \`tests/test_vfd_encoder.py\` — VFDEncoder
- \`tests/test_mce_encoder.py\` — MayaCalendarEncoder
- \`tests/test_sklearn_compat.py\` — sklearn pipeline compatibility

## Acceptance Criteria
- [ ] >90% code coverage
- [ ] Roundtrip tests: encode(decode(x)) == x
- [ ] Edge cases: 0, NaN, inf, negatives, single sample
- [ ] Sklearn pipeline tests pass
- [ ] Parametrized tests for known Maya dates" \
  "tests,size:M"

echo ""
echo "Creating Milestone 3: Benchmark issues..."

create_issue \
  "#8: VFD benchmark — California Housing regression" \
  "## Description
Compare VFD against baseline encodings on California Housing dataset.

## Experiment Design
- **Encodings**: Decimal normalized, Binary, VFD-lite, VFD-full
- **Models**: LinearRegression, RandomForest, XGBoost, MLP
- **Protocol**: 5-fold CV, RMSE + MAE
- **Statistical test**: Wilcoxon signed-rank

## Acceptance Criteria
- [ ] All 16 combinations (4 encodings × 4 models) run successfully
- [ ] Results saved as JSON
- [ ] Summary table printed
- [ ] Statistical significance tested" \
  "benchmark,size:M"

create_issue \
  "#9: VFD benchmark — Classification" \
  "## Description
Test VFD on classification task using sklearn digits dataset.

## Experiment Design
- **Encodings**: Decimal normalized, Binary, VFD-lite, VFD-full
- **Models**: LogisticRegression, RandomForest, XGBoost, MLP
- **Protocol**: 5-fold stratified CV, F1-weighted
- **Statistical test**: Wilcoxon signed-rank

## Acceptance Criteria
- [ ] All combinations run successfully
- [ ] Results saved and summarized" \
  "benchmark,size:M"

create_issue \
  "#10: MCE benchmark — Time series forecasting" \
  "## Description
Compare Maya Calendar Encoding against sine/cosine on time series.

## Experiment Design
- **Synthetic data**: Known cycles at 13, 20, 365 days (aligned with Maya periods)
- **Encodings**: No temporal, Sine/cosine, MCE full, MCE + sine/cosine
- **Models**: XGBoost, MLP
- **Protocol**: Walk-forward validation, RMSE + MAE

## Acceptance Criteria
- [ ] Synthetic data generation with known ground truth
- [ ] Walk-forward validation respects temporal ordering
- [ ] MCE vs sine/cosine comparison for each model" \
  "benchmark,size:L"

create_issue \
  "#11: Ablation study" \
  "## Description
Analyze which VFD/MCE components contribute most.

## Experiments
- A1: VFD-full vs VFD-lite (do bars/dots add value?)
- A2: MCE per component (Tzolkin vs Haab vs Long Count)
- A3: Performance delta by model complexity
- A4: VFD scaling with number magnitude

## Acceptance Criteria
- [ ] Clear table showing component contribution
- [ ] Per-model analysis
- [ ] Summary conclusions" \
  "benchmark,size:M"

echo ""
echo "Creating Milestone 4: Documentation & Visualization issues..."

create_issue \
  "#12: Maya glyph visualization" \
  "## Description
Render numbers and dates as Maya glyphs using matplotlib.

## File
\`src/maya_encoding/visualization/glyphs.py\`

## Functions
- \`plot_maya_number(n, ax)\` — render with bars (━), dots (●), shell (◎)
- \`plot_maya_grid(numbers, cols)\` — grid of multiple numbers
- \`render_maya_text(n)\` — text-based rendering

## Acceptance Criteria
- [ ] Correct glyph for all digits 0-19
- [ ] Multi-level numbers render vertically (MSB on top)
- [ ] Grid layout works for multiple numbers
- [ ] Graceful ImportError if matplotlib not installed" \
  "viz,size:M"

create_issue \
  "#13: Example notebooks" \
  "## Description
Create 4 Jupyter notebooks demonstrating the library.

## Notebooks
1. \`01_vfd_quickstart.ipynb\` — basic VFD encoding, visualization, simple pipeline
2. \`02_mce_time_series.ipynb\` — calendar encoding for temporal data
3. \`03_vfd_vs_baselines.ipynb\` — head-to-head comparison with benchmark results
4. \`04_mce_vs_sine_cosine.ipynb\` — MCE vs standard temporal encoding

## Acceptance Criteria
- [ ] Each notebook runs end-to-end without errors
- [ ] Clear narrative explaining the concepts
- [ ] Visualizations included" \
  "docs,size:M"

create_issue \
  "#14: Documentation site" \
  "## Description
Full documentation with MkDocs.

## Pages
- Getting started
- VFD: theory + API reference
- MCE: theory + calendar systems + API reference
- Benchmark results
- README with badges and quickstart

## Acceptance Criteria
- [ ] MkDocs builds without errors
- [ ] API reference auto-generated from docstrings
- [ ] README has install instructions, quickstart, and badges" \
  "docs,size:M"

echo ""
echo "Creating Milestone 5: Packaging & Release issues..."

create_issue \
  "#15: CI/CD pipeline" \
  "## Description
GitHub Actions for continuous integration and deployment.

## Files
- \`.github/workflows/ci.yml\` — tests + linting on PR
- \`.github/workflows/publish.yml\` — PyPI publish on tag

## Acceptance Criteria
- [ ] Tests run on Python 3.9-3.12
- [ ] Ruff linting passes
- [ ] Coverage report uploaded
- [ ] PyPI publish triggers on version tag" \
  "infra,size:S"

create_issue \
  "#16: PyPI release v0.1.0" \
  "## Description
Prepare and publish the first release to PyPI.

## Tasks
- Tag v0.1.0
- Verify \`pip install maya-encoding\` works
- Verify imports and quickstart code
- Update CHANGELOG.md

## Acceptance Criteria
- [ ] Package installable from PyPI
- [ ] All imports work
- [ ] README renders correctly on PyPI" \
  "release,size:S"

create_issue \
  "#17: Paper draft" \
  "## Description
Draft the academic paper for publication.

## Title
\"Vigesimal Feature Decomposition and Maya Calendar Encoding: Ancient Number Systems as Feature Engineering for Modern Machine Learning\"

## Sections
- Introduction
- Related Work (feature engineering, positional encoding, cultural computing)
- Method (VFD + MCE mathematical formulation)
- Experiments (from benchmarks)
- Results and Discussion
- Conclusion

## Target Venues
- NeurIPS/ICML workshop on representation learning
- AAAI (AI + cultural heritage track)
- Pattern Recognition Letters

## Acceptance Criteria
- [ ] All sections drafted
- [ ] Benchmark results tables included
- [ ] Related work section comprehensive" \
  "paper,size:L"

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "=================================================="
echo ""
echo "Repository: https://github.com/$(gh api user --jq .login)/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. cd maya-encoding"
echo "  2. pip install -e '.[dev]'"
echo "  3. pytest  (run tests)"
echo "  4. python benchmarks/run_vfd_benchmarks.py  (run benchmarks)"
echo ""
