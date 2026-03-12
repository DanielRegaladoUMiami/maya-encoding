# Visualization

The `maya_encoding.visualization` module provides tools for rendering Maya numbers visually.

## Text Rendering

```python
from maya_encoding.visualization.glyphs import render_maya_text

# Display any number as Maya glyphs
print(render_maya_text(347))
```

Symbols used:

- `●` — dot (value 1)
- `━` — bar (value 5)
- `◎` — shell (zero)

## Matplotlib Rendering

```python
from maya_encoding.visualization.glyphs import plot_maya_number

# Plot a single number
fig = plot_maya_number(347)

# Plot a grid of numbers
from maya_encoding.visualization.glyphs import plot_maya_grid
fig = plot_maya_grid([0, 7, 19, 20, 347, 8000])
```

## Installation

Visualization requires matplotlib:

```bash
pip install maya-encoding[viz]
```
