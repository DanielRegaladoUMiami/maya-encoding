"""Visualization of Maya numerals using matplotlib.

Renders numbers as Maya glyphs with dots (●), bars (━), and shells (◎ for zero).
"""

from __future__ import annotations

from maya_encoding.core.vigesimal import to_bars_dots, to_vigesimal


def render_maya_text(n: int, n_levels: int | None = None) -> str:
    """Render a number as a text-based Maya numeral (vertical, top = MSB).

    Parameters
    ----------
    n : int
        Non-negative integer to render.
    n_levels : int or None
        Number of vigesimal levels. Auto-detected if None.

    Returns
    -------
    str
        Multi-line string representation with dots and bars.

    Examples
    --------
    >>> print(render_maya_text(0))
    ◎

    >>> print(render_maya_text(347))
    ●●
    ━━━
    ━━━
    ━━━
    ───
    ●●
    ━

    """
    if n < 0:
        raise ValueError(f"Cannot render negative number: {n}")

    digits = to_vigesimal(n, n_levels)

    # Render from MSB (top) to LSB (bottom)
    levels = []
    for digit in reversed(digits):
        if digit == 0:
            levels.append("  ◎")
        else:
            bars, dots = to_bars_dots(digit)
            lines = []
            if dots > 0:
                lines.append(" " + "●" * dots)
            for _ in range(bars):
                lines.append(" ━━━")
            levels.append("\n".join(lines))

    return "\n───\n".join(levels)


def plot_maya_number(n: int, n_levels: int | None = None, ax=None, **kwargs):
    """Plot a number as a Maya glyph using matplotlib.

    Each vigesimal level is drawn vertically (highest level on top).
    Dots are circles, bars are horizontal rectangles, zero is a shell symbol.

    Parameters
    ----------
    n : int
        Non-negative integer to render.
    n_levels : int or None
        Number of vigesimal levels.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. If None, creates a new figure.
    **kwargs
        Additional keyword arguments passed to the figure creation.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the rendered glyph.

    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install maya-encoding[viz]"
        )

    digits = to_vigesimal(n, n_levels)

    if ax is None:
        fig_height = max(2, len(digits) * 1.5)
        fig, ax = plt.subplots(1, 1, figsize=(2.5, fig_height), **kwargs)

    ax.set_xlim(-1.5, 1.5)
    total_height = len(digits) * 3
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(str(n), fontsize=14, fontweight="bold", pad=10)

    # Colors
    dot_color = "#8B4513"   # Saddle brown
    bar_color = "#8B4513"
    zero_color = "#C0C0C0"

    for level_idx, digit in enumerate(digits):
        # Y base for this level (LSB at bottom)
        y_base = level_idx * 3

        if digit == 0:
            # Draw shell symbol (circle with cross)
            shell = patches.Circle((0, y_base + 1), 0.5, fill=False,
                                   edgecolor=zero_color, linewidth=2)
            ax.add_patch(shell)
            ax.plot([-0.3, 0.3], [y_base + 1, y_base + 1],
                    color=zero_color, linewidth=1.5)
            ax.plot([0, 0], [y_base + 0.7, y_base + 1.3],
                    color=zero_color, linewidth=1.5)
        else:
            bars, dots = to_bars_dots(digit)
            y_cursor = y_base + 0.3

            # Draw bars (bottom)
            for b in range(bars):
                bar_rect = patches.Rectangle(
                    (-1, y_cursor), 2, 0.3,
                    facecolor=bar_color, edgecolor="black", linewidth=0.5
                )
                ax.add_patch(bar_rect)
                y_cursor += 0.45

            # Draw dots (top of bars)
            if dots > 0:
                y_cursor += 0.1
                dot_spacing = 0.5
                x_start = -(dots - 1) * dot_spacing / 2
                for d in range(dots):
                    dot_circle = patches.Circle(
                        (x_start + d * dot_spacing, y_cursor + 0.15),
                        0.15, facecolor=dot_color, edgecolor="black", linewidth=0.5
                    )
                    ax.add_patch(dot_circle)

        # Draw level separator (except for top level)
        if level_idx < len(digits) - 1:
            y_sep = y_base + 2.7
            ax.plot([-1.2, 1.2], [y_sep, y_sep],
                    color="gray", linewidth=0.5, linestyle="--")

    return ax


def plot_maya_grid(numbers: list[int], cols: int = 5, n_levels: int | None = None, **kwargs):
    """Plot multiple Maya numbers in a grid layout.

    Parameters
    ----------
    numbers : list[int]
        Numbers to render.
    cols : int
        Number of columns in the grid.
    n_levels : int or None
        Vigesimal levels (shared across all numbers).
    **kwargs
        Additional keyword arguments for the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with all rendered glyphs.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install maya-encoding[viz]")

    rows = (len(numbers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3), **kwargs)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, n in enumerate(numbers):
        r, c = divmod(idx, cols)
        plot_maya_number(n, n_levels=n_levels, ax=axes[r][c])

    # Hide unused axes
    for idx in range(len(numbers), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    return fig
