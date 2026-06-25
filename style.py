"""
Theme for matplotlib
"""

import matplotlib.pyplot as plt

COLORS = {
    "background": "#080808",
    "panel": "#111111",
    "grid": "#222222",
    "text": "#e8e8e8",
    "text_dim": "#898989",
    "green": "#00e676",
    "blue": "#00b0ff",
    "orange": "#ff9100",
    "yellow": "#ffd600",
    "pink": "#ff4081",
    "red": "#ff5252",
    "purple": "#b388ff",
    "white": "#ffffff",
}


def apply_theme() -> None:
    """Apply the palette to all matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "text.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": COLORS["grid"],
        "legend.labelcolor": COLORS["text"],
        "font.family": "monospace",
        "axes.prop_cycle": plt.cycler(color=[
            COLORS["blue"], COLORS["orange"], COLORS["yellow"],
            COLORS["pink"], COLORS["purple"], COLORS["green"],
        ]),
    })
