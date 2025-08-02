import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import pandas as pd

# Output directory, change depending on user, I used my absolute path.
OUTPUT_DIR = "./out"

# Figure dimensions
FIG_WIDTH = 3  # inches = 152 mm
FIG_HEIGHT = 2  # inches = 102 mm
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)

# for the two histograms

FIG_SIZE_HISTOGRAM_barcode_lengths = (1.8, 1.8)
FIG_SIZE_HISTOGRAM_epitope_dist = (2.5, 1.8)
FIG_SIZE_HISTOGRAM_collisions = (1.1, 1.2)
FIG_SIZE_HEATMAP = (3.75, 2.2)

# Colors
# MAIN_COLOR = "#1f77b4"  # Blue
MAIN_COLOR = "silver"  # Silver
SECONDARY_COLOR = "#ff7f0e"  # Orange
COLOR_PALETTE = sns.color_palette(
    "tab10"
)  # Default colorful palette if more colors needed

# Font settings
FONT_FAMILY = "Arial"
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = [FONT_FAMILY, "sans-serif"]

# Font sizes
TITLE_SIZE = 8
LABEL_SIZE = 8
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

# Line settings
LINE_WIDTH = 1.5
GRID_LINE_WIDTH = 0.5
GRID_ALPHA = 0.3

# DPI for saved figures
DPI = 500


# Set the default style
def set_style():
    # make the style with grayscale
    """Apply the standard style to the current plot."""
    # Set the seaborn style
    sns.set_style(
        "whitegrid",
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": GRID_LINE_WIDTH,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "context": "paper",
        },
    )

    # Set matplotlib rcParams
    plt.rcParams.update(
        {
            "figure.figsize": FIG_SIZE,
            "figure.dpi": 100,  # Display DPI
            "savefig.dpi": DPI,
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "black",
        }
    )


# Apply style at module import
set_style()
# sns.color_palette(palette="Gre?ys")


def apply_fig_settings(fig, fig_size):
    fig.set_size_inches(fig_size)
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    sns.despine()
    # remove grid lines
    ax = fig.gca()
    ax.grid(False)
    # convert grid outlines to white
    for patch in ax.patches:
        patch.set_edgecolor("white")
    return fig
