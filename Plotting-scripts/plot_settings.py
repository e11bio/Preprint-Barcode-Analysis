import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import pandas as pd

# Figure dimensions
FIG_WIDTH = 10  # inches
FIG_HEIGHT = 6  # inches
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)

# Colors
MAIN_COLOR = "#1f77b4"  # Blue
SECONDARY_COLOR = "#ff7f0e"  # Orange
COLOR_PALETTE = sns.color_palette("tab10")  # Default colorful palette if more colors needed

# Font settings
FONT_FAMILY = "Arial"
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = [FONT_FAMILY, "sans-serif"]

# Font sizes
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
ANNOTATION_SIZE = 12

# Line settings
LINE_WIDTH = 1.5
GRID_LINE_WIDTH = 0.5
GRID_ALPHA = 0.3

# DPI for saved figures
DPI = 500

# Set the default style
def set_style():
    """Apply the standard style to the current plot."""
    # Set the seaborn style
    sns.set_style("whitegrid", {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": GRID_LINE_WIDTH,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0
    })
    
    # Set matplotlib rcParams
    plt.rcParams.update({
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
    })


# Apply style at module import
set_style() 