from enum import Enum, auto

import matplotlib.pyplot as plt
import os
import seaborn as sns


class PlotStyle(Enum):
    PAPER = auto()
    POSTER = auto()


# Default style
CURRENT_STYLE = PlotStyle.PAPER

# Base output directory
BASE_OUTPUT_DIR = "./out"


def get_script_output_dir(script_name: str) -> str:
    """Get the output directory for a specific script.

    Args:
        script_name: Name of the script (e.g. 'collisions', 'epitope-distribution')

    Returns:
        Path to script-specific output directory
    """
    script_dir = os.path.join(BASE_OUTPUT_DIR, script_name)
    os.makedirs(script_dir, exist_ok=True)
    return script_dir


# Figure dimensions - Paper sizes (default)
PAPER_SIZES = {
    "FIG_WIDTH": 3,  # inches = 152 mm
    "FIG_HEIGHT": 2,  # inches = 102 mm
    "HISTOGRAM_BARCODE_LENGTHS": (1.8, 1.9),
    "HISTOGRAM_EPITOPE_DIST": (2.8, 1.9),
    "HISTOGRAM_COLLISIONS": (1.4, 1.8),  #
    "HISTOGRAM_HAMMING_DISTANCE": (2, 1.8),  # Wider for distance distribution
    "HISTOGRAM_COLLISION_CATEGORIES": (1.3, 1.6),  # important one
    "HEATMAP": (1.2, 4),
}

# Poster sizes (1.5x paper sizes)
POSTER_SIZES = {
    "FIG_WIDTH": 4.5,  # 1.5x paper width
    "FIG_HEIGHT": 3,  # 1.5x paper height
    "HISTOGRAM_BARCODE_LENGTHS": (
        4,
        4,
    ),  # Taller aspect ratio for better poster visibility
    "HISTOGRAM_EPITOPE_DIST": (
        6,
        4,
    ),  # Taller aspect ratio for better poster visibility
    "HISTOGRAM_COLLISIONS": (2, 4),  # Taller aspect ratio for better poster visibility
    "HISTOGRAM_HAMMING_DISTANCE": (6, 4),  # Wider for distance distribution
    "HISTOGRAM_COLLISION_CATEGORIES": (3, 4),  # Narrower for binary categories
    "HEATMAP": (1, 3.3),
}

# Colors
MAIN_COLOR = "gray"  # Silver
SECONDARY_COLOR = "#ff7f0e"  # Orange
COLOR_PALETTE = sns.color_palette("tab10")

# Font settings
FONT_FAMILY = "Arial"
FONT_FALLBACK = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# Font sizes for paper and poster
PAPER_FONT_SIZES = {"TITLE": 8, "LABEL": 8, "TICK": 8, "LEGEND": 8, "ANNOTATION": 8}
POSTER_FONT_SIZES = (
    PAPER_FONT_SIZES.copy()
)  # Will be modified by set_plot_style if needed

# Line settings
LINE_WIDTH = 1.5
GRID_LINE_WIDTH = 0.5
GRID_ALPHA = 0.3

# DPI for saved figures
DPI = 500


# Function to set the plot style (paper or poster)
def set_plot_style(style: PlotStyle = PlotStyle.PAPER, font_size: int = None):
    """Set the plot style to either paper or poster mode.

    Args:
        style (PlotStyle): Either PlotStyle.PAPER or PlotStyle.POSTER
        font_size (int, optional): If provided and style is POSTER, use this font size for all text elements

    Returns:
        dict: Dictionary containing all plot settings
    """
    sizes = PAPER_SIZES if style == PlotStyle.PAPER else POSTER_SIZES

    # Set font sizes
    if style == PlotStyle.POSTER and font_size is not None:
        font_sizes = {k: font_size for k in PAPER_FONT_SIZES.keys()}
    else:
        font_sizes = PAPER_FONT_SIZES if style == PlotStyle.PAPER else POSTER_FONT_SIZES

    # Return settings dictionary
    return {
        "style": style,
        "fig_width": sizes["FIG_WIDTH"],
        "fig_height": sizes["FIG_HEIGHT"],
        "fig_size": (sizes["FIG_WIDTH"], sizes["FIG_HEIGHT"]),
        "histogram_barcode_lengths": sizes["HISTOGRAM_BARCODE_LENGTHS"],
        "histogram_epitope_dist": sizes["HISTOGRAM_EPITOPE_DIST"],
        "histogram_collisions": sizes["HISTOGRAM_COLLISIONS"],
        "histogram_hamming_distance": sizes["HISTOGRAM_HAMMING_DISTANCE"],
        "histogram_collision_categories": sizes["HISTOGRAM_COLLISION_CATEGORIES"],
        "heatmap": sizes["HEATMAP"],
        "title_size": font_sizes["TITLE"],
        "label_size": font_sizes["LABEL"],
        "tick_size": font_sizes["TICK"],
        "legend_size": font_sizes["LEGEND"],
        "annotation_size": font_sizes["ANNOTATION"],
        "font_family": FONT_FAMILY,
        "main_color": MAIN_COLOR,
        "secondary_color": SECONDARY_COLOR,
        "color_palette": COLOR_PALETTE,
    }


def apply_style(settings):
    """Apply plot style settings.

    Args:
        settings (dict): Dictionary of plot settings from set_plot_style()
    """
    # Set seaborn style
    sns.set_style(
        "whitegrid",
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": GRID_LINE_WIDTH,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
        },
    )

    # Set matplotlib rcParams
    plt.rcParams.update(
        {
            "figure.figsize": settings["fig_size"],
            "figure.dpi": 100,  # Display DPI
            "savefig.dpi": DPI,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FALLBACK,
            "font.size": settings["tick_size"],
            "axes.titlesize": settings["title_size"],
            "axes.labelsize": settings["label_size"],
            "xtick.labelsize": settings["tick_size"],
            "ytick.labelsize": settings["tick_size"],
            "legend.fontsize": settings["legend_size"],
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "black",
        }
    )


def apply_fig_settings(fig, settings):
    """Apply figure settings and return the modified figure.

    Args:
        fig: matplotlib figure object
        settings: dict of plot settings from set_plot_style()

    Returns:
        fig: modified matplotlib figure object
    """
    fig.set_size_inches(settings["fig_size"])
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


def get_output_filename(
    base_name: str, style: PlotStyle, extension: str = "png", script_name: str = None
) -> str:
    """Generate an output filename that includes the current style.

    Args:
        base_name: The base name of the file without extension
        style: The plot style (PlotStyle.PAPER or PlotStyle.POSTER)
        extension: The file extension (default: 'png')
        script_name: Name of the script for output directory organization (e.g. 'collisions')

    Returns:
        str: The complete filename including style and extension
    """
    style_suffix = "_poster" if style == PlotStyle.POSTER else "_paper"
    output_dir = get_script_output_dir(script_name) if script_name else BASE_OUTPUT_DIR
    return os.path.join(output_dir, f"{base_name}{style_suffix}.{extension}")
