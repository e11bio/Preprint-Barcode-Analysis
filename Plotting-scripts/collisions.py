# this script generates a plot of collisions in ground truthed somas
from collections import Counter
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from plot_settings import (
    DPI,
    PlotStyle,
    apply_style,
    get_output_filename,
    get_script_output_dir,
    set_plot_style,
)

from soma_preprocessing import generate_barcode_array
from barcode_simulations import true_barcodes_from_array

# Channel names (could be imported from soma_preprocessing if available)
CHANNEL_NAMES = [
    "E2",
    "S1",
    "ALFA",
    "Ty1",
    "HA",
    "T7",
    "VSVG",
    "AU5",
    "NWS",
    "SunTag",
    "ETAG",
    "SPOT",
    "MoonTag",
    "HSV",
    "Protein-C",
    "Tag100",
    "c-Myc",
    "OLLAS",
]


def configure_plot_style(settings):
    """Configure matplotlib and seaborn styling"""
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.family": settings["font_family"],
            "font.size": settings["tick_size"],
            "axes.labelsize": settings["label_size"],
            "axes.titlesize": settings["label_size"],
            "xtick.labelsize": settings["tick_size"],
            "ytick.labelsize": settings["tick_size"],
            "legend.fontsize": settings["tick_size"],
        }
    )


def configure_plot(fig, settings, title=None):
    """Apply consistent styling to plots"""
    # Ensure figure size is set correctly
    ax = fig.gca()

    # Remove grid lines
    ax.grid(False)

    # Style patches with white edges
    for patch in ax.patches:
        patch.set_edgecolor("white")

    # Add title if specified
    if title:
        ax.set_title(title, fontsize=settings["label_size"])

    # Apply seaborn despining for clean look
    sns.despine(fig=fig)
    fig.set_size_inches(*settings["histogram_collisions"])
    plt.tight_layout()


def calculate_collisions_and_distances(soma_barcodes):
    """Calculate collision counts and distance metrics in one pass"""
    # Calculate collisions
    true_barcodes = true_barcodes_from_array(soma_barcodes, CHANNEL_NAMES)
    collision_counts = Counter(true_barcodes)

    # Calculate distance matrix once
    distance_matrix = (
        squareform(pdist(soma_barcodes, metric="hamming")) * soma_barcodes.shape[1]
    )
    np.fill_diagonal(distance_matrix, np.inf)
    min_distances = np.min(distance_matrix, axis=1)

    return collision_counts, distance_matrix, min_distances


def create_collision_plot(collision_counts, settings):
    """Create collision histogram plot"""
    collision_histogram = Counter(collision_counts.values())
    collision_data = sorted(collision_histogram.items())
    x_vals, y_vals = zip(*collision_data)

    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    ax.bar(x_vals, y_vals, color=settings["main_color"])
    ax.set_xticks(x_vals)
    # Font sizes handled by rcParams
    return fig


def create_distance_plots(distance_matrix, min_distances, settings):
    """Create all distance-related plots"""
    plots = {}

    # Minimum distance plot
    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    ax.hist(
        min_distances,
        bins=np.arange(min_distances.min(), min_distances.max() + 2) - 0.5,
        align="mid",
        color=settings["main_color"],
    )
    ax.set_xticks(np.arange(min_distances.min(), min_distances.max() + 1))
    # Font sizes handled by rcParams
    plots["min_distances"] = fig

    # Average distance plot
    distance_matrix_avg = distance_matrix.copy()
    np.fill_diagonal(distance_matrix_avg, np.nan)
    average_distances = np.nanmean(distance_matrix_avg, axis=1)

    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    ax.hist(average_distances, bins=50, color=settings["main_color"])
    ax.set_xlim(0, 18)
    # Font sizes handled by rcParams
    plots["average_distances"] = fig

    # All distances plot
    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    ax.hist(distance_matrix_avg.flatten(), bins=17, color=settings["main_color"])
    ax.set_xlim(0, 18)
    # Font sizes handled by rcParams
    plots["all_distances"] = fig

    return plots


def save_plots_and_docs(
    plots, soma_barcodes, collision_counts, distance_matrix, output_dir, settings
):
    """Save all plots and create documentation"""
    plot_configs = {
        "collisions": {"plot": plots["collisions"], "title": None},
        "min_distances": {
            "plot": plots["min_distances"],
            "title": "Hamming distance to nearest neighbor",
        },
        "average_distances": {
            "plot": plots["average_distances"],
            "title": "Average distance to nearest neighbor",
        },
        "all_distances": {"plot": plots["all_distances"], "title": "All distances"},
    }

    saved_files = []
    for name, config in plot_configs.items():
        configure_plot(config["plot"], settings, config["title"])
        filename = get_output_filename(
            name, settings["style"], "png", script_name="collisions"
        )
        config["plot"].savefig(filename, dpi=DPI, bbox_inches="tight")
        saved_files.append(filename)
        plt.close(config["plot"])

        # Calculate distance statistics
    distance_matrix_stats = distance_matrix.copy()
    np.fill_diagonal(distance_matrix_stats, np.nan)  # Exclude self-distances
    all_distances = distance_matrix_stats.flatten()
    all_distances = all_distances[~np.isnan(all_distances)]  # Remove NaN values

    avg_distance = np.mean(all_distances)
    max_distance = np.max(all_distances)
    min_distance = np.min(all_distances)
    std_distance = np.std(all_distances)

    # Create markdown documentation
    unique_barcodes = len(collision_counts)
    total_somas = len(soma_barcodes)

    md_content = f"""# Collision Analysis Report

## Overview
Analysis of barcode collisions and distance metrics for ground truth somas.

## Results Summary
- **Total somas analyzed**: {total_somas}
- **Unique barcodes**: {unique_barcodes}
- **Collision rate**: {(total_somas - unique_barcodes) / total_somas * 100:.1f}%

## Distance Statistics
- **Average distance**: {avg_distance:.2f}
- **Maximum distance**: {max_distance:.0f}
- **Minimum distance**: {min_distance:.0f}
- **Standard deviation**: {std_distance:.2f}

## Generated Plots
- Collision histogram: `collisions.png`
- Minimum distances: `min_distances.png`  
- Average distances: `average_distances.png`
- All distances: `all_distances.png`

## Methodology
- **Distance Metric**: Hamming distance between binary barcode patterns
- **Collision Detection**: Identical barcode sequences across different somas
"""

    md_filename = os.path.join(output_dir, "collision_analysis.md")
    with open(md_filename, "w") as f:
        f.write(md_content)

    return saved_files + [md_filename]


if __name__ == "__main__":
    # Setup
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        choices=["paper", "poster"],
        default="paper",
        help="Plot style to use (paper or poster)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        help="Font size to use for poster mode",
    )
    args = parser.parse_args()

    # Get plot settings
    style = PlotStyle.POSTER if args.style == "poster" else PlotStyle.PAPER
    settings = set_plot_style(style, font_size=args.font_size)

    # Apply settings
    apply_style(settings)
    configure_plot_style(settings)

    output_dir = get_script_output_dir("collisions")
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)

    # Generate data
    soma_barcodes = generate_barcode_array()

    # Calculate all metrics in one pass
    collision_counts, distance_matrix, min_distances = (
        calculate_collisions_and_distances(soma_barcodes)
    )

    # Create all plots
    plots = {}
    plots["collisions"] = create_collision_plot(collision_counts, settings)
    plots.update(create_distance_plots(distance_matrix, min_distances, settings))

    # Save everything
    saved_files = save_plots_and_docs(
        plots, soma_barcodes, collision_counts, distance_matrix, output_dir, settings
    )

    # Print summary
    print("Collision analysis complete! Files saved:")
    for filepath in saved_files:
        print(f"  - {filepath}")
