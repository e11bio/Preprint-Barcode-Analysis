# this script generates a plot of collisions in ground truthed somas
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from barcode_simulations import true_barcodes_from_array
from plot_settings import (
    DPI,
    PlotStyle,
    apply_style,
    get_output_filename,
    get_script_output_dir,
    set_plot_style,
)
from scipy.spatial.distance import pdist, squareform
from soma_preprocessing import generate_barcode_array

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
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def configure_plot(
    fig, settings, title=None, plot_type="collisions", xlabel=None, ylabel=None
):
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

    # Add axis labels if specified
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=settings["label_size"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=settings["label_size"])

    # Apply seaborn despining for clean look
    sns.despine(fig=fig)

    # Set size based on plot type
    if plot_type == "collision_categories":
        fig.set_size_inches(*settings["histogram_collision_categories"])
    elif plot_type in ["min_distances", "average_distances", "all_distances"]:
        fig.set_size_inches(*settings["histogram_hamming_distance"])
    else:  # Default collision histogram
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
    x_vals, y_vals = zip(*collision_data)  # noqa: B905

    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    bars = ax.bar(x_vals, y_vals, color=settings["main_color"])

    # Add white edges to bars
    for bar in bars:
        bar.set_edgecolor("white")

    # Add value annotations
    for x, y in zip(x_vals, y_vals):  # noqa: B905
        ax.text(
            x,
            y,
            str(y),
            ha="center",
            va="bottom",
            fontsize=settings["tick_size"],
            color="black",
        )

    ax.set_xticks(x_vals)
    ax.set_xlabel("# of cells\nsharing barcode")
    ax.set_ylabel("Count of barcodes")

    return fig


def create_collision_category_plot(collision_counts, settings):
    """Create bar plot showing proportion of cells with/without collisions"""
    # Count cells in each category
    # For no_collision: count barcodes that appear once (count=1) * 1 cell each
    # For with_collision: sum of (count * number of barcodes with that count) for counts > 1
    no_collision = sum(1 for count in collision_counts.values() if count == 1)
    with_collision = sum(
        count * num_barcodes
        for count, num_barcodes in Counter(collision_counts.values()).items()
        if count > 1
    )

    # Calculate total cells and proportions
    total_cells = no_collision + with_collision
    no_collision_prop = no_collision / total_cells
    with_collision_prop = with_collision / total_cells

    # Create bar plot
    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    categories = ["No\ncollision", "Collision"]  # Add line breaks to labels
    proportions = [no_collision_prop, with_collision_prop]

    ax.bar(categories, proportions, color=settings["main_color"])
    ax.set_ylabel("Proportion of somas")

    # Remove x-axis label since categories are self-explanatory
    ax.set_xlabel("")

    return fig


def create_distance_plots(distance_matrix, min_distances, settings, soma_barcodes):
    """Create all distance-related plots"""
    plots = {}

    # Minimum distance plot
    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])

    # Separate cells by number of bits
    bit_counts = np.sum(soma_barcodes, axis=1)
    single_bit_mask = bit_counts == 1
    multi_bit_mask = bit_counts > 1

    # Calculate bin edges
    min_dist = np.floor(np.min(min_distances))
    max_dist = np.ceil(np.max(min_distances))
    bins = np.arange(min_dist, max_dist + 2) - 0.5

    # Create stacked histogram
    n, bins, patches = ax.hist(
        [min_distances[multi_bit_mask], min_distances[single_bit_mask]],
        bins=bins,
        stacked=True,
        align="mid",
        label=[">1 bit", "1 bit"],
        color=[settings["main_color"], settings["main_color"]],
    )

    # Apply hatching to distinguish categories
    for patch_set in patches:
        for patch in patch_set:
            patch.set_edgecolor("white")  # White edges for clean look

    # Apply hatching to multi-bit patches (bottom layer)
    for patch in patches[1]:
        patch.set_hatch("//////")  # Denser diagonal lines for single bit

    ax.set_xticks(np.arange(int(min_dist), int(max_dist) + 1))
    ax.legend(frameon=False, labelcolor="black")

    # Font sizes handled by rcParams
    plots["min_distances"] = fig

    # Average distance plot
    distance_matrix_avg = distance_matrix.copy()
    np.fill_diagonal(distance_matrix_avg, np.nan)
    average_distances = np.nanmean(distance_matrix_avg, axis=1)

    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])

    # Calculate appropriate bins
    min_dist = np.floor(np.min(average_distances))
    max_dist = np.ceil(np.max(average_distances))
    n_bins = int(max_dist - min_dist + 1)  # One bin per integer distance
    bins = np.linspace(min_dist, max_dist, n_bins + 1)

    ax.hist(average_distances, bins=bins, color=settings["main_color"])
    ax.set_xlim(min_dist, max_dist)
    ax.set_xticks(np.arange(int(min_dist), int(max_dist) + 1))

    # Font sizes handled by rcParams
    plots["average_distances"] = fig

    # All distances plot
    fig, ax = plt.subplots(figsize=settings["histogram_collisions"])
    # Convert to proportions
    weights = np.ones_like(distance_matrix_avg.flatten()) / len(
        distance_matrix_avg.flatten()
    )
    ax.hist(
        distance_matrix_avg.flatten(),
        bins=17,
        weights=weights,
        color=settings["main_color"],
    )
    ax.set_xlim(0, 18)
    ax.set_ylabel("Proportion")
    # Font sizes handled by rcParams
    plots["all_distances"] = fig

    return plots


def save_plots_and_docs(
    plots, soma_barcodes, collision_counts, distance_matrix, output_dir, settings
):
    """Save all plots and create documentation"""
    plot_configs = {
        "collisions": {
            "plot": plots["collisions"],
            "title": None,
            "xlabel": "# of cells\nsharing barcode",
            "ylabel": "Count of barcodes",
        },
        "collision_categories": {
            "plot": plots["collision_categories"],
            "title": None,  # No title for this plot
            "xlabel": None,  # Categories are self-explanatory
            "ylabel": "Fraction of somas",
        },
        "min_distances": {
            "plot": plots["min_distances"],
            "title": None,
            "xlabel": "Hamming distance\nto nearest neighbor",
            "ylabel": "Count of somas",
        },
        "average_distances": {
            "plot": plots["average_distances"],
            "title": None,
            "xlabel": "Mean Hamming distance\nto other cells",
            "ylabel": "Count of somas",
        },
        "all_distances": {
            "plot": plots["all_distances"],
            "title": None,
            "xlabel": "Hamming distance",
            "ylabel": "Fraction of pairwise\ndistances",
        },
    }

    saved_files = []
    for name, config in plot_configs.items():
        configure_plot(
            config["plot"],
            settings,
            title=config["title"],
            plot_type=name,
            xlabel=config["xlabel"],
            ylabel=config["ylabel"],
        )
        filename = get_output_filename(
            name, settings["style"], "pdf", script_name="collisions"
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

    # Calculate proportions of colliding cells by bit count
    colliding_mask = np.zeros(len(soma_barcodes), dtype=bool)
    bit_counts = np.sum(soma_barcodes, axis=1)  # Count bits for each cell

    # Find cells that share barcodes
    for barcode_set, count in collision_counts.items():
        if count > 1:  # Only look at barcodes that appear multiple times
            # Convert frozenset of channel names to binary array
            barcode_array = np.zeros(len(CHANNEL_NAMES), dtype=int)
            for i, channel in enumerate(CHANNEL_NAMES):
                if channel in barcode_set:
                    barcode_array[i] = 1

            # Find all cells with this barcode pattern
            matching_cells = np.all(soma_barcodes == barcode_array, axis=1)
            colliding_mask[matching_cells] = True

    # Count colliding cells by their bit count
    single_bit_collisions = np.sum((bit_counts == 1) & colliding_mask)
    multi_bit_collisions = np.sum((bit_counts > 1) & colliding_mask)
    total_colliding = np.sum(colliding_mask)

    print(f"Single-bit collisions: {single_bit_collisions}")
    print(f"Multi-bit collisions: {multi_bit_collisions}")
    print(f"Total colliding cells: {total_colliding}")
    print(
        f"Verification - total cells with shared barcodes: {sum(count for count in collision_counts.values() if count > 1)}"
    )

    md_content = f"""# Collision Analysis Report

## Overview
Analysis of barcode collisions and distance metrics for ground truth somas.

## Results Summary
- **Total somas analyzed**: {total_somas}
- **Unique barcodes**: {unique_barcodes}
- **Collision rate**: {(total_somas - unique_barcodes) / total_somas * 100:.1f}%

## Collision Analysis by Bit Count
- **Single-bit cells in collisions**: {single_bit_collisions} ({single_bit_collisions / total_colliding * 100:.1f}% of colliding cells)
- **Multi-bit cells in collisions**: {multi_bit_collisions} ({multi_bit_collisions / total_colliding * 100:.1f}% of colliding cells)

## Distance Statistics
- **Average distance**: {avg_distance:.2f}
- **Maximum distance**: {max_distance:.0f}
- **Minimum distance**: {min_distance:.0f}
- **Standard deviation**: {std_distance:.2f}

## Generated Plots
- Collision histogram: `collisions.pdf`
- Collision categories: `collision_categories.pdf`
- Minimum distances: `min_distances.pdf`  
- Average distances: `average_distances.pdf`
- All distances: `all_distances.pdf`

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
    plots["collision_categories"] = create_collision_category_plot(
        collision_counts, settings
    )
    plots.update(
        create_distance_plots(distance_matrix, min_distances, settings, soma_barcodes)
    )

    # Save everything
    saved_files = save_plots_and_docs(
        plots, soma_barcodes, collision_counts, distance_matrix, output_dir, settings
    )

    # Print summary
    print("Collision analysis complete! Files saved:")
    for filepath in saved_files:
        print(f"  - {filepath}")
