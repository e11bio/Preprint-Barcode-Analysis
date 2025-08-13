# This script generates a plot for barcode distribution given a numpy array of binary barcodes.

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from soma_preprocessing import generate_barcode_array

# Import plotting settings
from plot_settings import (
    DPI,
    PlotStyle,
    apply_style,
    get_output_filename,
    get_script_output_dir,
    set_plot_style,
)
# Import functions from soma-preprocessing.py

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()

# load the soma_barcode_info.csv file


def configure_barcode_plot(settings):
    """Configure plot styling for barcode length analysis"""
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.size": settings["tick_size"],
            "axes.labelsize": settings["label_size"],
            "axes.titlesize": settings["title_size"],
            "xtick.labelsize": settings["tick_size"],
            "ytick.labelsize": settings["tick_size"],
        }
    )


def create_barcode_length_plot(barcode_lengths, settings):
    """Create histogram plot of barcode lengths"""
    fig, ax = plt.subplots(figsize=settings["histogram_barcode_lengths"], dpi=DPI)

    # Create histogram
    sns.histplot(barcode_lengths, kde=False, bins=range(19), discrete=True, ax=ax)

    # Style the bars
    for patch in ax.patches:
        patch.set_facecolor(settings["main_color"])
        patch.set_linewidth(0.1)

    # Configure plot appearance
    ax.grid(False)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    # Set x-axis ticks
    plt.xticks(range(1, 19, 2))

    # Apply seaborn despining
    sns.despine()
    plt.tight_layout()

    return fig


def calculate_statistics(barcode_lengths):
    """Calculate summary statistics for barcode lengths"""
    return {
        "total_cells": len(barcode_lengths),
        "mean_length": np.mean(barcode_lengths),
        "median_length": np.median(barcode_lengths),
        "min_length": np.min(barcode_lengths),
        "max_length": np.max(barcode_lengths),
        "std_length": np.std(barcode_lengths),
    }


def save_plot_and_docs(fig, barcode_lengths, output_dir, settings):
    """Save plot and create documentation"""
    # Save plot
    plot_filename = get_output_filename(
        "barcode_length_distribution",
        settings["style"],
        "png",
        script_name="barcode-length",
    )
    fig.savefig(plot_filename, dpi=500, bbox_inches="tight")
    plt.close(fig)

    # Calculate statistics
    stats = calculate_statistics(barcode_lengths)

    # Create markdown documentation
    md_content = f"""# Barcode Length Distribution Analysis

## Overview
Distribution of barcode lengths (Hamming weights) across {stats["total_cells"]} soma cells.

## Statistics
- **Total cells analyzed**: {stats["total_cells"]}
- **Mean barcode length**: {stats["mean_length"]:.2f}
- **Median barcode length**: {stats["median_length"]:.0f}
- **Minimum barcode length**: {stats["min_length"]}
- **Maximum barcode length**: {stats["max_length"]}
- **Standard deviation**: {stats["std_length"]:.2f}

## Methodology
The barcode length for each cell was calculated by summing the number of positive markers (1s) in each cell's barcode array. The distribution was visualized using a histogram with discrete bins for each possible barcode length (0-18).

## Generated Files
- Plot: `barcode_length_distribution.png`

![Barcode Length Distribution](barcode_length_distribution.png)
"""

    md_filename = os.path.join(output_dir, "barcode_length_distribution.md")
    with open(md_filename, "w") as f:
        f.write(md_content)

    return plot_filename, md_filename, stats


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
    configure_barcode_plot(settings)

    output_dir = get_script_output_dir("barcode-length")

    # Generate data
    soma_barcodes = generate_barcode_array()
    barcode_lengths = np.sum(soma_barcodes, axis=1)

    # Create plot
    fig = create_barcode_length_plot(barcode_lengths, settings)

    # Save everything
    plot_file, doc_file, stats = save_plot_and_docs(
        fig, barcode_lengths, output_dir, settings
    )

    # Print summary
    print("Barcode length analysis complete!")
    print(f"Total cells: {stats['total_cells']}")
    print(f"Mean barcode length: {stats['mean_length']:.2f}")
    print("Files saved:")
    print(f"  - {plot_file}")
    print(f"  - {doc_file}")
