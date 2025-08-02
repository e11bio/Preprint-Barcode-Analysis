# This script generates a plot for barcode distribution given a numpy array of binary barcodes.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from soma_preprocessing import generate_barcode_array
from datetime import datetime

# Import plotting settings
from plot_settings import (
    MAIN_COLOR,
    FIG_SIZE_HISTOGRAM_barcode_lengths,
    set_style,
    DPI,
    OUTPUT_DIR,
)

FIG_SIZE = FIG_SIZE_HISTOGRAM_barcode_lengths
# Import functions from soma-preprocessing.py

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()

# load the soma_barcode_info.csv file


def configure_barcode_plot():
    """Configure plot styling for barcode length analysis"""
    set_style()
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def create_barcode_length_plot(barcode_lengths):
    """Create histogram plot of barcode lengths"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_HISTOGRAM_barcode_lengths, dpi=DPI)

    # Create histogram
    sns.histplot(barcode_lengths, kde=False, bins=range(19), discrete=True, ax=ax)

    # Style the bars
    for patch in ax.patches:
        patch.set_facecolor(MAIN_COLOR)
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


def save_plot_and_docs(fig, barcode_lengths, output_dir):
    """Save plot and create documentation"""
    # Save plot
    plot_filename = os.path.join(output_dir, "barcode_length_distribution.png")
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
    configure_barcode_plot()
    output_dir = os.path.join("./out", "barcode-length")
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    soma_barcodes = generate_barcode_array()
    barcode_lengths = np.sum(soma_barcodes, axis=1)

    # Create plot
    fig = create_barcode_length_plot(barcode_lengths)

    # Save everything
    plot_file, doc_file, stats = save_plot_and_docs(fig, barcode_lengths, output_dir)

    # Print summary
    print(f"Barcode length analysis complete!")
    print(f"Total cells: {stats['total_cells']}")
    print(f"Mean barcode length: {stats['mean_length']:.2f}")
    print(f"Files saved:")
    print(f"  - {plot_file}")
    print(f"  - {doc_file}")
