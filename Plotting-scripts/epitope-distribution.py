# this script generates a plot for epitope distribution
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
from soma_preprocessing import generate_barcode_array, target_channels


def create_epitope_plot(soma_barcodes, settings):
    """Create epitope distribution plot"""
    # Calculate epitope statistics
    epitope_counts = np.sum(soma_barcodes, axis=0)
    total_cells = len(soma_barcodes)
    epitope_percentages = (epitope_counts / total_cells) * 100
    mean_percentage = np.mean(epitope_percentages)
    median_percentage = np.median(epitope_percentages)

    # Create DataFrame with simplified epitope names
    simplified_epitopes = [name.split("-")[0] for name in target_channels]
    epitope_df = pd.DataFrame(
        {"Epitope": simplified_epitopes, "Percentage": epitope_percentages}
    )

    # Sort from lowest to highest percentage for better visualization
    epitope_df = epitope_df.sort_values("Percentage", ascending=True)

    # Create the plot
    plt.figure(figsize=settings["histogram_epitope_dist"], dpi=DPI)
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.size": settings["tick_size"],
            "axes.labelsize": settings["label_size"],
            "axes.titlesize": settings["title_size"],
            "xtick.labelsize": settings["tick_size"],
            "ytick.labelsize": settings["tick_size"],
            "legend.fontsize": settings["legend_size"],
        }
    )

    # Create barplot
    bars = sns.barplot(
        x="Epitope",
        y="Percentage",
        data=epitope_df,
        color=settings["main_color"],
        fill=True,
    )

    # Configure plot appearance
    ax = plt.gca()
    ax.grid(False)
    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    sns.despine()
    plt.xticks(rotation=90, ha="center")  # Font size handled by rcParams

    # Add mean line
    plt.axhline(
        y=mean_percentage,
        color="grey",
        linestyle="--",
        alpha=0.5,
        label="Average expression",
    )

    # Add legend with correct font size
    plt.legend(
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0, 1),
        fontsize=settings["legend_size"],
    )

    # Add percentage labels for lowest and highest bars
    for i, p in enumerate(bars.patches):
        if i == 0 or i == len(bars.patches) - 1:
            x_pos = p.get_x() + p.get_width() / 2.0
            if i == 0:  # Highest bar - shift label right
                x_pos += 0.8

            bars.annotate(
                f"{p.get_height():.1f}%",
                (x_pos, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=settings["tick_size"],  # Use tick size for annotations
            )

    # Set y-axis limits and adjust layout
    plt.ylim(0, 50)
    plt.tight_layout()

    return bars.figure, epitope_df, mean_percentage, median_percentage


if __name__ == "__main__":
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

    # Create output directory
    output_dir = get_script_output_dir("epitope-distribution")

    # Generate data and create plot
    soma_barcodes = generate_barcode_array()
    fig, epitope_df, mean_percentage, median_percentage = create_epitope_plot(
        soma_barcodes, settings
    )

    # Save plot with style-specific filename
    plot_filename = get_output_filename(
        "epitope_distribution",
        settings["style"],
        "png",
        script_name="epitope-distribution",
    )
    fig.savefig(plot_filename, dpi=500)
    plt.close(fig)

    # Create markdown documentation
    md_content = f"""# Epitope Distribution Analysis

## Overview
This plot shows the distribution of epitopes across {len(soma_barcodes)} somas in the barcode analysis, sorted in ascending order.

## Details
- **Total somas analyzed**: {len(soma_barcodes)}
- **Highest epitope presence**: {epitope_df["Percentage"].max():.1f}%
- **Lowest epitope presence**: {epitope_df["Percentage"].min():.1f}%
- **Mean epitope presence**: {mean_percentage:.1f}%
- **Median epitope presence**: {median_percentage:.1f}%
- **Standard deviation of epitope presence**: {np.std(epitope_df["Percentage"]):.1f}%

The histogram showcases the distribution of each epitope of the barcode, indicating the percentage of cells containing each epitope.

The values for each epitope are as follows:
{epitope_df.to_markdown()}

![Epitope Distribution](epitope_distribution.png)
"""

    md_filename = os.path.join(output_dir, "epitope_distribution.md")
    with open(md_filename, "w") as md_file:
        md_file.write(md_content)

    print(f"Plot saved to {plot_filename}")
    print(f"Markdown documentation saved to {md_filename}")
