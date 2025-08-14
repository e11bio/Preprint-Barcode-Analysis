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
    epitope_proportions = epitope_counts / total_cells  # Keep as proportions (0-1)
    mean_proportion = np.mean(epitope_proportions)
    median_proportion = np.median(epitope_proportions)

    # Create DataFrame with simplified epitope names
    simplified_epitopes = [name.split("-")[0] for name in target_channels]
    epitope_df = pd.DataFrame(
        {"Epitope": simplified_epitopes, "Proportion": epitope_proportions}
    )

    # Sort from lowest to highest proportion for better visualization
    epitope_df = epitope_df.sort_values("Proportion", ascending=True)

    # Create the plot (apply_style() already called in main, so just create figure)
    fig, ax = plt.subplots(figsize=settings["histogram_epitope_dist"], dpi=DPI)

    # Create barplot
    bars = sns.barplot(
        x="Epitope",
        y="Proportion",
        data=epitope_df,
        color=settings["main_color"],
        fill=True,
        ax=ax,
    )

    # Configure plot appearance
    ax.grid(False)
    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    # Set axis labels
    ax.set_xlabel(
        "Protein bit",
        fontsize=settings["label_size"],
        fontfamily=settings["font_family"],
        # labelpad=4,  # Reduced padding
    )
    ax.set_ylabel(
        "Proportion of somas",
        fontsize=settings["label_size"],
        fontfamily=settings["font_family"],
    )

    sns.despine()
    # Rotate x-axis labels to 90 degrees
    plt.xticks(rotation=90, ha="center")  # Font size handled by rcParams

    # Add mean line
    plt.axhline(
        y=mean_proportion,
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
                f"{p.get_height():.2f}",
                (x_pos, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=settings["tick_size"],  # Use tick size for annotations
            )

    # Set y-axis limits
    plt.ylim(0, 0.5)

    # Adjust layout with specific padding to preserve font sizes
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.4)

    return fig, epitope_df, mean_proportion, median_proportion


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
    fig, epitope_df, mean_proportion, median_proportion = create_epitope_plot(
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
- **Highest epitope presence**: {epitope_df["Proportion"].max():.3f}
- **Lowest epitope presence**: {epitope_df["Proportion"].min():.3f}
- **Mean epitope presence**: {mean_proportion:.3f}
- **Median epitope presence**: {median_proportion:.3f}
- **Standard deviation of epitope presence**: {np.std(epitope_df["Proportion"]):.3f}

The histogram showcases the distribution of each epitope of the barcode, indicating the proportion of cells containing each epitope.

The values for each epitope are as follows:
{epitope_df.to_markdown()}

![Epitope Distribution](epitope_distribution.png)
"""

    md_filename = os.path.join(output_dir, "epitope_distribution.md")
    with open(md_filename, "w") as md_file:
        md_file.write(md_content)

    print(f"Plot saved to {plot_filename}")
    print(f"Markdown documentation saved to {md_filename}")
