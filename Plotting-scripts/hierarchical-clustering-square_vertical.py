# this script generates a hierarchical clustering plot of the soma barcodes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os

# from scipy.stats import entropy
from scipy.spatial.distance import pdist  # , squareform
from scipy.cluster.hierarchy import linkage, leaves_list

from plot_settings import (
    FIG_SIZE_HEATMAP,
    set_style,
    DPI,
    OUTPUT_DIR,
    LABEL_SIZE,
    TICK_SIZE,
    FONT_FAMILY,
)

FIG_SIZE_HEATMAP = (2.3, 6)

from soma_preprocessing import (
    generate_barcode_array,
    # target_channels
)

# Create output directory if it doesn't exist
OUTPUT_DIR = OUTPUT_DIR

# generate barcodes array
soma_barcodes = generate_barcode_array()


# hierarchical clustering approach


# Function to create the hierarchical clustered heatmap
def create_hierarchical_clustering(soma_barcodes, method=None):
    """Create a clustering of soma barcodes with hierarchical clustering using Hamming distance."""
    n_somas, n_channels = soma_barcodes.shape
    print(
        f"Creating hierarchical clustered heatmap for {n_somas} somas with {n_channels} channels"
    )
    print(f"Using {method} linkage method")

    # Calculate Hamming distances between all pairs of somas
    hamming_distances = pdist(soma_barcodes, metric="hamming")

    # Create the linkage matrix
    Z = linkage(hamming_distances, method=method)

    # Get the order of samples from the hierarchical clustering
    cluster_order = leaves_list(Z)

    # Reorder the soma barcodes according to the clustering
    clustered_barcodes = soma_barcodes[cluster_order]

    return cluster_order, clustered_barcodes


def plot_hierarchical_clustering(
    clustered_barcodes,
    output_path=None,
    orientation="vertical",
    reverse_channels=False,
):
    """Plot the hierarchical clustering of soma barcodes."""
    n_somas, n_channels = clustered_barcodes.shape
    set_style()
    # Create figure for the heatmap
    fig = plt.figure(figsize=FIG_SIZE_HEATMAP, dpi=DPI)
    ax_heatmap = plt.gca()

    channel_names = [
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
    # reverse the order of the channels
    if reverse_channels:
        clustered_barcodes = clustered_barcodes[:, ::-1]
        channel_names = channel_names[::-1]
    # Transpose the data and plot the heatmap with inverted colors (binary_r) and no grid lines

    ax = sns.heatmap(
        # clustered_barcodes.T,
        clustered_barcodes,
        ax=ax_heatmap,
        cmap="binary_r",
        xticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0,
        linecolor="white",
    )  # Removed grid by setting linewidths=0
    # else:
    #     ax = sns.heatmap(
    #         clustered_barcodes,
    #         ax=ax_heatmap,
    #         cmap="binary_r",
    #         xticklabels=False,
    #         cbar=False,
    #         vmin=0,
    #         vmax=1,
    #         linewidths=0,
    #         linecolor="white",
    #     )

    # Set y-axis labels for the target channels with simplified names
    # if orientation == "horizontal":
    #     ax_heatmap.set_yticks(np.arange(n_channels) + 0.5)
    #     ax_heatmap.set_yticklabels(
    #         channel_names, rotation=0, fontsize=TICK_SIZE, fontfamily=FONT_FAMILY
    #     )
    #     ax_heatmap.set_ylabel("Channels", fontsize=LABEL_SIZE)
    #     ax_heatmap.set_xlabel(f"Somas (n={n_somas})", fontsize=LABEL_SIZE)
    #     # Add light gray lines between channels
    #     for i in range(1, n_channels):
    #         ax_heatmap.axhline(
    #             y=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.5
    #         )
    # else:
    ax_heatmap.set_xticks(np.arange(n_channels) + 0.5)
    ax_heatmap.set_xticklabels(
        channel_names, rotation=90, fontsize=TICK_SIZE, fontfamily=FONT_FAMILY
    )
    ax_heatmap.set_xlabel(f"Channels (n={n_channels})", fontsize=LABEL_SIZE)
    ax_heatmap.set_ylabel(f"Somas (n={n_somas})", fontsize=LABEL_SIZE)
    # clear the y axis
    ax_heatmap.set_yticks([])
    # Add light gray lines between channels
    for i in range(1, n_channels):
        ax_heatmap.axvline(
            x=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.5
        )

    ax.grid(False)

    # Set font properties for all text elements
    for text in ax_heatmap.get_xticklabels() + ax_heatmap.get_yticklabels():
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(TICK_SIZE)

    ax_heatmap.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax_heatmap.yaxis.label.set_fontfamily(FONT_FAMILY)

    # plt.title('Soma Barcodes with Hierarchical Clustering', fontsize=16)
    plt.tight_layout()

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    return fig, ax_heatmap


def create_markdown_description(output_path, method, n_somas, n_channels):
    """Create a markdown file describing the hierarchical clustering analysis"""
    markdown_content = f"""# Hierarchical Clustering Analysis of Soma Barcodes

## Overview
This analysis presents a hierarchical clustering of {n_somas} soma barcodes across {n_channels} epitope channels.

## Methodology
- **Data**: Binary barcode matrix where each row represents a soma and each column represents a target channel
- **Distance Metric**: Hamming distance between barcode patterns
- **Linkage Method**: {method} linkage
- **Visualization**: Binary heatmap where black indicates presence (1) and white indicates absence (0)

## Interpretation
The hierarchical clustering organizes somas with similar barcode patterns together, revealing potential groups or clusters of somas that share similar epitope expression patterns. Clusters of somas with similar patterns may indicate:

1. Somas with shared lineage
2. Somas with similar functional properties
3. Potential technical artifacts or batch effects

## Channels
The 18 epitope channels displayed from left to right are:
- E2, S1, ALFA, Ty1, HA, T7, VSVG, AU5, NWS, SunTag, ETAG, SPOT, MoonTag, HSV, ProteinC, Tag100, CMyc, OLLAS

![Hierarchical Clustering Heatmap]({os.path.basename(output_path)})
"""

    with open(output_path.replace(".png", ".md"), "w") as f:
        f.write(markdown_content)

    print(f"Markdown description saved to {output_path.replace('.png', '.md')}")


def add_highlighted_soma(ax, clustered_barcodes, soma_seg_ids, soma_id):
    """Add a highlighted soma row to the heatmap."""
    # get the index of the soma_id
    soma_index = soma_seg_ids.index(soma_id)
    # Get x-axis limits for the rectangle width
    xmin, xmax = ax.get_xlim()

    # add a red rectangle to highlight the row
    rect = patches.Rectangle(
        (xmin, soma_index),  # Start at left edge, at the row position
        xmax - xmin,  # Width spans the full heatmap width
        1,  # Height of one row
        fill=False,
        edgecolor="red",
        linewidth=0.5,
    )
    ax.add_patch(rect)


# ward': 'Minimizes variance within clusters; tends to create compact, equal-sized clusters',
# complete': 'Uses maximum distance between elements; sensitive to outliers, creates more balanced clusters',
# average': 'Uses average distance between all pairs; moderately robust to outliers, creates natural clusters',
# single': 'Uses minimum distance between elements; can create elongated clusters due to chaining effect'


if __name__ == "__main__":
    plotting = True
    # clustering =

    # Create the hierarchical clustered heatmap with x linkage
    orientation = "vertical"
    output_filename = f"hierarchical_clustering_{orientation}_square_250602.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    method = "ward"

    cluster_order, clustered_barcodes = create_hierarchical_clustering(
        soma_barcodes, method=method
    )
    # load the somas found in soma_barcode_info.csv
    soma_seg_ids = pd.read_csv("./soma_barcode_info.csv")["segment_id"].tolist()
    print(soma_seg_ids)
    # sort the soma_seg_ids by the cluster_order
    soma_seg_ids = [soma_seg_ids[i] for i in cluster_order]
    # print the index where the value is 1492

    fig, ax = plot_hierarchical_clustering(
        clustered_barcodes,
        output_path=output_path,
        orientation=orientation,
        reverse_channels=True,
    )

    # Get shape information for the markdown file
    n_somas, n_channels = soma_barcodes.shape

    x_width = 1  # category spacing is always 1 unit apart
    xmin, xmax = ax.get_xlim()  # Get x-axis limits instead of y-axis limits

    highlights = [1219, 1488]
    for highlight in highlights:
        highlight_index = soma_seg_ids.index(highlight)

        rect = patches.Rectangle(
            (xmin, highlight_index),  # Start at left edge, at the row position
            xmax - xmin,  # Width spans the full heatmap width
            1,  # Height of one row
            edgecolor="red",
            linewidth=0.5,
            fill=False,
        )
        ax.add_patch(rect)

        ax.text(
            xmax + (xmax - xmin) * 0.02,  # Position text to the right of the heatmap
            highlight_index + 0.5,  # Center vertically in the row
            f"Seg.{soma_seg_ids[highlight_index]}",
            ha="left",
            va="center",
            fontsize=8,
            rotation=0,  # No rotation needed for horizontal text
        )

    # save the figure
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Figure saved to {output_path}")

    # Get shape information for the markdown file
    # n_somas, n_channels = soma_barcodes.shape

    # Create the markdown description
    create_markdown_description(output_path, method, n_somas, n_channels)

# Show the plot (optional - can be commented out for automated runs)
# plt.show()
