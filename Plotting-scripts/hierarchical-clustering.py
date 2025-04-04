# this script generates a hierarchical clustering plot of the soma barcodes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

from plot_settings import (
    MAIN_COLOR, SECONDARY_COLOR, FIG_SIZE_HEATMAP, set_style, DPI, OUTPUT_DIR,
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, ANNOTATION_SIZE, FONT_FAMILY
)

# FIG_SIZE = (2.2, 3.5)

from soma_preprocessing import (
    generate_barcode_array,
    target_channels
)

# Create output directory if it doesn't exist
OUTPUT_DIR = OUTPUT_DIR

# generate barcodes array
soma_barcodes = generate_barcode_array()


# hierarchical clustering approach

# Function to create the hierarchical clustered heatmap
def create_hierarchical_clustered_heatmap(soma_barcodes, method=None, output_path=None):
    """Create a heatmap of soma barcodes with hierarchical clustering using Hamming distance."""
    n_somas, n_channels = soma_barcodes.shape
    print(f"Creating hierarchical clustered heatmap for {n_somas} somas with {n_channels} channels")
    print(f"Using {method} linkage method")
    
    # Calculate Hamming distances between all pairs of somas
    hamming_distances = pdist(soma_barcodes, metric='hamming')
    
    # Create the linkage matrix
    Z = linkage(hamming_distances, method=method)
    
    # Get the order of samples from the hierarchical clustering
    cluster_order = leaves_list(Z)
    
    # Reorder the soma barcodes according to the clustering
    clustered_barcodes = soma_barcodes[cluster_order]
    
    set_style()
    # Create figure for the heatmap
    fig = plt.figure(figsize=FIG_SIZE_HEATMAP, dpi=DPI)
    ax_heatmap = plt.gca()
 
    # Plot the heatmap with inverted colors (binary_r) and no grid lines
    ax = sns.heatmap(clustered_barcodes, ax=ax_heatmap, cmap='binary_r',
                yticklabels=False, cbar=False, vmin=0, vmax=1,
                linewidths=0, linecolor='white')  # Removed grid by setting linewidths=0

    # Set x-axis labels for the target channels with simplified names
    channel_names = ['E2', 'S1', 'ALFA', 'Ty1', 'HA', 'T7', 
                     'VSVG', 'AU5', 'NWS', 'SunTag', 'ETAG', 
                     'SPOT', 'MoonTag', 'HSV', 'ProteinC', 
                     'Tag100', 'CMyc', 'OLLAS']
    ax_heatmap.set_xticks(np.arange(n_channels) + 0.5)
    ax_heatmap.set_xticklabels(channel_names, rotation=90, fontsize=TICK_SIZE, fontfamily=FONT_FAMILY)
    ax_heatmap.set_xlabel('Channels', fontsize=LABEL_SIZE)

    ax_heatmap.set_ylabel('Somas (n=147)', fontsize=LABEL_SIZE)
    
    # add a space between each column
    #  for i in range(1, n_channels):
    #     ax_heatmap.axvline(x=i, color='white', linestyle='-', linewidth=1.5, alpha=0.5)
    # Add vertical grey lines between columns
    for i in range(1, n_channels):
        ax_heatmap.axvline(x=i, color='gray', linestyle='-', linewidth=.5, alpha=1)

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
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
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
    
    with open(output_path.replace('.png', '.md'), 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown description saved to {output_path.replace('.png', '.md')}")


# Create the hierarchical clustered heatmap with x linkage
output_filename = "hierarchical_clustering.png"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# ward': 'Minimizes variance within clusters; tends to create compact, equal-sized clusters',
# complete': 'Uses maximum distance between elements; sensitive to outliers, creates more balanced clusters',
# average': 'Uses average distance between all pairs; moderately robust to outliers, creates natural clusters',
# single': 'Uses minimum distance between elements; can create elongated clusters due to chaining effect'

method = 'ward'

fig, ax_heatmap = create_hierarchical_clustered_heatmap(
    soma_barcodes, 
    method=method,
    output_path=output_path
)

# Get shape information for the markdown file
n_somas, n_channels = soma_barcodes.shape

# Create the markdown description
create_markdown_description(output_path, method, n_somas, n_channels)

# Show the plot (optional - can be commented out for automated runs)
# plt.show()
