# this script generates a hierarchical clustering plot of the soma barcodes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from datetime import datetime

from scipy.spatial.distance import pdist
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
FIG_SIZE_SUBSET_HEATMAP = (2, 3)
highlights = [1488, 1219]

# Voxel size factors for converting to physical coordinates (z, y, x)
VOXEL_FACTOR = np.array([0.4, 0.168, 0.168])

from soma_preprocessing import (
    generate_barcode_array,
    generate_barcode_array_with_coordinates,
)

# Generate barcodes array
soma_barcodes = generate_barcode_array()


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
    soma_seg_ids=None,
    figsize=None,
):
    """Plot the hierarchical clustering of soma barcodes."""
    n_somas, n_channels = clustered_barcodes.shape
    set_style()
    # Create figure for the heatmap
    if figsize is None:
        figsize = FIG_SIZE_HEATMAP
    fig = plt.figure(figsize=figsize, dpi=DPI)
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
    # Reverse the order of the channels if requested
    if reverse_channels:
        clustered_barcodes = clustered_barcodes[:, ::-1]
        channel_names = channel_names[::-1]

    # Plot the heatmap with inverted colors (binary_r) and no grid lines
    ax = sns.heatmap(
        clustered_barcodes,
        ax=ax_heatmap,
        cmap="binary_r",
        xticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0,
        linecolor="white",
    )

    # Set axis labels and ticks
    ax_heatmap.set_xticks(np.arange(n_channels) + 0.5)
    ax_heatmap.set_xticklabels(
        channel_names, rotation=90, fontsize=8, fontfamily=FONT_FAMILY
    )
    ax_heatmap.set_xlabel(f"Channels (n={n_channels})", fontsize=8)
    ax_heatmap.set_ylabel(f"Somas (n={n_somas})", fontsize=8)

    # Clear the y axis ticks
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
        text.set_fontsize(8)

    ax_heatmap.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax_heatmap.yaxis.label.set_fontfamily(FONT_FAMILY)

    plt.tight_layout()

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    return fig, ax_heatmap


def plot_hierarchical_clustering_subset(
    clustered_barcodes,
    soma_seg_ids,
    start_segment_id,
    n_rows,
    output_path=None,
    reverse_channels=False,
    highlights=None,
    figsize=None,
    show_segment_ids=False,
):
    """
    Plot a subset of the hierarchical clustering starting from a specific segment ID.

    Parameters:
    -----------
    clustered_barcodes : numpy.ndarray
        The clustered barcode matrix
    soma_seg_ids : list
        List of segment IDs in clustered order
    start_segment_id : int
        Segment ID to start the subset from
    n_rows : int
        Number of rows to include in the subset
    output_path : str, optional
        Path to save the figure
    reverse_channels : bool, optional
        Whether to reverse the channel order
        highlights : list, optional
        List of segment IDs to highlight with red rectangles
        figsize : tuple, optional
        Figure size as (width, height). If None, uses automatic sizing based on n_rows
    show_segment_ids : bool, optional
        Whether to show segment IDs on the y-axis. Default is False

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Find the starting index for the specified segment ID
    try:
        start_idx = soma_seg_ids.index(start_segment_id)
    except ValueError:
        raise ValueError(f"Segment ID {start_segment_id} not found in soma_seg_ids")

    # Calculate end index, ensuring we don't go beyond the data
    end_idx = min(start_idx + n_rows, len(soma_seg_ids))
    actual_n_rows = end_idx - start_idx

    # Extract subset of data
    subset_barcodes = clustered_barcodes[start_idx:end_idx]
    subset_seg_ids = soma_seg_ids[start_idx:end_idx]

    n_somas_subset, n_channels = subset_barcodes.shape
    print(
        f"Creating subset heatmap: {actual_n_rows} rows starting from segment {start_segment_id} (index {start_idx})"
    )

    # Set up the plot
    set_style()
    if figsize is None:
        # Auto-size based on number of rows, with extra width if highlights are present
        base_width = FIG_SIZE_SUBSET_HEATMAP[0]
        if highlights and any(h in subset_seg_ids for h in highlights):
            # Add extra width for highlight annotations
            width = base_width + 0.8
        else:
            width = base_width
        figsize = (
            width,
            min(FIG_SIZE_SUBSET_HEATMAP[1], actual_n_rows * 0.1 + 1),
        )
    fig = plt.figure(figsize=figsize, dpi=DPI)
    ax_heatmap = plt.gca()

    # Channel names
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

    # Reverse the order of the channels if requested
    if reverse_channels:
        subset_barcodes = subset_barcodes[:, ::-1]
        channel_names = channel_names[::-1]

    # Plot the heatmap
    ax = sns.heatmap(
        subset_barcodes,
        ax=ax_heatmap,
        cmap="binary_r",
        xticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0,
        linecolor="white",
    )

    # Set axis labels and ticks
    ax_heatmap.set_xticks(np.arange(n_channels) + 0.5)
    ax_heatmap.set_xticklabels(
        channel_names, rotation=90, fontsize=8, fontfamily=FONT_FAMILY
    )
    ax_heatmap.set_xlabel(f"Channels (n={n_channels})", fontsize=8)
    ax_heatmap.set_ylabel(f"Somas (n={actual_n_rows})", fontsize=8)

    # Set y-axis ticks based on show_segment_ids parameter
    if show_segment_ids:
        ax_heatmap.set_yticks(np.arange(actual_n_rows) + 0.5)
        ax_heatmap.set_yticklabels(
            [f"Seg.{seg_id}" for seg_id in subset_seg_ids],
            rotation=0,
            fontsize=8,
            fontfamily=FONT_FAMILY,
        )
    else:
        ax_heatmap.set_yticks([])
        ax_heatmap.set_ylabel(None)
        ax_heatmap.set_xlabel(None)
        ax_heatmap.set_xticks([])

    # Add light gray lines between channels
    for i in range(1, n_channels):
        ax_heatmap.axvline(
            x=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.5
        )

    # Add light gray lines between rows
    for i in range(1, actual_n_rows):
        ax_heatmap.axhline(
            y=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.3
        )

    ax.grid(False)

    # Set font properties for all text elements
    for text in ax_heatmap.get_xticklabels() + ax_heatmap.get_yticklabels():
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(8)

    ax_heatmap.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax_heatmap.yaxis.label.set_fontfamily(FONT_FAMILY)

    # Ensure proper spacing for rotated labels
    plt.setp(
        ax_heatmap.get_xticklabels(), rotation=90, ha="center", va="top", fontsize=8
    )

    # Add highlighting for specific segments if provided
    if highlights:
        xmin, xmax = ax_heatmap.get_xlim()
        for highlight in highlights:
            if highlight in subset_seg_ids:
                # Find the index within the subset
                highlight_index = subset_seg_ids.index(highlight)

                rect = patches.Rectangle(
                    (xmin, highlight_index),
                    xmax - xmin,
                    1,
                    edgecolor="red",
                    linewidth=0.8,
                    fill=False,
                )
                ax_heatmap.add_patch(rect)

                # Add text label to the right of the heatmap
                ax_heatmap.text(
                    xmax + (xmax - xmin) * 0.02,
                    highlight_index + 0.5,
                    f"Seg.{highlight}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    rotation=0,
                    color="red",
                    weight="bold",
                )

    # Adjust layout to prevent label overlap
    if highlights and any(h in subset_seg_ids for h in highlights):
        # More padding when we have side annotations
        plt.tight_layout(pad=1.0)
    else:
        plt.tight_layout()

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Subset figure saved to {output_path}")

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


def get_spatial_coordinates_for_segments(segment_ids_list):
    """
    Get spatial coordinates for specific segment IDs.

    Parameters:
    -----------
    segment_ids_list : list
        List of segment IDs to get coordinates for

    Returns:
    --------
    dict : Dictionary mapping segment_id -> (z, y, x) coordinates in physical units
    """
    _, coordinates, all_segment_ids, _ = generate_barcode_array_with_coordinates()

    # Convert to physical coordinates
    coordinates_physical = coordinates / VOXEL_FACTOR

    coord_dict = {}
    for i, seg_id in enumerate(all_segment_ids):
        if seg_id in segment_ids_list:
            coord_dict[seg_id] = {
                "z": coordinates_physical[i, 0],
                "y": coordinates_physical[i, 1],
                "x": coordinates_physical[i, 2],
                "coords": coordinates_physical[i],  # Full coordinate array
                "coords_voxel": coordinates[i],  # Original voxel coordinates
            }

    return coord_dict


def get_clustered_spatial_coordinates(cluster_order):
    """
    Get spatial coordinates in clustered order.

    Parameters:
    -----------
    cluster_order : list
        The order of indices after clustering

    Returns:
    --------
    coordinates_clustered : numpy.ndarray
        Spatial coordinates in clustered order (in physical units)
    segment_ids_clustered : list
        Segment IDs in clustered order
    """
    _, coordinates, segment_ids, _ = generate_barcode_array_with_coordinates()

    # Convert to physical coordinates
    coordinates_physical = coordinates / VOXEL_FACTOR

    # Reorder coordinates and segment IDs according to clustering
    coordinates_clustered = coordinates_physical[cluster_order]
    segment_ids_clustered = [segment_ids[i] for i in cluster_order]

    return coordinates_clustered, segment_ids_clustered


def save_coordinates_to_csv(
    coordinates, segment_ids, output_path, clustered_order=False, verbose=True
):
    """
    Save spatial coordinates to a CSV file.

    Parameters:
    -----------
    coordinates : numpy.ndarray
        Array of coordinates with shape (n_somas, 3) for [z, y, x] in physical units
    segment_ids : list
        List of segment IDs corresponding to each coordinate
    output_path : str
        Path where to save the CSV file
    clustered_order : bool, optional
        Whether the coordinates are in clustered order. Default is False
    verbose : bool, optional
        Whether to print information about the saved file. Default is True

    Returns:
    --------
    str : Path to the saved CSV file
    """
    # Create DataFrame with coordinates and segment IDs
    df = pd.DataFrame(
        {
            "segment_id": segment_ids,
            "z_um": coordinates[:, 0],  # Changed to indicate micrometers
            "y_um": coordinates[:, 1],
            "x_um": coordinates[:, 2],
        }
    )

    # Add clustering position if in clustered order
    if clustered_order:
        df["cluster_position"] = range(len(segment_ids))

    # Save to CSV
    df.to_csv(output_path, index=False)

    if verbose:
        order_desc = "clustered order" if clustered_order else "original order"
        print(f"\nSaved {len(segment_ids)} soma coordinates to: {output_path}")
        print(f"Data is in {order_desc}")
        print(f"Columns: {list(df.columns)}")
        print(f"Coordinate ranges (micrometers):")
        print(f"  Z: {coordinates[:, 0].min():.2f} to {coordinates[:, 0].max():.2f} μm")
        print(f"  Y: {coordinates[:, 1].min():.2f} to {coordinates[:, 1].max():.2f} μm")
        print(f"  X: {coordinates[:, 2].min():.2f} to {coordinates[:, 2].max():.2f} μm")
        print(
            f"Voxel factors applied: Z={VOXEL_FACTOR[0]}, Y={VOXEL_FACTOR[1]}, X={VOXEL_FACTOR[2]}"
        )

    return output_path


if __name__ == "__main__":
    # Create the hierarchical clustered heatmap
    orientation = "vertical"
    date = datetime.now().strftime("%y%m%d")
    output_filename = f"hierarchical_clustering_{orientation}_square_{date}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Figure size settings
    main_figsize = FIG_SIZE_HEATMAP  # Width, height for main heatmap
    subset_figsize = FIG_SIZE_SUBSET_HEATMAP  # Width, height for subset heatmap

    method = "ward"

    cluster_order, clustered_barcodes = create_hierarchical_clustering(
        soma_barcodes, method=method
    )

    # Load the soma segment IDs and reorder according to clustering
    soma_seg_ids = pd.read_csv("./soma_barcode_info.csv")["segment_id"].tolist()
    soma_seg_ids = [soma_seg_ids[i] for i in cluster_order]

    fig, ax = plot_hierarchical_clustering(
        clustered_barcodes,
        output_path=output_path,
        orientation=orientation,
        reverse_channels=True,
        soma_seg_ids=soma_seg_ids,
        figsize=main_figsize,
    )

    # Add highlighting for specific somas
    n_somas, n_channels = soma_barcodes.shape
    xmin, xmax = ax.get_xlim()

    for highlight in highlights:
        highlight_index = soma_seg_ids.index(highlight)

        rect = patches.Rectangle(
            (xmin, highlight_index),
            xmax - xmin,
            1,
            edgecolor="red",
            linewidth=0.5,
            fill=False,
        )
        ax.add_patch(rect)

        ax.text(
            xmax + (xmax - xmin) * 0.02,
            highlight_index + 0.5,
            f"Seg.{soma_seg_ids[highlight_index]}",
            ha="left",
            va="center",
            fontsize=8,
            rotation=0,
        )

    # Save the figure
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Figure saved to {output_path}")

    # Create the markdown description
    create_markdown_description(output_path, method, n_somas, n_channels)

    # Example: Get spatial coordinates for highlighted segments
    # highlighted_segments = [1219, 1488]

    coordinates_dict = get_spatial_coordinates_for_segments(highlights)

    print("\nSpatial coordinates for highlighted segments:")
    for seg_id, coords in coordinates_dict.items():
        print(
            f"Segment {seg_id}: z={coords['z']:.2f} μm, y={coords['y']:.2f} μm, x={coords['x']:.2f} μm"
        )

    # Example: Get all coordinates in clustered order
    clustered_coords, clustered_seg_ids = get_clustered_spatial_coordinates(
        cluster_order
    )
    print(f"\nTotal somas with coordinates: {len(clustered_coords)}")
    print(
        f"Coordinate array shape: {clustered_coords.shape}"
    )  # Should be (n_somas, 3) for [z, y, x]

    # Save coordinates to CSV files
    coords_output_clustered = os.path.join(
        OUTPUT_DIR, f"soma_coordinates_clustered_{date}.csv"
    )
    save_coordinates_to_csv(
        clustered_coords,
        clustered_seg_ids,
        coords_output_clustered,
        clustered_order=True,
        verbose=True,
    )

    # Also save original order coordinates for comparison
    _, original_coords, original_seg_ids, _ = generate_barcode_array_with_coordinates()
    # Convert to physical coordinates
    original_coords_physical = original_coords / VOXEL_FACTOR
    coords_output_original = os.path.join(
        OUTPUT_DIR, f"soma_coordinates_original_{date}.csv"
    )
    save_coordinates_to_csv(
        original_coords_physical,
        original_seg_ids,
        coords_output_original,
        clustered_order=False,
        verbose=True,
    )

    # Create zoomed-in subset heatmaps - both versions
    # Example: Start from segment 1488 and show 30 rows
    subset_start_segment = highlights[0]
    subset_n_rows = 30

    # Version 1: Without segment IDs on y-axis
    subset_output_filename_no_ids = f"hierarchical_clustering_subset_{subset_start_segment}_{subset_n_rows}rows_no_ids_{date}.png"
    subset_output_path_no_ids = os.path.join(OUTPUT_DIR, subset_output_filename_no_ids)

    subset_fig_no_ids, subset_ax_no_ids = plot_hierarchical_clustering_subset(
        clustered_barcodes,
        soma_seg_ids,
        subset_start_segment,
        subset_n_rows,
        output_path=subset_output_path_no_ids,
        reverse_channels=True,
        highlights=None,
        figsize=subset_figsize,
        show_segment_ids=False,
    )

    # Version 2: With segment IDs on y-axis
    subset_output_filename_with_ids = f"hierarchical_clustering_subset_{subset_start_segment}_{subset_n_rows}rows_with_ids_{date}.png"
    subset_output_path_with_ids = os.path.join(
        OUTPUT_DIR, subset_output_filename_with_ids
    )

    subset_fig_with_ids, subset_ax_with_ids = plot_hierarchical_clustering_subset(
        clustered_barcodes,
        soma_seg_ids,
        subset_start_segment,
        subset_n_rows,
        output_path=subset_output_path_with_ids,
        reverse_channels=True,
        highlights=None,
        figsize=subset_figsize,
        show_segment_ids=True,
    )

    # Save coordinates for just the subset
    # Extract subset segment IDs and their corresponding coordinates
    try:
        start_idx = soma_seg_ids.index(subset_start_segment)
        end_idx = min(start_idx + subset_n_rows, len(soma_seg_ids))

        # Get subset segment IDs and coordinates
        subset_segment_ids = soma_seg_ids[start_idx:end_idx]
        subset_coordinates = clustered_coords[start_idx:end_idx]

        # Save subset coordinates
        subset_coords_filename = f"soma_coordinates_subset_{subset_start_segment}_{subset_n_rows}rows_{date}.csv"
        subset_coords_output = os.path.join(OUTPUT_DIR, subset_coords_filename)

        save_coordinates_to_csv(
            subset_coordinates,
            subset_segment_ids,
            subset_coords_output,
            clustered_order=True,  # These are in clustered order within the subset
            verbose=True,
        )

        print(f"\nSubset summary:")
        print(f"  Start segment: {subset_start_segment}")
        print(f"  Requested rows: {subset_n_rows}")
        print(f"  Actual rows saved: {len(subset_segment_ids)}")
        print(f"  First segment in subset: {subset_segment_ids[0]}")
        print(f"  Last segment in subset: {subset_segment_ids[-1]}")

    except ValueError:
        print(
            f"Warning: Could not find segment {subset_start_segment} in clustered data for coordinate extraction"
        )
