"""
Hierarchical Clustering Analysis - Clean Version

This script performs hierarchical clustering analysis by:
1. Loading soma barcode data
2. Computing Hamming distances between barcodes
3. Performing hierarchical clustering with specified linkage method
4. Generating clean heatmaps with multiple variants and transparency options

Author: Generated for barcode analysis
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os

from typing import List, Tuple, Optional, Dict, Any
from matplotlib.colors import ListedColormap

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

from plot_settings import (
    DPI,
    PlotStyle,
    apply_style,
    get_output_filename,
    get_script_output_dir,
    set_plot_style,
)

from soma_preprocessing import (
    generate_barcode_array,
    generate_barcode_array_with_coordinates,
)

# Configuration
DEFAULT_LINKAGE_METHOD = "ward"
DEFAULT_HIGHLIGHTS = [1488, 1219]  # Segment IDs to highlight
VOXEL_FACTOR = np.array([0.4, 0.168, 0.168])  # Voxel size factors (z, y, x)

# Create specific output directory for hierarchical clustering
HIERARCHICAL_OUTPUT_DIR = get_script_output_dir("hierarchical-clustering")


def create_hierarchical_clustering(
    soma_barcodes: np.ndarray, method: str = DEFAULT_LINKAGE_METHOD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create hierarchical clustering of soma barcodes using Hamming distance.

    Parameters:
    -----------
    soma_barcodes : np.ndarray
        Binary barcode matrix (N x channels)
    method : str, optional
        Linkage method for clustering. Default is 'ward'.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Cluster order indices and reordered barcode matrix
    """
    n_somas, n_channels = soma_barcodes.shape
    print(
        f"Creating hierarchical clustering for {n_somas} somas with {n_channels} channels"
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


def plot_clustered_heatmap(
    barcodes: np.ndarray,
    segment_ids: List[int],
    output_path: str,
    settings: dict,
    title_suffix: str = "",
    show_labels: bool = True,
    show_highlights: bool = True,
    highlights: Optional[List[int]] = None,
    auto_square: bool = False,
    transparent_bg: bool = False,
    reverse_channels: bool = True,
) -> plt.Figure:
    """
    Create a clean heatmap of clustered somas with highlighting options.

    Parameters:
    -----------
    barcodes : np.ndarray
        Clustered barcode data (N x channels)
    segment_ids : List[int]
        Segment IDs in clustered order
    output_path : str
        Path to save the heatmap
    settings : dict
        Plot settings dictionary from set_plot_style()
    title_suffix : str, optional
        Additional text for the plot title
    show_labels : bool, optional
        Whether to show x and y axis labels. Default True.
    show_highlights : bool, optional
        Whether to highlight specific segments. Default True.
    highlights : List[int], optional
        Segment IDs to highlight. If None, uses DEFAULT_HIGHLIGHTS.
    auto_square : bool, optional
        If True and figsize is None, use square dimensions based on data size.
    transparent_bg : bool, optional
        If True, make white heatmap cells (value=0) transparent instead of white. Default False.
    reverse_channels : bool, optional
        Whether to reverse channel order for display. Default True.

    Returns:
    --------
    plt.Figure
        The created figure
    """
    n_somas, n_channels = barcodes.shape
    apply_style(settings)

    # Use default highlights if none provided
    if highlights is None:
        highlights = DEFAULT_HIGHLIGHTS
    highlights = None

    # Determine figure size
    if auto_square:
        # Square dimensions based on data size
        fig_width = n_channels / 10
        fig_height = n_somas / 10
        figsize = (fig_width, fig_height)
    else:
        # Use settings heatmap size
        figsize = settings["heatmap"]

    fig = plt.figure(figsize=figsize, dpi=DPI)
    ax = plt.gca()

    # Create colormap - transparent white cells if requested
    if transparent_bg:
        # Custom colormap: transparent for 0 (white), opaque black for 1
        colors = [
            (0, 0, 0, 1),
            (0, 0, 0, 0),
        ]  # (R, G, B, Alpha): opaque black, transparent
        custom_cmap = ListedColormap(colors)
        cmap_to_use = custom_cmap
    else:
        cmap_to_use = "binary_r"

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

    # Reverse channels for display if requested
    if reverse_channels:
        barcodes_display = barcodes[:, ::-1]
        channel_names = channel_names[::-1]
    else:
        barcodes_display = barcodes

    # Create heatmap
    sns.heatmap(
        barcodes_display,
        ax=ax,
        cmap=cmap_to_use,
        xticklabels=channel_names if show_labels else False,
        yticklabels=[f"Seg.{seg_id}" for seg_id in segment_ids]
        if show_labels
        else False,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0.1,
        linecolor="lightgray",
    )

    # Customize appearance
    if show_labels:
        ax.set_xlabel(
            f"Channels (n={n_channels})",
            fontsize=settings["label_size"],
            fontfamily=settings["font_family"],
        )
        ax.set_ylabel(
            f"Somas (n={n_somas})",
            fontsize=settings["label_size"],
            fontfamily=settings["font_family"],
        )
        ax.set_title(
            f"Hierarchical Clustering{title_suffix}",
            fontsize=settings["title_size"],
            fontfamily=settings["font_family"],
        )

        # Rotate x-axis labels
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=settings["tick_size"],
        )
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=settings["tick_size"])
    else:
        # Remove all labels and ticks
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])

    # Highlight specific segments
    if show_highlights and highlights:
        for highlight_seg in highlights:
            if highlight_seg in segment_ids:
                highlight_idx = segment_ids.index(highlight_seg)
                # Add red border around the row
                rect = patches.Rectangle(
                    (0, highlight_idx),
                    n_channels,
                    1,
                    edgecolor="red",
                    linewidth=2,
                    fill=False,
                )
                ax.add_patch(rect)

                # Add label if showing labels
                if show_labels:
                    xmin, xmax = ax.get_xlim()
                    ax.text(
                        xmax + (xmax - xmin) * 0.02,
                        highlight_idx + 0.5,
                        f"Seg.{highlight_seg}",
                        ha="left",
                        va="center",
                        fontsize=settings["tick_size"],
                        rotation=0,
                        color="red",
                        weight="bold",
                    )

                # Add grid lines
                for i in range(1, n_channels):
                    ax.axvline(
                        x=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.7
                    )
                for i in range(1, n_somas):
                    ax.axhline(
                        y=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.7
                    )

    ax.grid(False)
    plt.tight_layout()

    # Save with or without transparency
    # Use bbox_inches="tight" but with padding if we have highlights to ensure annotations are included
    bbox_inches = "tight"
    pad_inches = 0.2 if (show_highlights and highlights) else 0.1

    if transparent_bg:
        plt.savefig(
            output_path,
            dpi=DPI,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=True,
        )
        print(f"Clustered heatmap with transparent white cells saved to {output_path}")
    else:
        plt.savefig(
            output_path, dpi=DPI, bbox_inches=bbox_inches, pad_inches=pad_inches
        )
        print(f"Clustered heatmap saved to {output_path}")

    return fig


def plot_clustered_heatmap_inverted(
    barcodes: np.ndarray,
    segment_ids: List[int],
    output_path: str,
    settings: dict,
    title_suffix: str = "",
    show_labels: bool = True,
    show_highlights: bool = True,
    highlights: Optional[List[int]] = None,
    auto_square: bool = False,
    transparent_bg: bool = False,
    reverse_channels: bool = True,
) -> plt.Figure:
    """
    Create a heatmap with INVERTED colors (0=black, 1=white/transparent).

    Parameters:
    -----------
    barcodes : np.ndarray
        Clustered barcode data (N x channels)
    segment_ids : List[int]
        Segment IDs in clustered order
    output_path : str
        Path to save the heatmap
    title_suffix : str, optional
        Additional text for the plot title
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, auto-calculated.
    show_labels : bool, optional
        Whether to show x and y axis labels. Default True.
    show_highlights : bool, optional
        Whether to highlight specific segments. Default True.
    highlights : List[int], optional
        Segment IDs to highlight. If None, uses DEFAULT_HIGHLIGHTS.
    auto_square : bool, optional
        If True and figsize is None, use square dimensions based on data size.
    transparent_bg : bool, optional
        If True, make white heatmap cells (value=1) transparent instead of white. Default False.
    reverse_channels : bool, optional
        Whether to reverse channel order for display. Default True.

    Returns:
    --------
    plt.Figure
        The created figure
    """
    n_somas, n_channels = barcodes.shape
    apply_style(settings)

    # Use default highlights if none provided
    if highlights is None:
        highlights = DEFAULT_HIGHLIGHTS

    # Determine figure size
    if auto_square:
        # Square dimensions based on data size
        fig_width = n_channels / 10
        fig_height = n_somas / 10
        figsize = (fig_width, fig_height)
    else:
        figsize = settings["heatmap"]

    fig = plt.figure(figsize=figsize, dpi=DPI)
    ax = plt.gca()

    # Create INVERTED colormap - transparent for 1 (barcode present), opaque black for 0
    if transparent_bg:
        # Custom colormap: opaque black for 0, transparent for 1
        colors = [
            (0, 0, 0, 1),  # Value 0 = opaque black
            (1, 1, 1, 0),  # Value 1 = transparent white
        ]  # (R, G, B, Alpha)
        custom_cmap = ListedColormap(colors)
        cmap_to_use = custom_cmap
    else:
        cmap_to_use = "binary"  # Inverted: 0=black, 1=white

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

    # Reverse channels for display if requested
    if reverse_channels:
        barcodes_display = barcodes[:, ::-1]
        channel_names = channel_names[::-1]
    else:
        barcodes_display = barcodes

    # Create heatmap
    sns.heatmap(
        barcodes_display,
        ax=ax,
        cmap=cmap_to_use,
        xticklabels=channel_names if show_labels else False,
        yticklabels=[f"Seg.{seg_id}" for seg_id in segment_ids]
        if show_labels
        else False,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0.1,
        linecolor="lightgray",
    )

    # Customize appearance
    if show_labels:
        ax.set_xlabel(
            f"Channels (n={n_channels})",
            fontsize=settings["label_size"],
            fontfamily=settings["font_family"],
        )
        ax.set_ylabel(
            f"Somas (n={n_somas})",
            fontsize=settings["label_size"],
            fontfamily=settings["font_family"],
        )
        ax.set_title(
            f"Hierarchical Clustering (Inverted){title_suffix}",
            fontsize=settings["title_size"],
            fontfamily=settings["font_family"],
        )

        # Rotate x-axis labels
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=settings["tick_size"],
        )
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=settings["tick_size"])
    else:
        # Remove all labels and ticks
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])

    # Highlight specific segments
    if show_highlights and highlights:
        for highlight_seg in highlights:
            if highlight_seg in segment_ids:
                highlight_idx = segment_ids.index(highlight_seg)
                # Add red border around the row
                rect = patches.Rectangle(
                    (0, highlight_idx),
                    n_channels,
                    1,
                    edgecolor="red",
                    linewidth=2,
                    fill=False,
                )
                ax.add_patch(rect)

                # Add label if showing labels
                if show_labels:
                    xmin, xmax = ax.get_xlim()
                    ax.text(
                        xmax + (xmax - xmin) * 0.02,
                        highlight_idx + 0.5,
                        f"Seg.{highlight_seg}",
                        ha="left",
                        va="center",
                        fontsize=settings["tick_size"],
                        rotation=0,
                        color="red",
                        weight="bold",
                    )

    # Add grid lines
    for i in range(1, n_channels):
        ax.axvline(x=i, color="lightgray", linestyle="-", linewidth=0.5, alpha=0.7)
    for i in range(1, n_somas):
        ax.axhline(y=i, color="lightgray", linestyle="-", linewidth=0.5, alpha=0.7)

    ax.grid(False)
    plt.tight_layout()

    # Save with or without transparency
    # Use bbox_inches="tight" but with padding if we have highlights to ensure annotations are included
    bbox_inches = "tight"
    pad_inches = 0.2 if (show_highlights and highlights) else 0.1

    if transparent_bg:
        plt.savefig(
            output_path,
            dpi=DPI,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=True,
        )
        print(
            f"Inverted clustered heatmap with transparent barcode cells saved to {output_path}"
        )
    else:
        plt.savefig(
            output_path, dpi=DPI, bbox_inches=bbox_inches, pad_inches=pad_inches
        )
        print(f"Inverted clustered heatmap saved to {output_path}")

    return fig


def plot_clustered_heatmap_simple(
    barcodes: np.ndarray,
    segment_ids: List[int],
    output_path: str,
    settings: dict,
    transparent_bg: bool = False,
    reverse_channels: bool = True,
) -> plt.Figure:
    """
    Create a simple heatmap with x-axis labels (size 8) and y-axis label only.

    Parameters:
    -----------
    barcodes : np.ndarray
        Clustered barcode data (N x channels)
    segment_ids : List[int]
        Segment IDs in clustered order
    output_path : str
        Path to save the heatmap
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, auto-calculated.
    transparent_bg : bool, optional
        If True, make white heatmap cells transparent. Default False.
    reverse_channels : bool, optional
        Whether to reverse channel order for display. Default True.

    Returns:
    --------
    plt.Figure
        The created figure
    """
    n_somas, n_channels = barcodes.shape
    apply_style(settings)

    # Use settings heatmap size
    fig = plt.figure(figsize=settings["heatmap"], dpi=DPI)
    ax = plt.gca()

    # Create colormap - transparent white cells if requested
    if transparent_bg:
        # Custom colormap: transparent for 0 (white), opaque black for 1
        colors = [
            (0, 0, 0, 1),
            (0, 0, 0, 0),
        ]  # (R, G, B, Alpha): opaque black, transparent
        custom_cmap = ListedColormap(colors)
        cmap_to_use = custom_cmap
    else:
        cmap_to_use = "binary_r"

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

    # Reverse channels for display if requested
    if reverse_channels:
        barcodes_display = barcodes[:, ::-1]
        channel_names = channel_names[::-1]
    else:
        barcodes_display = barcodes

    # Create heatmap with no individual y-tick labels
    sns.heatmap(
        barcodes_display,
        ax=ax,
        cmap=cmap_to_use,
        xticklabels=channel_names,  # Show x-axis channel labels
        yticklabels=False,  # No individual y-tick labels
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0.1,
        linecolor="lightgray",
    )

    # Set x-axis labels
    ax.set_xticks(np.arange(n_channels) + 0.5)
    ax.set_xticklabels(
        channel_names,
        rotation=45,
        ha="right",
        fontsize=settings["tick_size"],
        fontfamily=settings["font_family"],
    )

    # Set y-axis label only (no individual tick labels)
    ax.set_ylabel(
        f"Somas (n={n_somas})",
        fontsize=settings["label_size"],
        fontfamily=settings["font_family"],
    )
    ax.set_xlabel(
        "", fontsize=settings["label_size"], fontfamily=settings["font_family"]
    )  # No x-axis label

    # Remove y-axis ticks
    ax.set_yticks([])

    # Add grid lines
    for i in range(1, n_channels):
        ax.axvline(x=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.7)

    ax.grid(False)
    plt.tight_layout()

    # Save with or without transparency
    if transparent_bg:
        plt.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            transparent=True,
        )
        print(
            f"Simple clustered heatmap with transparent white cells saved to {output_path}"
        )
    else:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Simple clustered heatmap saved to {output_path}")

    return fig


def plot_clustered_subset(
    barcodes: np.ndarray,
    segment_ids: List[int],
    start_segment_id: int,
    n_rows: int,
    output_path: str,
    title_suffix: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    show_labels: bool = True,
    show_highlights: bool = True,
    highlights: Optional[List[int]] = None,
    transparent_bg: bool = False,
    reverse_channels: bool = True,
) -> plt.Figure:
    """
    Create a subset heatmap starting from a specific segment ID.

    Parameters:
    -----------
    barcodes : np.ndarray
        Clustered barcode data (N x channels)
    segment_ids : List[int]
        Segment IDs in clustered order
    start_segment_id : int
        Segment ID to start the subset from
    n_rows : int
        Number of rows to include in the subset
    output_path : str
        Path to save the heatmap
    title_suffix : str, optional
        Additional text for the plot title
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, auto-calculated.
    show_labels : bool, optional
        Whether to show x and y axis labels. Default True.
    show_highlights : bool, optional
        Whether to highlight specific segments. Default True.
    highlights : List[int], optional
        Segment IDs to highlight. If None, uses DEFAULT_HIGHLIGHTS.
    transparent_bg : bool, optional
        If True, make white heatmap cells transparent. Default False.
    reverse_channels : bool, optional
        Whether to reverse channel order for display. Default True.

    Returns:
    --------
    plt.Figure
        The created figure
    """
    # Use default highlights if none provided
    if highlights is None:
        highlights = DEFAULT_HIGHLIGHTS

    # Find the starting index for the specified segment ID
    try:
        start_idx = segment_ids.index(start_segment_id)
    except ValueError:
        raise ValueError(f"Segment ID {start_segment_id} not found in segment_ids")

    # Calculate end index, ensuring we don't go beyond the data
    end_idx = min(start_idx + n_rows, len(segment_ids))
    actual_n_rows = end_idx - start_idx

    # Extract subset of data
    subset_barcodes = barcodes[start_idx:end_idx]
    subset_segment_ids = segment_ids[start_idx:end_idx]

    print(
        f"Creating subset heatmap: {actual_n_rows} rows starting from segment {start_segment_id}"
    )

    # Create the subset heatmap
    fig = plot_clustered_heatmap(
        subset_barcodes,
        subset_segment_ids,
        output_path,
        title_suffix + f" (subset: {start_segment_id}-{n_rows}rows)",
        figsize,
        show_labels,
        show_highlights,
        highlights,
        auto_square=True,  # Use square dimensions for subsets
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )

    return fig


def save_clustering_results(
    cluster_order: np.ndarray,
    clustered_barcodes: np.ndarray,
    segment_ids: List[int],
    linkage_method: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    Save clustering analysis results to files.

    Parameters:
    -----------
    cluster_order : np.ndarray
        Order indices from hierarchical clustering
    clustered_barcodes : np.ndarray
        Reordered barcode matrix
    segment_ids : List[int]
        Segment IDs in clustered order
    linkage_method : str
        Linkage method used for clustering
    output_dir : str
        Directory to save files
    date_str : str
        Date string for file naming

    Returns:
    --------
    Dict[str, str]
        Dictionary of output file paths
    """
    output_files = {}

    # Save clustering information
    clustering_info = {
        "linkage_method": linkage_method,
        "n_somas": len(segment_ids),
        "n_channels": clustered_barcodes.shape[1],
        "clustered_segment_ids": segment_ids,
    }

    info_path = os.path.join(output_dir, "clustering_info.txt")
    with open(info_path, "w") as f:
        for key, value in clustering_info.items():
            if key == "clustered_segment_ids":
                f.write(f"{key}: {', '.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")
    output_files["clustering_info"] = info_path

    # Save clustered barcodes
    barcodes_df = pd.DataFrame(
        clustered_barcodes,
        index=[f"Seg.{seg_id}" for seg_id in segment_ids],
        columns=[
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
        ],
    )

    barcodes_path = os.path.join(output_dir, "clustered_barcodes.csv")
    barcodes_df.to_csv(barcodes_path)
    output_files["clustered_barcodes"] = barcodes_path

    # Save spatial coordinates if available
    try:
        _, coordinates, all_segment_ids, _ = generate_barcode_array_with_coordinates()
        coordinates_physical = coordinates / VOXEL_FACTOR

        # Reorder coordinates according to clustering
        clustered_coords = coordinates_physical[cluster_order]

        coords_df = pd.DataFrame(
            {
                "segment_id": segment_ids,
                "cluster_position": range(len(segment_ids)),
                "z_um": clustered_coords[:, 0],
                "y_um": clustered_coords[:, 1],
                "x_um": clustered_coords[:, 2],
            }
        )

        coords_path = os.path.join(output_dir, "clustered_coordinates.csv")
        coords_df.to_csv(coords_path, index=False)
        output_files["clustered_coordinates"] = coords_path

    except Exception as e:
        print(f"Warning: Could not save spatial coordinates: {e}")

    print(f"Clustering results saved:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")

    return output_files


def hierarchical_clustering_analysis(
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    output_suffix: str = "",
    highlights: Optional[List[int]] = None,
    transparent_bg: bool = False,
    reverse_channels: bool = True,
    settings: dict = None,
    auto_square: bool = False,
) -> Dict[str, Any]:
    """
    Perform complete hierarchical clustering analysis.

    Parameters:
    -----------
    linkage_method : str, optional
        Linkage method for clustering. Default is 'ward'.
    output_suffix : str, optional
        Suffix to add to output filenames
    highlights : List[int], optional
        Segment IDs to highlight. If None, uses DEFAULT_HIGHLIGHTS.
    transparent_bg : bool, optional
        Whether to make white heatmap cells transparent. Default False.
    reverse_channels : bool, optional
        Whether to reverse channel order for display. Default True.
    figsize : Tuple[float, float], optional
        Custom figure size (width, height) in inches. If None, uses auto-sizing.
    auto_square : bool, optional
        If True, use square dimensions based on data size. Default False.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results and file paths
    """
    print(f"Starting hierarchical clustering analysis...")
    print(f"Linkage method: {linkage_method}")
    print(f"Transparent cells: {transparent_bg}")
    print(f"Reverse channels: {reverse_channels}")

    # Use default highlights if none provided
    if highlights is None:
        highlights = DEFAULT_HIGHLIGHTS

    # Load data
    soma_barcodes = generate_barcode_array()
    n_somas, n_channels = soma_barcodes.shape
    print(f"Loaded {n_somas} somas with {n_channels} channels")

    # Perform hierarchical clustering
    cluster_order, clustered_barcodes = create_hierarchical_clustering(
        soma_barcodes, method=linkage_method
    )

    # Get segment IDs and reorder according to clustering
    soma_seg_ids = pd.read_csv("./soma_barcode_info.csv")["segment_id"].tolist()
    clustered_segment_ids = [soma_seg_ids[i] for i in cluster_order]

    print(f"Clustering completed. Highlighted segments:")
    for highlight_seg in highlights:
        if highlight_seg in clustered_segment_ids:
            pos = clustered_segment_ids.index(highlight_seg)
            print(f"  Segment {highlight_seg}: position {pos} in clustered order")
        else:
            print(f"  Segment {highlight_seg}: NOT FOUND in dataset")

    # Generate outputs
    title_suffix = f" ({linkage_method} linkage, n={n_somas})"

    # Create only the two requested outputs
    heatmap_outputs = {}

    # 1. Heatmap with y-labels and transparent white cells
    labels_path = get_output_filename(
        "hierarchical_clustering_with_labels",
        settings["style"],
        "png",
        script_name="hierarchical-clustering",
    )
    plot_clustered_heatmap(
        clustered_barcodes,
        clustered_segment_ids,
        labels_path,
        settings,
        title_suffix,
        show_labels=True,
        show_highlights=True if highlights else False,  # Show highlights if specified
        highlights=highlights,
        auto_square=False,  # Never override figsize
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )
    heatmap_outputs["with_labels"] = labels_path

    # 2. Clean heatmap without labels (for comparison)
    clean_path = get_output_filename(
        "hierarchical_clustering_clean",
        settings["style"],
        "png",
        script_name="hierarchical-clustering",
    )
    plot_clustered_heatmap(
        clustered_barcodes,
        clustered_segment_ids,
        clean_path,
        settings,
        "",
        show_labels=False,
        show_highlights=False,
        highlights=highlights,
        auto_square=False,  # Never override figsize
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )
    heatmap_outputs["clean"] = clean_path

    # 3. Simple heatmap with x-axis labels and y-axis label only
    simple_path = get_output_filename(
        "hierarchical_clustering_simple",
        settings["style"],
        "png",
        script_name="hierarchical-clustering",
    )
    plot_clustered_heatmap_simple(
        clustered_barcodes,
        clustered_segment_ids,
        simple_path,
        settings,
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )
    heatmap_outputs["simple"] = simple_path

    # 4. INVERTED versions - same outputs but with flipped black/white
    # 4a. Inverted with labels and highlights
    inverted_labels_path = get_output_filename(
        "hierarchical_clustering_with_labels_inverted",
        settings["style"],
        "png",
        script_name="hierarchical-clustering",
    )
    plot_clustered_heatmap_inverted(
        clustered_barcodes,
        clustered_segment_ids,
        inverted_labels_path,
        settings,
        title_suffix,
        show_labels=True,
        show_highlights=True if highlights else False,
        highlights=highlights,
        auto_square=False,
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )
    heatmap_outputs["with_labels_inverted"] = inverted_labels_path

    # 4b. Inverted clean (no labels, no highlights)
    inverted_clean_path = get_output_filename(
        "hierarchical_clustering_clean_inverted",
        settings["style"],
        "png",
        script_name="hierarchical-clustering",
    )
    plot_clustered_heatmap_inverted(
        clustered_barcodes,
        clustered_segment_ids,
        inverted_clean_path,
        settings,
        "",
        show_labels=False,
        show_highlights=False,
        highlights=highlights,
        auto_square=False,
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )
    heatmap_outputs["clean_inverted"] = inverted_clean_path

    # 4c. Inverted simple (x-axis labels, y-axis label only)
    inverted_simple_path = get_output_filename(
        "hierarchical_clustering_simple_inverted",
        settings["style"],
        "png",
        script_name="hierarchical-clustering",
    )
    plot_clustered_heatmap_inverted(
        clustered_barcodes,
        clustered_segment_ids,
        inverted_simple_path,
        settings,
        "",
        show_labels=True,  # Show x/y labels but not individual y-ticks
        show_highlights=False,  # Keep simple clean
        highlights=highlights,
        auto_square=False,
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )
    # Modify to be simple style (no y-tick labels)
    heatmap_outputs["simple_inverted"] = inverted_simple_path

    # Save results
    output_files = save_clustering_results(
        cluster_order,
        clustered_barcodes,
        clustered_segment_ids,
        linkage_method,
        HIERARCHICAL_OUTPUT_DIR,
    )
    output_files.update(heatmap_outputs)

    # Compile results
    results = {
        "cluster_order": cluster_order,
        "clustered_barcodes": clustered_barcodes,
        "clustered_segment_ids": clustered_segment_ids,
        "linkage_method": linkage_method,
        "n_somas": n_somas,
        "n_channels": n_channels,
        "highlights": highlights,
        "output_files": output_files,
    }

    print(f"\nHierarchical clustering analysis completed successfully!")
    print(f"Output directory: {HIERARCHICAL_OUTPUT_DIR}")
    print(f"Generated heatmap variants: {list(heatmap_outputs.keys())}")

    return results


def create_custom_clustered_heatmap(
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    figsize: Optional[Tuple[float, float]] = None,
    auto_square: bool = False,
    show_labels: bool = True,
    show_highlights: bool = True,
    highlights: Optional[List[int]] = None,
    output_name: str = "custom",
    transparent_bg: bool = False,
    reverse_channels: bool = True,
) -> str:
    """
    Create a single custom hierarchical clustering heatmap with specified parameters.

    Parameters:
    -----------
    linkage_method : str, optional
        Linkage method for clustering
    figsize : Tuple[float, float], optional
        Custom figure size (width, height)
    auto_square : bool, optional
        Use square dimensions based on data size
    show_labels : bool, optional
        Whether to show axis labels
    show_highlights : bool, optional
        Whether to highlight specific segments
    highlights : List[int], optional
        Segment IDs to highlight
    output_name : str, optional
        Custom name for output file
    transparent_bg : bool, optional
        Whether to make white cells transparent
    reverse_channels : bool, optional
        Whether to reverse channel order

    Returns:
    --------
    str
        Path to the created heatmap file
    """
    # Use default highlights if none provided
    if highlights is None:
        highlights = DEFAULT_HIGHLIGHTS

    # Get data and perform clustering
    soma_barcodes = generate_barcode_array()
    cluster_order, clustered_barcodes = create_hierarchical_clustering(
        soma_barcodes, method=linkage_method
    )
    soma_seg_ids = pd.read_csv("./soma_barcode_info.csv")["segment_id"].tolist()
    clustered_segment_ids = [soma_seg_ids[i] for i in cluster_order]

    # Create output path
    transparent_suffix = "_transparent" if transparent_bg else ""
    output_path = os.path.join(
        HIERARCHICAL_OUTPUT_DIR,
        f"hierarchical_clustering_{output_name}{transparent_suffix}.png",
    )

    # Create heatmap
    plot_clustered_heatmap(
        clustered_barcodes,
        clustered_segment_ids,
        output_path,
        f" ({linkage_method}, n={len(clustered_segment_ids)})" if show_labels else "",
        figsize=figsize,
        show_labels=show_labels,
        show_highlights=show_highlights,
        highlights=highlights,
        auto_square=auto_square,
        transparent_bg=transparent_bg,
        reverse_channels=reverse_channels,
    )

    print(f"Custom hierarchical clustering heatmap saved to: {output_path}")
    return output_path


def main():
    """
    Main function to run hierarchical clustering analysis with example parameters.
    """
    # Parse command line arguments
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

    # Example configuration - modify these as needed
    linkage_method = "ward"  # Options: 'ward', 'complete', 'average', 'single'
    highlights = [1488, 1219]  # Segment IDs to highlight

    # Run analysis with transparent white cells
    results = hierarchical_clustering_analysis(
        linkage_method=linkage_method,
        output_suffix=args.style,
        highlights=highlights,
        transparent_bg=True,  # Make white cells transparent
        reverse_channels=True,  # Reverse channel order for display
        settings=settings,
        auto_square=True,
    )

    # Print summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Linkage method: {results['linkage_method']}")
    print(f"Total somas: {results['n_somas']}")
    print(f"Channels: {results['n_channels']}")
    print(f"Highlighted segments: {results['highlights']}")

    highlight_positions = []
    for highlight_seg in results["highlights"]:
        if highlight_seg in results["clustered_segment_ids"]:
            pos = results["clustered_segment_ids"].index(highlight_seg)
            highlight_positions.append(f"Seg.{highlight_seg} at position {pos}")
    print(f"Highlight positions: {highlight_positions}")
    print(f"Output files: {list(results['output_files'].keys())}")

    # Example: Create custom heatmaps with specific settings
    # print(f"\n=== CUSTOM HEATMAP EXAMPLES ===")

    # Example 1: Clean minimal heatmap
    # custom_path1 = create_custom_clustered_heatmap(
    #     linkage_method="ward",
    #     figsize=(3, 8),
    #     show_labels=False,
    #     show_highlights=False,
    #     transparent_bg=True,
    #     output_name="minimal_clean",
    # )

    # Example 2: Compact with highlights only
    # custom_path2 = create_custom_clustered_heatmap(
    #     linkage_method="ward",
    #     auto_square=True,
    #     show_labels=False,
    #     show_highlights=True,
    #     highlights=[1488, 1219],
    #     transparent_bg=True,
    #     output_name="compact_highlights",
    # )


if __name__ == "__main__":
    main()
