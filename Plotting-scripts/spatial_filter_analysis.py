"""
Spatial Filtering Analysis

This script performs spatial analysis by:
1. Taking two segment IDs as reference points
2. Calculating a bounding box around those points
3. Filtering all somas within the bounding box
4. Generating a heatmap of the filtered cells

Author: Generated for barcode analysis
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.colors import ListedColormap

from plot_settings import (
    FIG_SIZE_HEATMAP,
    set_style,
    DPI,
    OUTPUT_DIR,
    FONT_FAMILY,
)

from soma_preprocessing import generate_barcode_array_with_coordinates

# Configuration
VOXEL_FACTOR = np.array([0.4, 0.168, 0.168])  # Voxel size factors (z, y, x)
DEFAULT_XY_MARGIN = 50.0  # Default margin in micrometers for x and y
DEFAULT_Z_MARGIN = 10.0  # Default margin in micrometers for z

# Create specific output directory for spatial filter analysis
SPATIAL_FILTER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "spatial_filter_somas")
os.makedirs(SPATIAL_FILTER_OUTPUT_DIR, exist_ok=True)


def get_segment_coordinates(
    coordinates: np.ndarray, segment_ids: List[int], target_segments: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Retrieve coordinates for specific segment IDs.

    Parameters:
    -----------
    coordinates : np.ndarray
        Array of coordinates for all segments (N x 3: z, y, x)
    segment_ids : List[int]
        List of all segment IDs
    target_segments : List[int]
        List of segment IDs to retrieve coordinates for

    Returns:
    --------
    Tuple[np.ndarray, List[int]]
        Coordinates of target segments and their indices in the original array
    """
    target_coords = []
    target_indices = []

    for seg_id in target_segments:
        try:
            idx = segment_ids.index(seg_id)
            target_coords.append(coordinates[idx])
            target_indices.append(idx)
        except ValueError:
            raise ValueError(f"Segment ID {seg_id} not found in dataset")

    return np.array(target_coords), target_indices


def calculate_bounding_box(
    target_coords: np.ndarray,
    xy_margin: float = DEFAULT_XY_MARGIN,
    z_margin: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bounding box around target coordinates.

    Parameters:
    -----------
    target_coords : np.ndarray
        Coordinates of target points (N x 3: z, y, x)
    xy_margin : float, optional
        Margin to add/subtract in x and y dimensions (micrometers)
    z_margin : float, optional
        Margin to add/subtract in z dimension (micrometers). If None, no z filtering.

    Returns:
    --------
    Dict[str, Tuple[float, float]]
        Dictionary with 'x', 'y', and optionally 'z' keys, each containing (min, max) tuples
    """
    # Find min/max coordinates for each dimension
    z_coords = target_coords[:, 0]
    y_coords = target_coords[:, 1]
    x_coords = target_coords[:, 2]

    # Calculate bounding box with margins
    bounding_box = {
        "x": (x_coords.min() - xy_margin, x_coords.max() + xy_margin),
        "y": (y_coords.min() - xy_margin, y_coords.max() + xy_margin),
    }

    # Add z dimension if margin is specified
    if z_margin is not None:
        bounding_box["z"] = (z_coords.min() - z_margin, z_coords.max() + z_margin)

    return bounding_box


def filter_somas_in_bounding_box(
    coordinates: np.ndarray,
    barcodes: np.ndarray,
    segment_ids: List[int],
    bounding_box: Dict[str, Tuple[float, float]],
    specific_segment_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Filter somas that fall within the specified bounding box.

    By default, returns ALL somas within the spatial bounding box.
    Optionally can further filter to only include specific segment IDs.

    Parameters:
    -----------
    coordinates : np.ndarray
        All coordinates (N x 3: z, y, x)
    barcodes : np.ndarray
        All barcode data (N x channels)
    segment_ids : List[int]
        All segment IDs
    bounding_box : Dict[str, Tuple[float, float]]
        Bounding box specifications (spatial filtering)
    specific_segment_ids : List[int], optional
        If provided, further filter to only include these specific segment IDs.
        If None (default), returns all somas within the spatial bounding box.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, List[int], List[int]]
        Filtered coordinates, barcodes, segment IDs, and original indices
    """
    # Create boolean mask for spatial filtering
    mask = np.ones(len(coordinates), dtype=bool)

    # Apply spatial filters based on bounding box
    # Apply x filter
    x_min, x_max = bounding_box["x"]
    mask &= (coordinates[:, 2] >= x_min) & (coordinates[:, 2] <= x_max)

    # Apply y filter
    y_min, y_max = bounding_box["y"]
    mask &= (coordinates[:, 1] >= y_min) & (coordinates[:, 1] <= y_max)

    # Apply z filter if specified
    if "z" in bounding_box:
        z_min, z_max = bounding_box["z"]
        mask &= (coordinates[:, 0] >= z_min) & (coordinates[:, 0] <= z_max)

    # Optionally apply segment ID filter (additional filtering)
    if specific_segment_ids is not None:
        segment_mask = np.array(
            [seg_id in specific_segment_ids for seg_id in segment_ids]
        )
        mask &= segment_mask

    # Apply combined filter
    filtered_indices = np.where(mask)[0]
    filtered_coordinates = coordinates[mask]
    filtered_barcodes = barcodes[mask]
    filtered_segment_ids = [segment_ids[i] for i in filtered_indices]

    return (
        filtered_coordinates,
        filtered_barcodes,
        filtered_segment_ids,
        filtered_indices.tolist(),
    )


def plot_filtered_heatmap(
    barcodes: np.ndarray,
    segment_ids: List[int],
    target_segments: List[int],
    output_path: str,
    title_suffix: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    show_labels: bool = True,
    show_highlights: bool = True,
    auto_square: bool = False,
    transparent_bg: bool = False,
) -> plt.Figure:
    """
    Create a heatmap of filtered somas with target segments highlighted.

    Parameters:
    -----------
    barcodes : np.ndarray
        Barcode data for filtered somas (N x channels)
    segment_ids : List[int]
        Segment IDs for filtered somas
    target_segments : List[int]
        Segment IDs to highlight
    output_path : str
        Path to save the heatmap
    title_suffix : str, optional
        Additional text for the plot title
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, auto-calculated.
    show_labels : bool, optional
        Whether to show x and y axis labels. Default True.
    show_highlights : bool, optional
        Whether to highlight target segments. Default True.
    auto_square : bool, optional
        If True and figsize is None, use square dimensions based on n_channels/10, n_somas/10.
    transparent_bg : bool, optional
        If True, make white heatmap cells (value=0) transparent instead of white. Default False.

    Returns:
    --------
    plt.Figure
        The created figure
    """
    n_somas, n_channels = barcodes.shape
    set_style()

    # Determine figure size
    if figsize is not None:
        fig_width, fig_height = figsize
    elif auto_square:
        # Square dimensions based on data size
        fig_width = n_channels / 10
        fig_height = n_somas / 10
    else:
        # Default auto-sizing
        fig_width = max(4, n_channels / 3)
        fig_height = max(3, n_somas / 5)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=DPI)
    ax = plt.gca()

    # Create colormap - transparent white cells if requested
    if transparent_bg:
        # Custom colormap: transparent for 0 (white), opaque black for 1
        colors = [
            (0, 0, 0, 1),
            (0, 0, 0, 0),
        ]  # (R, G, B, Alpha): transparent, opaque black
        custom_cmap = ListedColormap(colors)
        cmap_to_use = custom_cmap
    else:
        cmap_to_use = "binary_r"

    # Channel names (reversed for display)
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
    ][::-1]

    # Reverse channels for display
    barcodes_display = barcodes[:, ::-1]

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
        ax.set_xlabel(f"Channels (n={n_channels})", fontsize=10, fontfamily=FONT_FAMILY)
        ax.set_ylabel(f"Somas (n={n_somas})", fontsize=10, fontfamily=FONT_FAMILY)
        ax.set_title(
            f"Spatial Filter Analysis{title_suffix}",
            fontsize=12,
            fontfamily=FONT_FAMILY,
        )

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    else:
        # Remove all labels and ticks
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])

    # Highlight target segments
    if show_highlights:
        for target_seg in target_segments:
            if target_seg in segment_ids:
                target_idx = segment_ids.index(target_seg)
                # Add red border around the row
                rect = patches.Rectangle(
                    (0, target_idx),
                    n_channels,
                    1,
                    edgecolor="red",
                    linewidth=2,
                    fill=False,
                )
                ax.add_patch(rect)

    # Add grid lines
    for i in range(1, n_channels):
        ax.axvline(x=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.7)
    for i in range(1, n_somas):
        ax.axhline(y=i, color="whitesmoke", linestyle="-", linewidth=0.5, alpha=0.7)

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
        print(f"Filtered heatmap with transparent white cells saved to {output_path}")
    else:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Filtered heatmap saved to {output_path}")

    return fig


def save_analysis_results(
    bounding_box: Dict[str, Tuple[float, float]],
    filtered_coordinates: np.ndarray,
    filtered_segment_ids: List[int],
    target_segments: List[int],
    output_dir: str,
    date_str: str,
    shuffle_suffix: str = "",
    shuffle_somas: bool = False,
    random_seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Save analysis results to files.

    Parameters:
    -----------
    bounding_box : Dict[str, Tuple[float, float]]
        Bounding box specifications
    filtered_coordinates : np.ndarray
        Coordinates of filtered somas
    filtered_segment_ids : List[int]
        Segment IDs of filtered somas
    target_segments : List[int]
        Original target segment IDs
    output_dir : str
        Directory to save files
    date_str : str
        Date string for file naming
    shuffle_suffix : str, optional
        Suffix indicating if shuffling was applied
    shuffle_somas : bool, optional
        Whether shuffling was applied
    random_seed : int, optional
        Random seed used for shuffling

    Returns:
    --------
    Dict[str, str]
        Dictionary of output file paths
    """
    output_files = {}

    # Save bounding box information
    bbox_info = {
        "target_segments": target_segments,
        "x_min": bounding_box["x"][0],
        "x_max": bounding_box["x"][1],
        "y_min": bounding_box["y"][0],
        "y_max": bounding_box["y"][1],
        "n_filtered_somas": len(filtered_segment_ids),
        "filtered_segment_ids": filtered_segment_ids,
        "shuffled": shuffle_somas,
    }

    if "z" in bounding_box:
        bbox_info["z_min"] = bounding_box["z"][0]
        bbox_info["z_max"] = bounding_box["z"][1]

    if shuffle_somas:
        bbox_info["random_seed"] = random_seed

    bbox_path = os.path.join(
        output_dir, f"spatial_filter_bbox_info{shuffle_suffix}_{date_str}.txt"
    )
    with open(bbox_path, "w") as f:
        for key, value in bbox_info.items():
            if key == "filtered_segment_ids":
                f.write(f"{key}: {', '.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")
    output_files["bbox_info"] = bbox_path

    # Save filtered coordinates
    coords_df = pd.DataFrame(
        {
            "segment_id": filtered_segment_ids,
            "z_um": filtered_coordinates[:, 0],
            "y_um": filtered_coordinates[:, 1],
            "x_um": filtered_coordinates[:, 2],
        }
    )

    coords_path = os.path.join(
        output_dir, f"spatial_filter_coordinates{shuffle_suffix}_{date_str}.csv"
    )
    coords_df.to_csv(coords_path, index=False)
    output_files["coordinates"] = coords_path

    print(f"Analysis results saved:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")

    return output_files


def spatial_filter_analysis(
    target_segments: List[int],
    xy_margin: float = DEFAULT_XY_MARGIN,
    z_margin: Optional[float] = None,
    output_suffix: str = "",
    filter_to_specific_segments: Optional[List[int]] = None,
    shuffle_somas: bool = False,
    random_seed: Optional[int] = None,
    transparent_bg: bool = False,
) -> Dict[str, Any]:
    """
    Perform complete spatial filtering analysis.

    Parameters:
    -----------
    target_segments : List[int]
        List of segment IDs to use as reference points (typically 2)
    xy_margin : float, optional
        Margin in micrometers for x and y dimensions
    z_margin : float, optional
        Margin in micrometers for z dimension. If None, no z filtering applied.
    output_suffix : str, optional
        Suffix to add to output filenames
    filter_to_specific_segments : List[int], optional
        If provided, further filter results to only include these segment IDs.
        If None (default), returns ALL somas within the spatial bounding box.
    shuffle_somas : bool, optional
        Whether to randomly shuffle the order of filtered somas. Default False.
    random_seed : int, optional
        Random seed for reproducible shuffling. If None, uses random shuffling.
    transparent_bg : bool, optional
        Whether to make white heatmap cells transparent instead of white. Default False.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results and file paths
    """
    print(f"Starting spatial filter analysis...")
    print(f"Target segments: {target_segments}")
    print(f"XY margin: {xy_margin} μm")
    print(f"Z margin: {z_margin} μm" if z_margin else "Z filtering: disabled")

    # Load data
    barcodes, coordinates, segment_ids, _ = generate_barcode_array_with_coordinates()
    coordinates_physical = coordinates / VOXEL_FACTOR

    print(f"Loaded {len(segment_ids)} somas with {barcodes.shape[1]} channels")

    # Get target coordinates
    target_coords, target_indices = get_segment_coordinates(
        coordinates_physical, segment_ids, target_segments
    )

    print(f"Target segment coordinates:")
    for i, seg_id in enumerate(target_segments):
        coord = target_coords[i]
        print(
            f"  Segment {seg_id}: z={coord[0]:.1f}, y={coord[1]:.1f}, x={coord[2]:.1f} μm"
        )

    # Calculate bounding box
    bounding_box = calculate_bounding_box(target_coords, xy_margin, z_margin)

    print(f"Bounding box:")
    for dim, (min_val, max_val) in bounding_box.items():
        print(f"  {dim}: {min_val:.1f} to {max_val:.1f} μm")

    # Filter somas - by default returns ALL somas within the spatial bounding box
    filtered_coords, filtered_barcodes, filtered_segment_ids, filtered_indices = (
        filter_somas_in_bounding_box(
            coordinates_physical,
            barcodes,
            segment_ids,
            bounding_box,
            specific_segment_ids=filter_to_specific_segments,
        )
    )

    if filter_to_specific_segments is None:
        print(
            f"Filtered to {len(filtered_segment_ids)} somas within spatial bounding box"
        )
        print(
            f"  (Note: This includes ALL somas within the spatial bounds, not just target segments)"
        )
    else:
        print(
            f"Filtered to {len(filtered_segment_ids)} somas within spatial bounding box"
        )
        print(
            f"  (Further filtered to only include specific segment IDs: {filter_to_specific_segments})"
        )

    # Optionally shuffle the filtered somas
    if shuffle_somas:
        if random_seed is not None:
            np.random.seed(random_seed)

        n_filtered = len(filtered_segment_ids)
        shuffle_order = np.random.permutation(n_filtered)

        # Apply shuffle to all filtered data
        filtered_coords = filtered_coords[shuffle_order]
        filtered_barcodes = filtered_barcodes[shuffle_order]
        filtered_segment_ids = [filtered_segment_ids[i] for i in shuffle_order]

        print(f"Shuffled {n_filtered} somas (random_seed: {random_seed})")

    # Generate outputs
    date_str = datetime.now().strftime("%y%m%d")
    suffix = f"_{output_suffix}" if output_suffix else ""
    shuffle_suffix = "_shuffled" if shuffle_somas else ""
    title_suffix = f" (n={len(filtered_segment_ids)})"

    # Create multiple heatmap variants
    heatmap_outputs = {}

    # 1. Standard heatmap with labels and highlights
    heatmap_path = os.path.join(
        SPATIAL_FILTER_OUTPUT_DIR,
        f"spatial_filter_heatmap{suffix}{shuffle_suffix}_{date_str}.png",
    )
    fig = plot_filtered_heatmap(
        filtered_barcodes,
        filtered_segment_ids,
        target_segments,
        heatmap_path,
        title_suffix,
        show_labels=True,
        show_highlights=True,
        transparent_bg=transparent_bg,
    )
    heatmap_outputs["standard"] = heatmap_path

    # 2. Square dimensions heatmap with labels and highlights
    square_path = os.path.join(
        SPATIAL_FILTER_OUTPUT_DIR,
        f"spatial_filter_heatmap_square{suffix}{shuffle_suffix}_{date_str}.png",
    )
    plot_filtered_heatmap(
        filtered_barcodes,
        filtered_segment_ids,
        target_segments,
        square_path,
        title_suffix,
        auto_square=True,
        show_labels=True,
        show_highlights=True,
        transparent_bg=transparent_bg,
    )
    heatmap_outputs["square"] = square_path

    # 3. Clean heatmap - no labels, but with highlights
    clean_highlights_path = os.path.join(
        SPATIAL_FILTER_OUTPUT_DIR,
        f"spatial_filter_heatmap_clean_highlights{suffix}{shuffle_suffix}_{date_str}.png",
    )
    plot_filtered_heatmap(
        filtered_barcodes,
        filtered_segment_ids,
        target_segments,
        clean_highlights_path,
        "",
        auto_square=True,
        show_labels=False,
        show_highlights=True,
        transparent_bg=transparent_bg,
    )
    heatmap_outputs["clean_highlights"] = clean_highlights_path

    # 4. Minimal heatmap - no labels, no highlights
    minimal_path = os.path.join(
        SPATIAL_FILTER_OUTPUT_DIR,
        f"spatial_filter_heatmap_minimal{suffix}{shuffle_suffix}_{date_str}.png",
    )
    plot_filtered_heatmap(
        filtered_barcodes,
        filtered_segment_ids,
        target_segments,
        minimal_path,
        "",
        auto_square=True,
        show_labels=False,
        show_highlights=False,
        transparent_bg=transparent_bg,
    )
    heatmap_outputs["minimal"] = minimal_path

    # Save results
    output_files = save_analysis_results(
        bounding_box,
        filtered_coords,
        filtered_segment_ids,
        target_segments,
        SPATIAL_FILTER_OUTPUT_DIR,
        date_str,
        shuffle_suffix,
        shuffle_somas,
        random_seed,
    )
    output_files.update(heatmap_outputs)

    # Compile results
    results = {
        "bounding_box": bounding_box,
        "filtered_coordinates": filtered_coords,
        "filtered_segment_ids": filtered_segment_ids,
        "filtered_barcodes": filtered_barcodes,
        "target_segments": target_segments,
        "n_filtered": len(filtered_segment_ids),
        "output_files": output_files,
    }

    print(f"\nSpatial filter analysis completed successfully!")
    print(f"Results saved with date: {date_str}")
    print(f"Output directory: {SPATIAL_FILTER_OUTPUT_DIR}")
    print(f"Generated heatmap variants: {list(heatmap_outputs.keys())}")

    return results


def create_custom_heatmap(
    target_segments: List[int],
    xy_margin: float = DEFAULT_XY_MARGIN,
    z_margin: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    auto_square: bool = False,
    show_labels: bool = True,
    show_highlights: bool = True,
    output_name: str = "custom",
    shuffle_somas: bool = False,
    random_seed: Optional[int] = None,
    transparent_bg: bool = False,
) -> str:
    """
    Create a single custom heatmap with specified parameters.

    Parameters:
    -----------
    target_segments : List[int]
        List of segment IDs to use as reference points
    xy_margin : float, optional
        Margin in micrometers for x and y dimensions
    z_margin : float, optional
        Margin in micrometers for z dimension
    figsize : Tuple[float, float], optional
        Custom figure size (width, height)
    auto_square : bool, optional
        Use square dimensions based on data size
    show_labels : bool, optional
        Whether to show axis labels
    show_highlights : bool, optional
        Whether to highlight target segments
    output_name : str, optional
        Custom name for output file
    shuffle_somas : bool, optional
        Whether to randomly shuffle the order of filtered somas
    random_seed : int, optional
        Random seed for reproducible shuffling
    transparent_bg : bool, optional
        Whether to make white heatmap cells transparent instead of white. Default False.

    Returns:
    --------
    str
        Path to the created heatmap file
    """
    # Get data and filter
    barcodes, coordinates, segment_ids, _ = generate_barcode_array_with_coordinates()
    coordinates_physical = coordinates / VOXEL_FACTOR
    target_coords, _ = get_segment_coordinates(
        coordinates_physical, segment_ids, target_segments
    )
    bounding_box = calculate_bounding_box(target_coords, xy_margin, z_margin)
    filtered_coords, filtered_barcodes, filtered_segment_ids, _ = (
        filter_somas_in_bounding_box(
            coordinates_physical, barcodes, segment_ids, bounding_box
        )
    )

    # Optionally shuffle the filtered somas
    if shuffle_somas:
        if random_seed is not None:
            np.random.seed(random_seed)

        n_filtered = len(filtered_segment_ids)
        shuffle_order = np.random.permutation(n_filtered)

        # Apply shuffle to all filtered data
        filtered_coords = filtered_coords[shuffle_order]
        filtered_barcodes = filtered_barcodes[shuffle_order]
        filtered_segment_ids = [filtered_segment_ids[i] for i in shuffle_order]

    # Create output path
    date_str = datetime.now().strftime("%y%m%d")
    shuffle_suffix = "_shuffled" if shuffle_somas else ""
    output_path = os.path.join(
        SPATIAL_FILTER_OUTPUT_DIR,
        f"spatial_filter_heatmap_{output_name}{shuffle_suffix}_{date_str}.png",
    )

    # Create heatmap
    plot_filtered_heatmap(
        filtered_barcodes,
        filtered_segment_ids,
        target_segments,
        output_path,
        f" (n={len(filtered_segment_ids)})" if show_labels else "",
        figsize=figsize,
        show_labels=show_labels,
        show_highlights=show_highlights,
        auto_square=auto_square,
        transparent_bg=transparent_bg,
    )

    print(f"Custom heatmap saved to: {output_path}")
    return output_path


def main():
    """
    Main function to run spatial filtering analysis with example parameters.
    """
    # Example configuration - modify these as needed
    target_segments = [1488, 1219]  # Segment IDs to use as reference points
    xy_margin = 100.0  # Margin in micrometers for x and y
    z_margin = None  # Margin in micrometers for z (set to None to disable z filtering)

    # Run analysis - by default returns ALL somas within the spatial bounding box
    # To filter to specific segment IDs, add: filter_to_specific_segments=[1488, 1219, 1234]
    # To shuffle somas, add: shuffle_somas=True, random_seed=42
    # To make backgrounds transparent: transparent_bg=True
    results = spatial_filter_analysis(
        target_segments=target_segments,
        xy_margin=xy_margin,
        z_margin=z_margin,
        output_suffix="shuffled",
        # filter_to_specific_segments=None,  # Default: return ALL somas in spatial bounds
        shuffle_somas=True,  # Shuffle the found somas
        random_seed=24,  # For reproducible shuffling
        transparent_bg=True,  # Make white heatmap cells transparent
    )

    # Print summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Target segments: {results['target_segments']}")
    print(f"Bounding box dimensions:")
    for dim, (min_val, max_val) in results["bounding_box"].items():
        size = max_val - min_val
        print(f"  {dim.upper()}: {size:.1f} μm ({min_val:.1f} to {max_val:.1f})")
    print(f"Filtered somas: {results['n_filtered']}")
    print(f"Output files: {list(results['output_files'].keys())}")

    # Example: Create a custom heatmap with specific dimensions
    print(f"\n=== CUSTOM HEATMAP EXAMPLES ===")

    # # Example 1: Custom figure size with no labels and no highlights
    # custom_path1 = create_custom_heatmap(
    #     target_segments=[1488, 1219],
    #     xy_margin=100.0,
    #     figsize=(5, 8),  # Custom figure size
    #     show_labels=False,
    #     show_highlights=False,
    #     output_name="custom_5x8_clean",
    # )

    # # Example 2: Auto-square with highlights but no labels
    # custom_path2 = create_custom_heatmap(
    #     target_segments=[1488, 1219],
    #     xy_margin=75.0,
    #     auto_square=True,  # Auto-square based on data dimensions
    #     show_labels=False,
    #     show_highlights=True,
    #     output_name="custom_square_highlights_only",
    # )

    # # Example 3: Shuffled heatmap with custom settings
    # custom_path3 = create_custom_heatmap(
    #     target_segments=[1488, 1219],
    #     xy_margin=100.0,
    #     figsize=(4, 6),
    #     show_labels=True,
    #     show_highlights=True,
    #     shuffle_somas=True,  # Shuffle the somas
    #     random_seed=123,  # Reproducible shuffle
    #     output_name="custom_shuffled",
    # )


if __name__ == "__main__":
    main()
