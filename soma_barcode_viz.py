#!/usr/bin/env python3
"""
Soma Barcode Visualization

This module provides functions to create a heatmap visualization of soma barcodes,
where the somas are clustered based on their minimum hamming distance to any other soma
and further organized by their minimum distance pairs.

Example usage in a Jupyter notebook:
```python
import soma_barcode_viz as viz

# Assuming soma_barcodes is a (147, 18) numpy array of 0s and 1s
fig, ax = viz.create_soma_barcode_heatmap(soma_barcodes)
```
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import networkx as nx

def create_soma_barcode_heatmap(soma_barcodes, output_path=None):
    """
    Create a heatmap of soma barcodes clustered by minimum hamming distance
    and further organized by their minimum distance pairs.
    
    Parameters:
    -----------
    soma_barcodes : numpy.ndarray
        A (n_somas, n_channels) array of 0s and 1s representing the presence/absence
        of protein epitopes for each soma.
    output_path : str, optional
        Path to save the figure. If None, the figure will be displayed but not saved.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    n_somas, n_channels = soma_barcodes.shape
    print(f"Creating heatmap for {n_somas} somas with {n_channels} channels")
    
    # Calculate hamming distances between all pairs of somas
    # This computes the number of bits that differ between each pair
    bit_differences = squareform(pdist(soma_barcodes, metric='hamming') * n_channels)
    
    # Calculate the minimum hamming distance for each soma (excluding self-comparison)
    # Also keep track of which soma corresponds to that minimum distance
    min_hamming_distances = []
    min_distance_partners = []
    
    for i in range(len(bit_differences)):
        # Get all hamming distances for this soma
        distances = bit_differences[i,:]
        
        # Create a mask to exclude self-comparison (which would have distance 0)
        mask = np.ones(len(distances), dtype=bool)
        mask[i] = False
        
        # Find the minimum distance excluding self
        min_distance = np.min(distances[mask])
        min_hamming_distances.append(min_distance)
        
        # Find the index of the soma with the minimum distance
        # If there are multiple somas with the same minimum distance, take the first one
        min_partner_idx = np.where((distances == min_distance) & mask)[0][0]
        min_distance_partners.append(min_partner_idx)
    
    # Convert to numpy arrays
    min_hamming_distances = np.array(min_hamming_distances)
    min_distance_partners = np.array(min_distance_partners)
    
    # Create a graph where nodes are somas and edges connect somas that are nearest neighbors
    G = nx.Graph()
    for i in range(len(min_hamming_distances)):
        G.add_node(i, min_distance=min_hamming_distances[i])
        # Add edge between soma i and its nearest neighbor
        G.add_edge(i, min_distance_partners[i], weight=min_hamming_distances[i])
    
    # Group somas by their minimum hamming distance
    distance_groups = {}
    for i, dist in enumerate(min_hamming_distances):
        if dist not in distance_groups:
            distance_groups[dist] = []
        distance_groups[dist].append(i)
    
    # Now, for each distance group, identify connected components (clusters of related somas)
    sorted_indices = []
    boundaries = []
    current_pos = 0
    
    # Process each minimum distance group in ascending order
    for dist in sorted(distance_groups.keys()):
        group_somas = distance_groups[dist]
        
        # Skip if the group is empty
        if not group_somas:
            continue
        
        # Create a subgraph for just this distance group
        subgraph = G.subgraph(group_somas)
        
        # Find connected components (clusters) within this distance group
        components = list(nx.connected_components(subgraph))
        
        # For each connected component, add its nodes to the sorted indices
        for component in components:
            component_list = sorted(list(component))  # Sort for reproducibility
            sorted_indices.extend(component_list)
            
            # Mark the starting boundary of this group
            if len(sorted_indices) > current_pos:
                boundaries.append(current_pos)
                current_pos = len(sorted_indices)
    
    # Reorder soma_barcodes based on the sorted indices
    sorted_barcodes = soma_barcodes[sorted_indices]
    sorted_min_distances = min_hamming_distances[sorted_indices]
    
    # Create figure and subplot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot the barcode heatmap
    sns.heatmap(sorted_barcodes, ax=ax, cmap='binary',
                yticklabels=False, cbar=False)
    
    # Set x-axis labels for the channels
    ax.set_xticks(np.arange(n_channels) + 0.5)
    ax.set_xticklabels(range(n_channels))
    ax.set_xlabel('Channel')
    
    # Add horizontal lines to separate different distance groups
    for boundary in boundaries[1:]:
        ax.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
    
    # Add annotations for each group
    unique_distances = np.unique(sorted_min_distances)
    distance_boundaries = []
    
    for dist in unique_distances:
        # Find where this distance group starts
        idx = np.where(sorted_min_distances == dist)[0]
        if len(idx) > 0:
            distance_boundaries.append(idx[0])
    
    # Add annotations for each distance group
    for i in range(len(distance_boundaries)):
        start = distance_boundaries[i]
        end = distance_boundaries[i+1] if i < len(distance_boundaries)-1 else len(sorted_min_distances)
        mid = (start + end) // 2
        
        # Add text annotation for the distance
        dist = sorted_min_distances[start]
        ax.text(-1.5, mid, f"Dist={int(dist)}", 
                verticalalignment='center', fontsize=10, rotation=90)
    
    plt.suptitle('Soma Barcodes Clustered by Minimum Hamming Distance and Related Pairs', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig, ax

def analyze_min_hamming_distances(soma_barcodes):
    """
    Analyze the distribution of minimum hamming distances for the given soma barcodes.
    
    Parameters:
    -----------
    soma_barcodes : numpy.ndarray
        A (n_somas, n_channels) array of 0s and 1s representing the presence/absence
        of protein epitopes for each soma.
        
    Returns:
    --------
    min_hamming_distances : numpy.ndarray
        Array of minimum hamming distances for each soma.
    fig : matplotlib.figure.Figure
        The histogram figure.
    """
    # Calculate hamming distances
    bit_differences = squareform(pdist(soma_barcodes, metric='hamming') * soma_barcodes.shape[1])
    min_hamming_distances = []
    
    for i in range(len(bit_differences)):
        distances = bit_differences[i,:]
        mask = np.ones(len(distances), dtype=bool)
        mask[i] = False
        min_distance = np.min(distances[mask])
        min_hamming_distances.append(min_distance)
    
    min_hamming_distances = np.array(min_hamming_distances)
    
    # Create a histogram
    fig = plt.figure(figsize=(10, 6))
    plt.hist(min_hamming_distances, 
             bins=range(int(min(min_hamming_distances)), int(max(min_hamming_distances))+2), 
             edgecolor='black', align='left')
    plt.xlabel('Minimum Hamming Distance')
    plt.ylabel('Number of Somas')
    plt.title('Distribution of Minimum Hamming Distances')
    plt.xticks(range(int(min(min_hamming_distances)), int(max(min_hamming_distances))+1))
    plt.grid(axis='y', alpha=0.75)
    
    # Print statistics
    print(f"Mean minimum hamming distance: {np.mean(min_hamming_distances):.2f}")
    print(f"Median minimum hamming distance: {np.median(min_hamming_distances)}")
    print(f"Min hamming distance: {np.min(min_hamming_distances)}")
    print(f"Max hamming distance: {np.max(min_hamming_distances)}")
    
    # Count number of somas for each minimum hamming distance
    unique_distances, counts = np.unique(min_hamming_distances, return_counts=True)
    print("\nNumber of somas per minimum hamming distance:")
    for dist, count in zip(unique_distances, counts):
        print(f"  Distance {int(dist)}: {count} somas")
    
    return min_hamming_distances, fig

def example_usage():
    """Show an example of how to use this module with random data."""
    # Generate random binary barcodes: 147 somas with 18-bit barcodes
    np.random.seed(42)  # For reproducibility
    example_barcodes = np.random.randint(0, 2, size=(147, 18))
    
    # Create and show the heatmap
    create_soma_barcode_heatmap(example_barcodes, 'soma_barcode_heatmap.png')
    analyze_min_hamming_distances(example_barcodes)
    plt.show()

if __name__ == "__main__":
    # This will run the example if the script is executed directly
    example_usage() 