#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def create_soma_barcode_heatmap(soma_barcodes, output_path=None):
    """
    Create a heatmap of soma barcodes clustered by minimum hamming distance.
    
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
    min_hamming_distances = []
    
    for i in range(len(bit_differences)):
        # Get all hamming distances for this soma
        distances = bit_differences[i,:]
        
        # Create a mask to exclude self-comparison (which would have distance 0)
        mask = np.ones(len(distances), dtype=bool)
        mask[i] = False
        
        # Find the minimum distance excluding self
        min_distance = np.min(distances[mask])
        min_hamming_distances.append(min_distance)
    
    # Convert to numpy array for easier analysis
    min_hamming_distances = np.array(min_hamming_distances)
    
    # Create an index array to sort somas by their minimum hamming distance
    sorted_indices = np.argsort(min_hamming_distances)
    
    # Reorder soma_barcodes based on the sorted indices
    sorted_barcodes = soma_barcodes[sorted_indices]
    sorted_min_distances = min_hamming_distances[sorted_indices]
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), 
                                   gridspec_kw={'width_ratios': [1, 20]})
    
    # Plot minimum hamming distances as a separate heatmap (left side)
    sns.heatmap(sorted_min_distances.reshape(-1, 1), 
                ax=ax1, cmap='viridis', cbar=True,
                yticklabels=False, xticklabels=['Min Dist'],
                cbar_kws={'label': 'Minimum Hamming Distance'})
    
    # Plot the main barcode heatmap (right side)
    sns.heatmap(sorted_barcodes, ax=ax2, cmap='binary',
                yticklabels=False, cbar=False)
    
    # Set x-axis labels for the channels
    ax2.set_xticks(np.arange(n_channels) + 0.5)
    ax2.set_xticklabels(range(n_channels))
    ax2.set_xlabel('Channel')
    
    # Add horizontal lines to separate somas with different minimum distances
    unique_distances = np.unique(sorted_min_distances)
    boundaries = []
    
    for dist in unique_distances:
        # Find where the minimum distance changes
        idx = np.where(sorted_min_distances == dist)[0]
        if len(idx) > 0:
            boundaries.append(idx[0])
    
    # Add lines at the boundaries (skip the first which is at 0)
    for boundary in boundaries[1:]:
        ax1.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
        ax2.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
    
    # Add annotations for each group
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i+1] if i < len(boundaries)-1 else len(sorted_min_distances)
        mid = (start + end) // 2
        
        # Add text annotation to the left heatmap
        ax1.text(-0.5, mid, f"{int(sorted_min_distances[start])}", 
                 verticalalignment='center', fontsize=10)
    
    plt.suptitle('Soma Barcodes Clustered by Minimum Hamming Distance', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig, (ax1, ax2)

def example_usage():
    """Example of how to use the function with randomly generated data."""
    # Generate random binary barcodes: 147 somas with 18-bit barcodes
    np.random.seed(42)  # For reproducibility
    example_barcodes = np.random.randint(0, 2, size=(147, 18))
    
    # Create and show the heatmap
    create_soma_barcode_heatmap(example_barcodes, 'soma_barcode_heatmap.png')
    plt.show()

if __name__ == "__main__":
    # This will run the example if the script is executed directly
    example_usage() 