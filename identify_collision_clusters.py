#!/usr/bin/env python3
"""
Script to identify clusters of segments that share identical barcode patterns,
focusing on barcode patterns with high collision counts.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from bimodal_barcode_analysis import load_barcode_data

def identify_collision_clusters(data_path='Data/neuron_barcodes_full_roi.npz', 
                               min_collisions=10):
    """
    Identify clusters of segments that share identical barcode patterns.
    
    Parameters:
    -----------
    data_path : str
        Path to the neuron barcode data
    min_collisions : int
        Minimum number of collisions to report (e.g., 10 means patterns shared by at least 11 cells)
    
    Returns:
    --------
    dict
        Dictionary mapping collision counts to lists of segment clusters
    """
    # Load data
    discrete, thresholded, expressions_per_object, total_cells, threshold = load_barcode_data(data_path)
    
    # Convert binary arrays to strings for hashability
    barcode_strings = [''.join(map(str, row)) for row in thresholded]
    
    # Create a mapping from barcode pattern to segment IDs
    pattern_to_segments = defaultdict(list)
    for i, pattern in enumerate(barcode_strings):
        pattern_to_segments[pattern].append(i)
    
    # Count occurrences of each barcode pattern
    barcode_counts = Counter(barcode_strings)
    
    # Find patterns with high collision counts
    high_collision_patterns = {pattern: segments for pattern, segments in pattern_to_segments.items() 
                               if len(segments) > min_collisions}
    
    # Group by number of collisions
    collision_clusters = defaultdict(list)
    for pattern, segments in high_collision_patterns.items():
        collisions = len(segments) - 1  # Number of collisions = cluster size - 1
        collision_clusters[collisions].append(segments)
    
    # Print summary
    print(f"Found {len(high_collision_patterns)} barcode patterns with > {min_collisions} collisions")
    print("\nCollision counts summary:")
    for collisions, clusters in sorted(collision_clusters.items(), reverse=True):
        total_segments = sum(len(cluster) for cluster in clusters)
        print(f"{collisions} collisions: {len(clusters)} patterns, affecting {total_segments} segments")
    
    # Detailed output for top collision clusters
    print("\nDetailed segment IDs for top collision clusters:")
    for collisions, clusters in sorted(collision_clusters.items(), reverse=True)[:5]:  # Top 5 collision counts
        print(f"\n{collisions} collisions (pattern shared by {collisions+1} cells):")
        for i, cluster in enumerate(clusters[:3]):  # Show first 3 clusters for each collision count
            # Truncate the list if it's very long
            if len(cluster) > 20:
                segment_str = ', '.join(map(str, cluster[:10])) + f", ... and {len(cluster)-10} more"
            else:
                segment_str = ', '.join(map(str, cluster))
            
            # Get the actual barcode pattern for this cluster
            pattern = barcode_strings[cluster[0]]
            bit_sum = sum(int(bit) for bit in pattern)  # Count '1's in the pattern
            
            print(f"  Cluster {i+1}: {segment_str}")
            print(f"    Pattern: {bit_sum} bits set out of 18 ({bit_sum/18:.1%})")
            
            # Show the actual binary pattern for the first few clusters
            if i < 1:  # Only for the first cluster
                binary_pattern = np.array([int(bit) for bit in pattern])
                channel_indices = np.where(binary_pattern == 1)[0]
                if len(channel_indices) > 0:
                    print(f"    Active channels: {channel_indices.tolist()}")
                else:
                    print(f"    Active channels: None (all zeros)")
    
    # Create a visualization of collision distribution
    plt.figure(figsize=(12, 6))
    
    all_collisions = [len(segments)-1 for segments in pattern_to_segments.values()]
    max_to_show = 50  # Limit x-axis for readability
    
    # Count how many patterns have each collision count
    collision_distribution = Counter(all_collisions)
    x = list(range(min(max_to_show+1, max(collision_distribution.keys())+1)))
    y = [collision_distribution.get(i, 0) for i in x]
    
    plt.bar(x, y, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Collisions')
    plt.ylabel('Number of Unique Patterns')
    plt.title('Distribution of Barcode Pattern Collision Counts')
    plt.grid(alpha=0.3)
    plt.savefig('collision_pattern_distribution.png', dpi=300, bbox_inches='tight')
    
    return collision_clusters

if __name__ == "__main__":
    # Run the analysis with default parameters
    collision_clusters = identify_collision_clusters(min_collisions=10)
    
    print("\nScript completed. See output above for segment IDs in high-collision clusters.")
    print("A visualization of collision distribution has been saved as 'collision_pattern_distribution.png'") 