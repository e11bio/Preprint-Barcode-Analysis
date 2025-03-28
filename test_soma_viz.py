#!/usr/bin/env python3
"""
Test script to demonstrate the soma barcode visualization with the enhanced clustering.

This script generates a test dataset of soma barcodes and visualizes them using
the enhanced clustering method that groups somas not only by their minimum hamming
distance but also by their relationships to each other.
"""

import numpy as np
import matplotlib.pyplot as plt
from soma_barcode_viz import create_soma_barcode_heatmap, analyze_min_hamming_distances

def main():
    """Run the test visualization."""
    # Generate a test dataset with some structure
    # Create 147 somas with 18-bit barcodes, but with some patterns
    np.random.seed(42)  # For reproducibility
    
    n_somas = 147
    n_channels = 18
    soma_barcodes = np.zeros((n_somas, n_channels), dtype=int)
    
    # Create some patterns to show the clustering effect
    
    # Group 1: Similar barcodes with minimum distance 1 to each other
    for i in range(20):
        # Start with a random barcode
        if i == 0:
            soma_barcodes[i, :] = np.random.randint(0, 2, n_channels)
        else:
            # Copy the previous barcode and flip one random bit
            soma_barcodes[i, :] = soma_barcodes[i-1, :].copy()
            flip_idx = np.random.randint(0, n_channels)
            soma_barcodes[i, flip_idx] = 1 - soma_barcodes[i, flip_idx]
    
    # Group 2: Similar barcodes with minimum distance 2 to each other
    base_barcode = np.random.randint(0, 2, n_channels)
    for i in range(20, 40):
        soma_barcodes[i, :] = base_barcode.copy()
        # Flip two random bits
        flip_indices = np.random.choice(n_channels, 2, replace=False)
        for idx in flip_indices:
            soma_barcodes[i, idx] = 1 - soma_barcodes[i, idx]
    
    # Group 3: Similar barcodes with minimum distance 3
    base_barcode = np.random.randint(0, 2, n_channels)
    for i in range(40, 60):
        soma_barcodes[i, :] = base_barcode.copy()
        # Flip three random bits
        flip_indices = np.random.choice(n_channels, 3, replace=False)
        for idx in flip_indices:
            soma_barcodes[i, idx] = 1 - soma_barcodes[i, idx]
    
    # Group 4: A cluster of somas with min distance 1 but forming a different cluster
    base_barcode = np.random.randint(0, 2, n_channels)
    for i in range(60, 80):
        if i == 60:
            soma_barcodes[i, :] = base_barcode
        else:
            # Copy the previous barcode and flip one random bit
            soma_barcodes[i, :] = soma_barcodes[i-1, :].copy()
            flip_idx = np.random.randint(0, n_channels)
            soma_barcodes[i, flip_idx] = 1 - soma_barcodes[i, flip_idx]
    
    # Fill the rest with random barcodes
    for i in range(80, n_somas):
        soma_barcodes[i, :] = np.random.randint(0, 2, n_channels)
    
    # Create the heatmap visualization
    print("Creating heatmap visualization...")
    fig, ax = create_soma_barcode_heatmap(soma_barcodes, "test_soma_barcode_heatmap.png")
    
    # Analyze the minimum hamming distances
    print("\nAnalyzing minimum hamming distances...")
    min_distances, hist_fig = analyze_min_hamming_distances(soma_barcodes)
    
    # Show the figures
    plt.show()

if __name__ == "__main__":
    main() 