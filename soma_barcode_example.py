#!/usr/bin/env python3
"""
Example script to demonstrate how to use the soma_barcode_viz module 
with your existing soma_barcodes data.

Usage in a Jupyter notebook:
```python
# Import this script
%run soma_barcode_example.py

# Run the visualization with your existing soma_barcodes variable
visualize_soma_barcodes(soma_barcodes)
```
"""

import numpy as np
import matplotlib.pyplot as plt
from soma_barcode_viz import create_soma_barcode_heatmap, analyze_min_hamming_distances

def visualize_soma_barcodes(soma_barcodes, save_path=None):
    """
    Visualize the soma barcodes with the heatmap clustering 
    and analyze the minimum hamming distances.
    
    Parameters:
    -----------
    soma_barcodes : numpy.ndarray
        A (n_somas, n_channels) array of 0s and 1s representing the presence/absence
        of protein epitopes for each soma.
    save_path : str, optional
        Path to save the heatmap figure. If None, the figure will not be saved.
    """
    # Check the input data
    if not isinstance(soma_barcodes, np.ndarray):
        raise TypeError("soma_barcodes must be a numpy array")
    
    if len(soma_barcodes.shape) != 2:
        raise ValueError(f"soma_barcodes must be a 2D array, got shape {soma_barcodes.shape}")
    
    n_somas, n_channels = soma_barcodes.shape
    print(f"Input data: {n_somas} somas with {n_channels} channels (protein epitopes)")
    
    # Ensure the array contains only 0s and 1s
    if not np.all(np.isin(soma_barcodes, [0, 1])):
        raise ValueError("soma_barcodes must contain only 0s and 1s")
    
    # Display a small sample of the data
    print("\nSample of first 5 soma barcodes:")
    print(soma_barcodes[:5])
    
    # Create the heatmap visualization
    print("\nCreating heatmap visualization...")
    fig_heatmap, _ = create_soma_barcode_heatmap(soma_barcodes, save_path)
    
    # Analyze the minimum hamming distances
    print("\nAnalyzing minimum hamming distances...")
    min_distances, fig_hist = analyze_min_hamming_distances(soma_barcodes)
    
    return fig_heatmap, fig_hist, min_distances

if __name__ == "__main__":
    # Generate a random example dataset
    print("Generating example data (147 somas with 18-bit barcodes)...")
    np.random.seed(42)  # For reproducibility
    example_barcodes = np.random.randint(0, 2, size=(147, 18))
    
    # Run the visualization
    visualize_soma_barcodes(example_barcodes, "example_soma_barcode_heatmap.png")
    
    # Show the figures
    plt.show() 