import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def create_barcode_heatmap(soma_barcodes):
    """
    Create a heatmap and related visualizations to analyze barcode uniqueness.
    
    Parameters:
    -----------
    soma_barcodes : numpy.ndarray
        A binary matrix of shape (n_somas, n_barcodes) representing the barcode calls.
        Each row is a soma, each column is a barcode element (0 or 1).
    
    Returns:
    --------
    None
        This function creates and displays multiple visualizations.
    """
    # Calculate the Hamming distance matrix between all barcodes
    # This counts how many elements differ between each pair of barcodes
    hamming_distance_matrix = pairwise_distances(soma_barcodes, metric='hamming') * soma_barcodes.shape[1]

    # Create a figure with appropriate size
    plt.figure(figsize=(12, 10))

    # Create the heatmap
    # Lower values (darker colors) indicate more similar barcodes
    sns.heatmap(hamming_distance_matrix, 
                cmap='viridis_r',  # Reversed viridis (darker = more similar)
                xticklabels=False, 
                yticklabels=False,
                cbar_kws={'label': 'Hamming Distance'})

    plt.title('Barcode Similarity Heatmap\n(Lower values indicate greater similarity)')
    plt.xlabel('Soma Index')
    plt.ylabel('Soma Index')
    plt.tight_layout()
    plt.show()

    # We can also analyze the distribution of distances to understand uniqueness
    plt.figure(figsize=(10, 6))
    # Flatten the upper triangle of the distance matrix (excluding diagonal)
    distances = hamming_distance_matrix[np.triu_indices_from(hamming_distance_matrix, k=1)]
    sns.histplot(distances, bins=range(0, int(np.max(distances))+2), kde=True)
    plt.title('Distribution of Hamming Distances Between Barcodes')
    plt.xlabel('Hamming Distance (Number of Different Bits)')
    plt.ylabel('Frequency')
    plt.axvline(x=np.mean(distances), color='red', linestyle='--', 
               label=f'Mean: {np.mean(distances):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Identifying potentially problematic pairs with low Hamming distances
    threshold = 3  # Consider barcodes with distance <= 3 as potentially problematic
    problematic_indices = np.where(np.triu(hamming_distance_matrix, k=1) <= threshold)
    problematic_pairs = list(zip(problematic_indices[0], problematic_indices[1]))

    print(f"Number of barcode pairs with Hamming distance <= {threshold}: {len(problematic_pairs)}")

    if len(problematic_pairs) > 0:
        print("Potentially problematic barcode pairs (low Hamming distance):")
        for i, j in problematic_pairs:
            print(f"Soma {i} and Soma {j}: Distance = {hamming_distance_matrix[i, j]}")
            # Optionally, show the actual barcodes to see where they differ
            differ_indices = np.where(soma_barcodes[i] != soma_barcodes[j])[0]
            print(f"  Soma {i} barcode: {soma_barcodes[i]}")
            print(f"  Soma {j} barcode: {soma_barcodes[j]}")
            print(f"  Differing at indices: {differ_indices}")
            print()
    else:
        print(f"All barcode pairs have Hamming distance > {threshold}. Good barcode design!")

    # Create a clustered heatmap to visualize pattern similarity
    plt.figure(figsize=(12, 10))
    # Using clustermap to cluster similar barcodes together
    clustered_map = sns.clustermap(hamming_distance_matrix, 
                                  cmap='viridis_r',
                                  figsize=(12, 10),
                                  cbar_kws={'label': 'Hamming Distance'})
    plt.title('Clustered Barcode Similarity Heatmap', fontsize=15, pad=30)
    plt.tight_layout()
    plt.show()
    
    return hamming_distance_matrix

# Example usage:
# soma_barcodes = np.array(filtered_df[target_channels].values)
# create_barcode_heatmap(soma_barcodes) 