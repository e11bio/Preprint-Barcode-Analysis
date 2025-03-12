import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import statsmodels.stats.multitest
import os
import networkx as nx
from matplotlib.lines import Line2D
from bimodal_barcode_analysis import load_barcode_data
from collections import Counter

# Set publication-quality figure aesthetics
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def epitope_co_occurrence_analysis(thresholded, neuron_classifications, channel_names=None):
    """
    Analyze the co-occurrence of epitope expression in different neuron populations
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    channel_names : list or None
        List of channel names corresponding to columns in thresholded
    
    Returns:
    --------
    dict
        Dictionary containing co-occurrence matrices and visualization
    """
    # Split data by population
    expected_pop_mask = neuron_classifications['Population'] == 'Expected Population'
    overexp_mask = neuron_classifications['Population'] == 'Overexpressors'
    
    expected_pop_data = thresholded[expected_pop_mask]
    overexp_data = thresholded[overexp_mask]
    
    # Create co-occurrence matrices
    expected_pop_cooccur = np.dot(expected_pop_data.T, expected_pop_data) / len(expected_pop_data)
    overexp_cooccur = np.dot(overexp_data.T, overexp_data) / len(overexp_data)
    
    # Calculate differential co-occurrence
    diff_cooccur = overexp_cooccur - expected_pop_cooccur
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(thresholded.shape[1])]
    
    # Plot expected population co-occurrence
    sns.heatmap(expected_pop_cooccur, ax=axes[0], cmap="Blues", 
                xticklabels=channel_names, yticklabels=channel_names)
    axes[0].set_title("Expected Population Co-occurrence")
    
    # Plot overexpressors co-occurrence
    sns.heatmap(overexp_cooccur, ax=axes[1], cmap="Reds", 
                xticklabels=channel_names, yticklabels=channel_names)
    axes[1].set_title("Overexpressors Co-occurrence")
    
    # Plot differential co-occurrence
    sns.heatmap(diff_cooccur, ax=axes[2], cmap="coolwarm", 
                xticklabels=channel_names, yticklabels=channel_names,
                center=0)
    axes[2].set_title("Differential Co-occurrence (Overexp - Expected)")
    
    plt.tight_layout()
    
    return {
        "expected_pop_cooccur": expected_pop_cooccur,
        "overexp_cooccur": overexp_cooccur,
        "diff_cooccur": diff_cooccur,
        "figure": fig
    }

def hierarchical_clustering_analysis(thresholded, neuron_classifications, channel_names=None):
    """
    Perform hierarchical clustering analysis on neuron expression patterns
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    channel_names : list or None
        List of channel names corresponding to columns in thresholded
    
    Returns:
    --------
    dict
        Dictionary containing clustering results and visualization
    """
    # Get population data
    population_data = np.zeros(len(thresholded))
    population_data[neuron_classifications['Population'] == 'Overexpressors'] = 1
    
    # Calculate distance matrix
    dist_matrix = pdist(thresholded, metric='jaccard')
    
    # Perform hierarchical clustering
    Z = linkage(dist_matrix, method='ward')
    
    # Create visualization
    fig = plt.figure(figsize=(14, 8))
    
    # Plot dendrogram
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    dendrogram(
        Z,
        ax=ax1,
        leaf_rotation=90.,
        leaf_font_size=10.,
    )
    ax1.set_title('Hierarchical Clustering of Neurons by Epitope Expression')
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Distance')
    
    # Plot population distribution under dendrogram to see correspondence
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    idx = dendrogram(Z, no_plot=True)['leaves']
    ax2.imshow(population_data[idx].reshape(1, -1), aspect='auto', cmap='coolwarm')
    ax2.set_yticks([])
    ax2.set_title('Population Membership (Blue = Expected, Red = Overexpressors)')
    
    plt.tight_layout()
    
    return {
        "linkage": Z,
        "distance_matrix": dist_matrix,
        "figure": fig
    }

def infection_marker_correlation_analysis(discrete, thresholded, neuron_classifications):
    """
    Correlate channel expression with overall signal intensity (a proxy for viral load)
    
    Parameters:
    -----------
    discrete : numpy.ndarray
        Raw signal intensities for each neuron across channels
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    
    Returns:
    --------
    dict
        Dictionary containing correlation results and visualization
    """
    # Calculate proxy for viral load: sum of signal intensities across all channels
    viral_load_proxy = np.sum(discrete, axis=1)
    
    # Get channel count data
    channel_counts = np.sum(thresholded, axis=1)
    
    # Split by population
    expected_pop_mask = neuron_classifications['Population'] == 'Expected Population'
    overexp_mask = neuron_classifications['Population'] == 'Overexpressors'
    
    # Calculate correlations
    overall_corr, overall_pval = stats.pearsonr(viral_load_proxy, channel_counts)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data points, colored by population
    ax.scatter(viral_load_proxy[expected_pop_mask], channel_counts[expected_pop_mask], 
               alpha=0.7, label='Expected Population', c='blue')
    ax.scatter(viral_load_proxy[overexp_mask], channel_counts[overexp_mask], 
               alpha=0.7, label='Overexpressors', c='red')
    
    # Overall regression
    X = viral_load_proxy.reshape(-1, 1)
    model = LinearRegression().fit(X, channel_counts)
    x_range = np.linspace(np.min(viral_load_proxy), np.max(viral_load_proxy), 100)
    ax.plot(x_range, model.predict(x_range.reshape(-1, 1)), 'k--', 
            label=f'Overall (r={overall_corr:.2f}, p={overall_pval:.4f})')
    
    # Add regression lines for each population
    if np.sum(expected_pop_mask) > 2:
        exp_corr, exp_pval = stats.pearsonr(viral_load_proxy[expected_pop_mask], 
                                           channel_counts[expected_pop_mask])
        X_exp = viral_load_proxy[expected_pop_mask].reshape(-1, 1)
        model_exp = LinearRegression().fit(X_exp, channel_counts[expected_pop_mask])
        ax.plot(x_range, model_exp.predict(x_range.reshape(-1, 1)), 'b--', 
                label=f'Expected Pop. (r={exp_corr:.2f}, p={exp_pval:.4f})')
    
    if np.sum(overexp_mask) > 2:
        over_corr, over_pval = stats.pearsonr(viral_load_proxy[overexp_mask], 
                                             channel_counts[overexp_mask])
        X_over = viral_load_proxy[overexp_mask].reshape(-1, 1)
        model_over = LinearRegression().fit(X_over, channel_counts[overexp_mask])
        ax.plot(x_range, model_over.predict(x_range.reshape(-1, 1)), 'r--', 
                label=f'Overexpressors (r={over_corr:.2f}, p={over_pval:.4f})')
    
    ax.set_xlabel('Sum of Signal Intensities (proxy for viral load)')
    ax.set_ylabel('Number of Channels Expressed')
    ax.set_title('Correlation Between Viral Load and Channel Expression')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return {
        "overall_correlation": (overall_corr, overall_pval),
        "figure": fig
    }

def channel_preference_analysis(thresholded, neuron_classifications, channel_names=None):
    """
    Analyze channel preferences in different neuron populations
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    channel_names : list or None
        List of channel names corresponding to columns in thresholded
    
    Returns:
    --------
    dict
        Dictionary containing preference analysis results and visualization
    """
    # Split data by population
    expected_pop_mask = neuron_classifications['Population'] == 'Expected Population'
    overexp_mask = neuron_classifications['Population'] == 'Overexpressors'
    
    expected_pop_data = thresholded[expected_pop_mask]
    overexp_data = thresholded[overexp_mask]
    
    # Calculate expression frequency for each channel in each population
    expected_pop_freq = np.mean(expected_pop_data, axis=0)
    overexp_freq = np.mean(overexp_data, axis=0)
    
    # Calculate overall expression frequency
    overall_freq = np.mean(thresholded, axis=0)
    
    # Calculate preference scores (ratio of observed to expected expression)
    # A value > 1 indicates preference, < 1 indicates avoidance
    expected_pop_pref = expected_pop_freq / overall_freq
    overexp_pref = overexp_freq / overall_freq
    
    # Perform statistical testing
    p_values = []
    effect_sizes = []
    
    for channel in range(thresholded.shape[1]):
        # Contingency table for Fisher's exact test
        a = np.sum(expected_pop_data[:, channel])  # Expected pop, channel expressed
        b = np.sum(overexp_data[:, channel])      # Overexp, channel expressed
        c = len(expected_pop_data) - a             # Expected pop, channel not expressed
        d = len(overexp_data) - b                 # Overexp, channel not expressed
        
        contingency = np.array([[a, b], [c, d]])
        _, p_value = stats.fisher_exact(contingency)
        p_values.append(p_value)
        
        # Calculate effect size (odds ratio)
        odds_ratio = (a/c) / (b/d) if (c > 0 and d > 0) else np.nan
        effect_sizes.append(odds_ratio)
    
    # Multiple testing correction
    reject, p_adjusted, _, _ = statsmodels.stats.multitest.multipletests(
        p_values, alpha=0.05, method='fdr_bh')
    
    # Create visualization
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(thresholded.shape[1])]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    x = np.arange(len(channel_names))
    width = 0.35
    
    # Plot preference scores
    rects1 = ax1.bar(x - width/2, expected_pop_pref, width, label='Expected Population')
    rects2 = ax1.bar(x + width/2, overexp_pref, width, label='Overexpressors')
    
    # Add reference line at 1.0 (no preference)
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    ax1.set_ylabel('Preference Score')
    ax1.set_title('Channel Preference by Population')
    ax1.legend()
    
    # Plot p-values on log scale
    ax2.bar(x, -np.log10(p_adjusted), color='gray', alpha=0.7)
    ax2.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.7, 
                label='p=0.05 (FDR corrected)')
    
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_xlabel('Channel')
    ax2.set_title('Statistical Significance of Preference Difference')
    ax2.set_xticks(x)
    ax2.set_xticklabels(channel_names, rotation=90)
    ax2.legend()
    
    plt.tight_layout()
    
    return {
        "expected_pop_pref": expected_pop_pref,
        "overexp_pref": overexp_pref,
        "p_values": p_values,
        "p_adjusted": p_adjusted,
        "effect_sizes": effect_sizes,
        "figure": fig
    }

def dimensionality_reduction_analysis(thresholded, neuron_classifications):
    """
    Perform dimensionality reduction to visualize neuron populations
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    
    Returns:
    --------
    dict
        Dictionary containing dimensionality reduction results and visualization
    """
    # Get population data as colors
    colors = np.zeros(len(thresholded), dtype=int)
    colors[neuron_classifications['Population'] == 'Overexpressors'] = 1
    
    # Apply standardization
    scaled_data = StandardScaler().fit_transform(thresholded)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(scaled_data)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define color map for populations
    cmap = plt.cm.get_cmap('coolwarm', 2)
    
    # Plot t-SNE points
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, 
                        cmap=cmap, alpha=0.7, s=50, edgecolors='k')
    
    # Add legend
    legend1 = ax.legend(scatter.legend_elements()[0],
                       ['Expected Population', 'Overexpressors'],
                       title="Population")
    ax.add_artist(legend1)
    
    # Add channel count information as point size
    channel_counts = np.sum(thresholded, axis=1)
    
    # Create a second scatter plot with sizes based on channel counts
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], s=channel_counts*5, 
              facecolors='none', edgecolors='gray', alpha=0.3)
    
    ax.set_title('t-SNE Visualization of Neuron Populations')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    
    return {
        "tsne_result": tsne_result,
        "figure": fig
    }

def epitope_network_analysis(thresholded, neuron_classifications, channel_names=None):
    """
    Create network diagrams of epitope co-expression patterns
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    channel_names : list or None
        List of channel names corresponding to columns in thresholded
    
    Returns:
    --------
    dict
        Dictionary containing network analysis results and visualization
    """
    # Split data by population
    expected_pop_mask = neuron_classifications['Population'] == 'Expected Population'
    overexp_mask = neuron_classifications['Population'] == 'Overexpressors'
    
    expected_pop_data = thresholded[expected_pop_mask]
    overexp_data = thresholded[overexp_mask]
    
    # Create co-occurrence matrices
    expected_pop_cooccur = np.dot(expected_pop_data.T, expected_pop_data) / len(expected_pop_data)
    overexp_cooccur = np.dot(overexp_data.T, overexp_data) / len(overexp_data)
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(thresholded.shape[1])]
    
    # Create networks for each population
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Function to create network
    def create_network(cooccur_matrix, ax, title):
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(channel_names):
            G.add_node(name, frequency=np.mean(thresholded[:, i]))
        
        # Add edges with weights based on co-occurrence
        for i in range(cooccur_matrix.shape[0]):
            for j in range(i+1, cooccur_matrix.shape[1]):
                if cooccur_matrix[i, j] > 0.2:  # Threshold for visualization clarity
                    G.add_edge(channel_names[i], channel_names[j], 
                              weight=cooccur_matrix[i, j])
        
        # Node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node sizes based on frequency
        node_sizes = [G.nodes[node]['frequency'] * 1000 for node in G.nodes]
        
        # Edge widths based on co-occurrence strength
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                              node_color='skyblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                              edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        ax.set_title(title)
        ax.axis('off')
        
        return G
    
    # Create networks
    G1 = create_network(expected_pop_cooccur, ax1, "Expected Population Co-expression Network")
    G2 = create_network(overexp_cooccur, ax2, "Overexpressors Co-expression Network")
    
    # Add a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
              markersize=10, label='Epitope (size = frequency)'),
        Line2D([0], [0], color='gray', linewidth=1, label='Co-expression (width = strength)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
              ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return {
        "expected_pop_network": G1,
        "overexp_network": G2,
        "figure": fig
    }

def ngs_correlation_analysis(thresholded, neuron_classifications, ngs_data_path, channel_names=None):
    """
    Correlate epitope expression patterns with NGS sequencing data
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame
        DataFrame with neuron classification into populations
    ngs_data_path : str
        Path to NGS sequencing data CSV file
    channel_names : list or None
        List of channel names corresponding to columns in thresholded
    
    Returns:
    --------
    dict
        Dictionary containing correlation analysis results and visualization
    """
    # Load NGS data
    ngs_data = pd.read_csv(ngs_data_path)
    
    # Calculate expression frequency for each channel in each population
    expected_pop_mask = neuron_classifications['Population'] == 'Expected Population'
    overexp_mask = neuron_classifications['Population'] == 'Overexpressors'
    
    expected_pop_data = thresholded[expected_pop_mask]
    overexp_data = thresholded[overexp_mask]
    
    expected_pop_freq = np.mean(expected_pop_data, axis=0)
    overexp_freq = np.mean(overexp_data, axis=0)
    overall_freq = np.mean(thresholded, axis=0)
    
    # Extract relevant columns from NGS data
    viral_freq = ngs_data['Frequency (Virus)'].str.rstrip('%').astype(float)
    plasmid_freq = ngs_data['Frequency (Plasmid)'].str.rstrip('%').astype(float)
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(thresholded.shape[1])]
    
    # Create a DataFrame for correlation analysis
    corr_data = pd.DataFrame({
        'Channel': channel_names,
        'Overall_Freq': overall_freq,
        'Expected_Pop_Freq': expected_pop_freq,
        'Overexp_Freq': overexp_freq,
        'Viral_Freq': viral_freq,
        'Plasmid_Freq': plasmid_freq
    })
    
    # Calculate correlations
    viral_overall_corr = np.corrcoef(overall_freq, viral_freq)[0, 1]
    viral_expected_corr = np.corrcoef(expected_pop_freq, viral_freq)[0, 1]
    viral_overexp_corr = np.corrcoef(overexp_freq, viral_freq)[0, 1]
    
    plasmid_overall_corr = np.corrcoef(overall_freq, plasmid_freq)[0, 1]
    plasmid_expected_corr = np.corrcoef(expected_pop_freq, plasmid_freq)[0, 1]
    plasmid_overexp_corr = np.corrcoef(overexp_freq, plasmid_freq)[0, 1]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot viral frequency correlations
    axes[0, 0].scatter(viral_freq, overall_freq, alpha=0.7)
    axes[0, 0].set_title(f'Viral Freq. vs Overall Freq. (r={viral_overall_corr:.2f})')
    axes[0, 0].set_xlabel('Viral Frequency (%)')
    axes[0, 0].set_ylabel('Overall Expression Frequency')
    
    axes[0, 1].scatter(viral_freq, expected_pop_freq, alpha=0.7)
    axes[0, 1].set_title(f'Viral Freq. vs Expected Pop. Freq. (r={viral_expected_corr:.2f})')
    axes[0, 1].set_xlabel('Viral Frequency (%)')
    axes[0, 1].set_ylabel('Expected Pop. Expression Frequency')
    
    axes[0, 2].scatter(viral_freq, overexp_freq, alpha=0.7)
    axes[0, 2].set_title(f'Viral Freq. vs Overexp. Freq. (r={viral_overexp_corr:.2f})')
    axes[0, 2].set_xlabel('Viral Frequency (%)')
    axes[0, 2].set_ylabel('Overexpressor Expression Frequency')
    
    # Plot plasmid frequency correlations
    axes[1, 0].scatter(plasmid_freq, overall_freq, alpha=0.7)
    axes[1, 0].set_title(f'Plasmid Freq. vs Overall Freq. (r={plasmid_overall_corr:.2f})')
    axes[1, 0].set_xlabel('Plasmid Frequency (%)')
    axes[1, 0].set_ylabel('Overall Expression Frequency')
    
    axes[1, 1].scatter(plasmid_freq, expected_pop_freq, alpha=0.7)
    axes[1, 1].set_title(f'Plasmid Freq. vs Expected Pop. Freq. (r={plasmid_expected_corr:.2f})')
    axes[1, 1].set_xlabel('Plasmid Frequency (%)')
    axes[1, 1].set_ylabel('Expected Pop. Expression Frequency')
    
    axes[1, 2].scatter(plasmid_freq, overexp_freq, alpha=0.7)
    axes[1, 2].set_title(f'Plasmid Freq. vs Overexp. Freq. (r={plasmid_overexp_corr:.2f})')
    axes[1, 2].set_xlabel('Plasmid Frequency (%)')
    axes[1, 2].set_ylabel('Overexpressor Expression Frequency')
    
    # Add channel labels to points
    for i, channel in enumerate(channel_names):
        for ax in axes.flatten():
            ax.annotate(channel, (ax.get_xlim()[0] + 0.1, overall_freq[i]), 
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    return {
        "correlation_data": corr_data,
        "viral_correlations": {
            "overall": viral_overall_corr,
            "expected_pop": viral_expected_corr,
            "overexpressors": viral_overexp_corr
        },
        "plasmid_correlations": {
            "overall": plasmid_overall_corr,
            "expected_pop": plasmid_expected_corr,
            "overexpressors": plasmid_overexp_corr
        },
        "figure": fig
    }

def hamming_distance_analysis(thresholded, neuron_classifications=None):
    """
    Analyze the Hamming distances between barcodes, focusing on each segment's closest match
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    neuron_classifications : pandas.DataFrame or None
        DataFrame with neuron classification into populations
    
    Returns:
    --------
    dict
        Dictionary containing Hamming distance analysis results and visualizations
    """
    # Calculate the full pairwise Hamming distance matrix
    hamming_distances = squareform(pdist(thresholded, metric='hamming'))
    
    # Set diagonal to infinity to exclude self-comparisons when finding minimum
    np.fill_diagonal(hamming_distances, np.inf)
    
    # Find closest match for each segment
    closest_match_distances = np.min(hamming_distances, axis=1)
    
    # Convert from Hamming distance (ratio) to absolute bit differences
    # Hamming distance is a ratio, so multiply by the number of bits/channels
    num_channels = thresholded.shape[1]
    closest_match_bit_diffs = closest_match_distances * num_channels
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot histogram of bit differences to closest match
    bins = np.arange(0, int(np.max(closest_match_bit_diffs) + 2)) - 0.5
    ax1.hist(closest_match_bit_diffs, bins=bins, alpha=0.7, edgecolor='black', color='royalblue')
    ax1.set_xlabel('Bit Differences to Closest Match')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Hamming Distances to Closest Matching Segment')
    ax1.grid(alpha=0.3)
    
    # Add statistics to the plot
    mean_diff = np.mean(closest_match_bit_diffs)
    median_diff = np.median(closest_match_bit_diffs)
    min_diff = np.min(closest_match_bit_diffs)
    
    stats_text = f"Mean: {mean_diff:.2f}\nMedian: {median_diff:.2f}\nMin: {min_diff:.2f}"
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Calculate percentiles for closest-match distances
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    p_values = np.percentile(closest_match_bit_diffs, percentiles)
    
    # Plot the CDF of the closest match distances
    sorted_diffs = np.sort(closest_match_bit_diffs)
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    
    ax2.plot(sorted_diffs, cumulative, '-', lw=2)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel('Bit Differences to Closest Match')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of Hamming Distances')
    
    # Add percentile information to the CDF plot
    percentile_df = pd.DataFrame({
        'Percentile': percentiles,
        'Bit Diff': p_values
    })
    
    percentile_text = "Percentiles:\n" + "\n".join([f"{p}%: {v:.1f}" for p, v in zip(percentiles, p_values)])
    ax2.text(0.05, 0.95, percentile_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # If neuron classifications are provided, also analyze distances by population
    pop_analysis = None
    if neuron_classifications is not None:
        pop_fig, pop_ax = plt.subplots(figsize=(10, 6))
        
        # Get masks for each population
        expected_pop_mask = neuron_classifications['Population'] == 'Expected Population'
        overexp_mask = neuron_classifications['Population'] == 'Overexpressors'
        
        # Get distances for each population
        expected_distances = closest_match_bit_diffs[expected_pop_mask]
        overexp_distances = closest_match_bit_diffs[overexp_mask]
        
        # Plot histogram of distances for each population
        bins = np.arange(0, int(np.max(closest_match_bit_diffs) + 2)) - 0.5
        pop_ax.hist([expected_distances, overexp_distances], bins=bins, alpha=0.7, 
                    edgecolor='black', color=['blue', 'red'], label=['Expected Population', 'Overexpressors'])
        
        pop_ax.set_xlabel('Bit Differences to Closest Match')
        pop_ax.set_ylabel('Frequency')
        pop_ax.set_title('Hamming Distances to Closest Match by Population')
        pop_ax.grid(alpha=0.3)
        pop_ax.legend()
        
        # Run statistical test to compare populations
        try:
            t_stat, p_val = stats.ttest_ind(expected_distances, overexp_distances, equal_var=False)
            mann_whitney_stat, mann_whitney_p = stats.mannwhitneyu(expected_distances, overexp_distances)
            
            # Add statistics to the plot
            exp_mean = np.mean(expected_distances)
            overexp_mean = np.mean(overexp_distances)
            stats_text = f"Expected Pop Mean: {exp_mean:.2f}\nOverexpressors Mean: {overexp_mean:.2f}\n"
            stats_text += f"t-test p-value: {p_val:.4f}\nMann-Whitney p-value: {mann_whitney_p:.4f}"
            
            pop_ax.text(0.95, 0.95, stats_text, transform=pop_ax.transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            pop_analysis = {
                "expected_pop_mean": exp_mean,
                "overexp_mean": overexp_mean,
                "t_test": (t_stat, p_val),
                "mann_whitney": (mann_whitney_stat, mann_whitney_p),
                "figure": pop_fig
            }
        except:
            # If there's an error in the statistical test (e.g., not enough data)
            pop_analysis = {
                "error": "Could not perform statistical test - insufficient data",
                "figure": pop_fig
            }
    
    return {
        "hamming_distances": hamming_distances,
        "closest_match_distances": closest_match_bit_diffs,
        "mean_distance": mean_diff,
        "median_distance": median_diff,
        "min_distance": min_diff,
        "percentiles": percentile_df,
        "figure": fig,
        "population_analysis": pop_analysis
    }

def barcode_collision_analysis(thresholded):
    """
    Analyze barcode collisions - how many cells have unique barcodes, 
    how many share with one other cell, etc.
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    
    Returns:
    --------
    dict
        Dictionary containing collision analysis results and visualization
    """
    # Convert binary arrays to strings for hashability
    barcode_strings = [''.join(map(str, row)) for row in thresholded]
    
    # Count occurrences of each barcode
    barcode_counts = Counter(barcode_strings)
    
    # Count how many barcodes appear exactly once, twice, etc.
    collision_counts = Counter(barcode_counts.values())
    
    # Calculate the total number of unique barcode patterns
    unique_patterns = len(barcode_counts)
    
    # Calculate how many cells have 0, 1, 2, etc. collisions
    cells_with_n_collisions = {}
    for count_value, frequency in collision_counts.items():
        cells_with_n_collisions[count_value-1] = count_value * frequency
    
    # Create a DataFrame for easier plotting and analysis
    collision_df = pd.DataFrame({
        'Collisions': [k for k in sorted(cells_with_n_collisions.keys())],
        'Number of Cells': [cells_with_n_collisions[k] for k in sorted(cells_with_n_collisions.keys())]
    })
    
    # Calculate percentages
    total_cells = thresholded.shape[0]
    collision_df['Percentage'] = (collision_df['Number of Cells'] / total_cells) * 100
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of collision counts
    ax1.bar(collision_df['Collisions'], collision_df['Number of Cells'], 
            alpha=0.7, edgecolor='black', color='forestgreen')
    ax1.set_xlabel('Number of Collisions')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Barcode Collision Distribution')
    ax1.set_xticks(collision_df['Collisions'])
    ax1.grid(alpha=0.3)
    
    # Add counts to each bar
    for i, v in enumerate(collision_df['Number of Cells']):
        ax1.text(collision_df['Collisions'].iloc[i], v + 5, str(v), 
                ha='center', va='bottom')
    
    # Pie chart of collision percentages
    ax2.pie(collision_df['Number of Cells'], labels=[f"{c} collisions" for c in collision_df['Collisions']], 
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title('Percentage of Cells by Collision Count')
    
    plt.tight_layout()
    
    # Create a summary statistics dictionary
    summary_stats = {
        "total_cells": total_cells,
        "unique_barcode_patterns": unique_patterns,
        "max_collisions": max(cells_with_n_collisions.keys()) if cells_with_n_collisions else 0,
        "unique_cells_percentage": collision_df.loc[collision_df['Collisions'] == 0, 'Percentage'].iloc[0] if 0 in collision_df['Collisions'].values else 0,
    }
    
    # Generate a markdown table for the report
    table_rows = [f"| {row['Collisions']} | {row['Number of Cells']} | {row['Percentage']:.2f}% |" 
                  for _, row in collision_df.iterrows()]
    markdown_table = "| Collisions | Number of Cells | Percentage |\n"
    markdown_table += "|:----------:|:-------------:|:----------:|\n"
    markdown_table += "\n".join(table_rows)
    
    return {
        "collision_df": collision_df,
        "cells_with_n_collisions": cells_with_n_collisions,
        "unique_patterns": unique_patterns,
        "summary_stats": summary_stats,
        "markdown_table": markdown_table,
        "figure": fig
    }

def run_comprehensive_analysis(data_path='Data/neuron_barcodes_full_roi.npz', 
                             ngs_data_path='Data/221208-pool-seq-clean.csv',
                             output_dir='advanced_analysis_results'):
    """
    Run all comprehensive analyses and generate publication-quality figures
    
    Parameters:
    -----------
    data_path : str
        Path to the neuron barcode data
    ngs_data_path : str
        Path to NGS sequencing data
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    dict
        Dictionary containing all analysis results
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    discrete, thresholded, expressions_per_object, total_cells, threshold = load_barcode_data(data_path)
    
    # Load existing classifications
    neuron_classifications = pd.read_csv('bimodal_analysis_results/neuron_classifications.csv')
    
    # Define channel names
    channel_names = [
        'E2-barcode-R1',
        'S1-barcode-R1',
        'ALFA-barcode-R1',
        'Ty1-barcode-R2',
        'HA-barcode-R3',
        'T7-barcode-R6',
        'VSVG-barcode-R6',
        'AU5-barcode-R8',
        'NWS-barcode-R9',
        'SunTag-barcode-R9',
        'ETAG-barcode-R9',
        'SPOT-barcode-R10',
        'MoonTag-barcode-R10',
        'HSV Tag-barcode-R10',
        'Protein C-barcode-R11',
        'Tag100-barcode-R11',
        'c-Myc-barcode-R11',
        'OLLAS-barcode-R12'
    ]
    
    # Run all analyses
    results = {}
    
    # 1. Co-occurrence Analysis
    print("Running co-occurrence analysis...")
    results['co_occurrence'] = epitope_co_occurrence_analysis(
        thresholded, neuron_classifications, channel_names)
    results['co_occurrence']['figure'].savefig(
        f'{output_dir}/epitope_co_occurrence.png', dpi=300, bbox_inches='tight')
    
    # 3. Infection Marker Correlation Analysis
    print("Running infection marker correlation analysis...")
    results['infection_correlation'] = infection_marker_correlation_analysis(
        discrete, thresholded, neuron_classifications)
    results['infection_correlation']['figure'].savefig(
        f'{output_dir}/infection_correlation.png', dpi=300, bbox_inches='tight')
    
    # 7. NGS Correlation Analysis
    print("Running NGS correlation analysis...")
    results['ngs_correlation'] = ngs_correlation_analysis(
        thresholded, neuron_classifications, ngs_data_path, channel_names)
    results['ngs_correlation']['figure'].savefig(
        f'{output_dir}/ngs_correlation.png', dpi=300, bbox_inches='tight')
    
    # 8. Hamming Distance Analysis
    print("Running Hamming distance analysis...")
    results['hamming_distance'] = hamming_distance_analysis(
        thresholded, neuron_classifications)
    results['hamming_distance']['figure'].savefig(
        f'{output_dir}/hamming_distance_analysis.png', dpi=300, bbox_inches='tight')
    
    if results['hamming_distance']['population_analysis'] is not None:
        results['hamming_distance']['population_analysis']['figure'].savefig(
            f'{output_dir}/hamming_distance_by_population.png', dpi=300, bbox_inches='tight')
    
    # 9. Barcode Collision Analysis
    print("Running barcode collision analysis...")
    results['barcode_collision'] = barcode_collision_analysis(thresholded)
    results['barcode_collision']['figure'].savefig(
        f'{output_dir}/barcode_collision_analysis.png', dpi=300, bbox_inches='tight')
    
    # Generate summary CSV file
    summary_data = {
        'Analysis': [
            'Co-occurrence Analysis',
            'Infection Marker Correlation',
            'NGS Correlation Analysis',
            'Hamming Distance Analysis',
            'Barcode Collision Analysis'
        ],
        'Key Findings': [
            f"Differential co-occurrence strength: {np.max(results['co_occurrence']['diff_cooccur']):.2f}",
            f"Overall correlation with viral load: {results['infection_correlation']['overall_correlation'][0]:.2f} (p={results['infection_correlation']['overall_correlation'][1]:.4f})",
            f"Viral freq. correlation with overexpressors: {results['ngs_correlation']['viral_correlations']['overexpressors']:.2f}",
            f"Mean Hamming distance to closest match: {results['hamming_distance']['mean_distance']:.2f} bits",
            f"Unique barcode patterns: {results['barcode_collision']['unique_patterns']} out of {results['barcode_collision']['summary_stats']['total_cells']} cells"
        ]
    }
    
    pd.DataFrame(summary_data).to_csv(f'{output_dir}/analysis_summary.csv', index=False)
    
    # Save a comprehensive report
    with open(f'{output_dir}/comprehensive_analysis_report.md', 'w') as f:
        f.write("# Comprehensive Analysis of AAV Infection Dynamics and Neuron Populations\n\n")
        
        f.write("## 1. Epitope Co-occurrence Analysis\n")
        f.write("This analysis reveals which epitopes tend to be expressed together in each population.\n")
        f.write("The heatmaps show co-expression frequencies, with the differential map highlighting\n")
        f.write("epitope combinations that are enriched in the Overexpressors population.\n\n")
        
        f.write("## 2. Infection Marker Correlation Analysis\n")
        f.write("This analysis examines the relationship between viral load (as measured by total signal intensity)\n")
        f.write("and the number of channels expressed, providing insights into infection dynamics in each population.\n\n")
        
        f.write("## 3. NGS Correlation Analysis\n")
        f.write("This analysis correlates epitope expression frequencies with viral and plasmid frequencies from NGS data,\n")
        f.write("investigating how the input viral pool composition influences expression patterns in each population.\n\n")
        
        f.write("## 4. Hamming Distance Analysis\n")
        f.write("This analysis examines the Hamming distances between barcodes, focusing on each neuron's closest match.\n")
        f.write(f"The mean Hamming distance to the closest match is {results['hamming_distance']['mean_distance']:.2f} bits.\n")
        f.write("The distribution of these distances provides insights into the uniqueness and distinguishability of the barcodes.\n\n")
        
        f.write("## 5. Barcode Collision Analysis\n")
        f.write("This analysis quantifies how many cells have unique barcodes versus how many share the same barcode pattern with other cells.\n")
        f.write(f"Out of {results['barcode_collision']['summary_stats']['total_cells']} total cells, there are {results['barcode_collision']['unique_patterns']} unique barcode patterns.\n")
        f.write("### Collision Distribution Table\n\n")
        f.write(results['barcode_collision']['markdown_table'])
        f.write("\n\n")
    
    print(f"All analyses complete. Results saved to {output_dir}/")
    return results 