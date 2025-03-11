import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Set publication-quality figure aesthetics
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_barcode_data(file_path='neuron_barcodes_fixed_keys.npz'):
    """Load barcode data from NPZ file"""
    data = np.load(file_path, allow_pickle=True)
    barcode_data = data['arr_0'].item()
    
    # Extract discrete values (signal intensities) for each object across channels
    discrete = np.array([obj_dict['discrete'] for obj_dict in barcode_data.values()])
    
    # Calculate the threshold for each channel (mean of discrete values)
    threshold = np.mean(discrete, axis=0)
    
    # Create binary matrix where 1 = expression above threshold, 0 = expression below threshold
    thresholded = (discrete > threshold).astype(int)
    
    # Count number of channels expressed for each neuron
    expressions_per_object = np.nansum(thresholded, axis=1)
    
    # Total number of cells/objects analyzed
    total_cells = thresholded.shape[0]
    
    return discrete, thresholded, expressions_per_object, total_cells, threshold

def basic_stats(expressions_per_object):
    """Calculate basic statistical measures of the distribution"""
    mean = np.mean(expressions_per_object)
    median = np.median(expressions_per_object)
    # mode = stats.mode(expressions_per_object).mode[0]
    variance = np.var(expressions_per_object)
    std_dev = np.std(expressions_per_object)
    
    # Calculate dispersion index (variance/mean ratio)
    # For a Poisson distribution, this should be close to 1
    dispersion_index = variance / mean
    
    stats_dict = {
        "Mean": mean,
        "Median": median,
        # "Mode": mode,
        "Variance": variance,
        "Standard Deviation": std_dev,
        "Dispersion Index": dispersion_index
    }
    
    return stats_dict

def fit_poisson(data, cutoff=None):
    """Fit a Poisson distribution to the data, optionally using a cutoff value"""
    if cutoff is not None:
        filtered_data = data[data <= cutoff]
    else:
        filtered_data = data
    
    # Calculate lambda (mean) for the Poisson distribution
    lambda_est = np.mean(filtered_data)
    
    # Generate Poisson probabilities
    x_range = np.arange(0, int(np.max(data)) + 1)
    poisson_pmf = stats.poisson.pmf(x_range, lambda_est)
    
    # Scale PMF to match histogram frequency
    expected_counts = poisson_pmf * len(filtered_data)
    
    # Calculate chi-square goodness of fit
    observed, _ = np.histogram(filtered_data, bins=np.arange(-0.5, np.max(filtered_data)+1.5))
    expected = expected_counts[:len(observed)]
    
    # Combine bins with expected counts < 5 (standard practice for chi-square)
    valid_bins = expected >= 5
    if not all(valid_bins):
        combined_observed = []
        combined_expected = []
        current_obs = 0
        current_exp = 0
        
        for i in range(len(observed)):
            current_obs += observed[i]
            current_exp += expected[i]
            
            if current_exp >= 5 or i == len(observed) - 1:
                combined_observed.append(current_obs)
                combined_expected.append(current_exp)
                current_obs = 0
                current_exp = 0
        
        chi2_stat = np.sum(((np.array(combined_observed) - np.array(combined_expected))**2) / np.array(combined_expected))
        dof = len(combined_observed) - 1 - 1  # bins - 1 - parameters estimated (1 for lambda)
    else:
        chi2_stat = np.sum(((observed - expected)**2) / expected)
        dof = len(observed) - 1 - 1

    p_value = stats.chi2.sf(chi2_stat, dof)
    
    return {
        "lambda": lambda_est,
        "x_range": x_range,
        "pmf": poisson_pmf,
        "expected_counts": expected_counts,
        "chi2_stat": chi2_stat,
        "dof": dof,
        "p_value": p_value
    }

def fit_mixture_model(expressions_per_object, n_components=2):
    """Fit a Gaussian Mixture Model to identify the two populations"""
    # Reshape the data for sklearn
    X = expressions_per_object.reshape(-1, 1)
    
    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Get the parameters
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    # Sort components by their means (to have consistent ordering)
    sort_idx = np.argsort(means)
    means = means[sort_idx]
    variances = variances[sort_idx]
    weights = weights[sort_idx]
    
    # Calculate BIC and AIC for model evaluation
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    
    # Predict the component for each data point
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    
    # Create a x-range for plotting the GMM
    x = np.linspace(0, np.max(expressions_per_object), 1000)
    
    # Calculate the density of each component
    densities = []
    for i in range(n_components):
        component_idx = sort_idx[i]
        density = weights[i] * stats.norm.pdf(x, means[i], np.sqrt(variances[i]))
        densities.append(density)
    
    # Total density
    total_density = np.sum(densities, axis=0)
    
    return {
        "gmm": gmm,
        "means": means,
        "variances": variances,
        "weights": weights,
        "bic": bic,
        "aic": aic,
        "labels": labels,
        "probabilities": probabilities,
        "x_range": x,
        "densities": densities,
        "total_density": total_density
    }

def plot_basic_distribution(expressions_per_object, total_cells, save_path=None):
    """Create a basic histogram of the distribution"""
    plt.figure(figsize=(10, 7))
    
    bins = np.arange(-0.5, np.max(expressions_per_object) + 1.5)
    plt.hist(
        expressions_per_object,
        bins=bins,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    
    plt.xlabel("Number of Channels Expressed")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Barcode Expressions\n({total_cells} neurons)")
    
    plt.xticks(np.arange(0, np.max(expressions_per_object) + 1))
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_poisson_fit(expressions_per_object, cutoff=15, save_path=None):
    """Plot histogram with Poisson fit using a cutoff value"""
    # Create a filtered dataset using the cutoff
    filtered_indices = expressions_per_object <= cutoff
    filtered_expressions = expressions_per_object[filtered_indices]
    
    # Get Poisson fit data
    poisson_fit = fit_poisson(filtered_expressions)
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    
    # Plot histogram of full observed data
    plt.hist(
        expressions_per_object,
        bins=np.arange(-0.5, np.max(expressions_per_object) + 1.5),
        alpha=0.4,
        color='darkblue',
        label='All observed data',
        edgecolor='black'
    )
    
    # Plot histogram of filtered data
    plt.hist(
        filtered_expressions,
        bins=np.arange(-0.5, cutoff + 1.5),
        alpha=0.7,
        color='royalblue',
        label=f'Filtered data (0-{cutoff} channels)',
        edgecolor='black'
    )
    
    # Plot Poisson PMF for the range
    plt.plot(
        poisson_fit["x_range"][:cutoff+1],
        poisson_fit["expected_counts"][:cutoff+1],
        'r-',
        linewidth=2.5,
        label=f'Poisson fit (λ={poisson_fit["lambda"]:.2f})'
    )
    
    # Add points at the actual x-values for clarity
    plt.plot(
        np.arange(cutoff+1),
        poisson_fit["expected_counts"][:cutoff+1],
        'ro',
        markersize=5
    )
    
    # Add statistical information to the plot
    stats_text = f"Chi-square = {poisson_fit['chi2_stat']:.2f}, df = {poisson_fit['dof']}, p = {poisson_fit['p_value']:.4f}\n"
    stats_text += f"Mean = {poisson_fit['lambda']:.2f}, Variance = {np.var(filtered_expressions):.2f}\n"
    stats_text += f"Dispersion index = {np.var(filtered_expressions)/poisson_fit['lambda']:.2f} (=1 for ideal Poisson)"
    
    plt.annotate(
        stats_text,
        xy=(0.68, 0.82),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        va='top',
        fontsize=9
    )
    
    plt.xlabel('Number of Channels Expressed')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Barcode Expressions with Poisson Fit (Range: 0-{cutoff} channels)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(0, np.max(expressions_per_object) + 1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return poisson_fit

def plot_gaussian_mixture(expressions_per_object, n_components=2, save_path=None):
    """Plot the GMM fit on the histogram"""
    # Fit the mixture model
    gmm_results = fit_mixture_model(expressions_per_object, n_components)
    
    # Plot the histogram and the fitted GMM
    plt.figure(figsize=(12, 7))
    
    # Plot histogram
    hist_counts, bin_edges, _ = plt.hist(
        expressions_per_object,
        bins=np.arange(-0.5, np.max(expressions_per_object) + 1.5),
        alpha=0.6,
        color='steelblue',
        edgecolor='black',
        density=True,
        label='Observed data'
    )
    
    # Scale for plotting (convert density to counts)
    scale_factor = len(expressions_per_object) * (bin_edges[1] - bin_edges[0])
    
    # Plot the individual components
    colors = ['red', 'green', 'orange', 'purple']  # colors for components
    component_names = ['Poisson-like population', 'High-expressing population']
    
    for i in range(n_components):
        plt.plot(
            gmm_results["x_range"],
            gmm_results["densities"][i],
            '-',
            color=colors[i],
            linewidth=2,
            label=f'{component_names[i]}: μ={gmm_results["means"][i]:.2f}, σ²={gmm_results["variances"][i]:.2f}, w={gmm_results["weights"][i]:.2f}'
        )
    
    # Plot the total density
    plt.plot(
        gmm_results["x_range"],
        gmm_results["total_density"],
        'k-',
        linewidth=2.5,
        label='Total mixture density'
    )
    
    # Add model evaluation metrics
    metrics_text = f"BIC: {gmm_results['bic']:.2f}\nAIC: {gmm_results['aic']:.2f}"
    plt.annotate(
        metrics_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        va='top',
        fontsize=9
    )
    
    plt.xlabel('Number of Channels Expressed')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model Analysis of Barcode Expression Distribution')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.xticks(np.arange(0, np.max(expressions_per_object) + 1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return gmm_results

def classify_neurons(expressions_per_object, gmm_results, discrete, threshold):
    """Classify neurons into populations and analyze their characteristics"""
    # Get classification labels from GMM
    labels = gmm_results["labels"]
    
    # Sort labels by means for consistent ordering
    sort_idx = np.argsort(gmm_results["means"])
    
    # Map labels to "Population 1" and "Population 2" based on the sorted order
    population_mapping = {}
    for i, idx in enumerate(sort_idx):
        population_mapping[idx] = f"Population {i+1}"
    
    # Apply the mapping to get population names
    populations = np.array([population_mapping[label] for label in labels])
    
    # Create a DataFrame with the classifications
    neuron_data = pd.DataFrame({
        'Neuron_ID': np.arange(len(expressions_per_object)),
        'Channels_Expressed': expressions_per_object,
        'Population': populations
    })
    
    # Calculate the channel expression rate for each population
    channel_counts = {}
    for pop in neuron_data['Population'].unique():
        pop_indices = neuron_data[neuron_data['Population'] == pop].index
        channel_counts[pop] = np.mean(discrete[pop_indices], axis=0)
    
    return neuron_data, channel_counts

def plot_population_channel_patterns(neuron_data, channel_counts, channel_names=None, save_path=None):
    """Plot the channel expression patterns for each population"""
    if channel_names is None:
        # Default channel names if not provided
        channel_names = [f"Channel_{i+1}" for i in range(len(channel_counts['Population 1']))]
    
    # Create a DataFrame for easier plotting
    channel_data = pd.DataFrame(channel_counts)
    channel_data.index = channel_names
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Normalize the values for better visualization
    normalized_data = channel_data.copy()
    for col in normalized_data.columns:
        normalized_data[col] = normalized_data[col] / normalized_data[col].max()
    
    # Create heatmap
    sns.heatmap(
        normalized_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Normalized Expression Level'}
    )
    
    plt.title('Channel Expression Patterns Across Neuronal Populations')
    plt.ylabel('Channel')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return normalized_data

def run_full_analysis(file_path='neuron_barcodes_fixed_keys.npz', channel_names=None, output_dir='.'):
    """Run the complete analysis pipeline and save all results"""
    # Load data
    discrete, thresholded, expressions_per_object, total_cells, threshold = load_barcode_data(file_path)
    
    # Basic statistics
    stats = basic_stats(expressions_per_object)
    print("\n=== Basic Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Basic distribution plot
    plot_basic_distribution(
        expressions_per_object, 
        total_cells, 
        save_path=f"{output_dir}/basic_distribution.png"
    )
    
    # 2. Poisson fit for the lower population
    poisson_fit = plot_poisson_fit(
        expressions_per_object, 
        cutoff=15, 
        save_path=f"{output_dir}/poisson_fit.png"
    )
    
    # 3. Gaussian Mixture Model analysis
    gmm_results = plot_gaussian_mixture(
        expressions_per_object, 
        n_components=2, 
        save_path=f"{output_dir}/gmm_analysis.png"
    )
    
    # 4. Classify neurons and analyze population differences
    neuron_data, channel_counts = classify_neurons(
        expressions_per_object,
        gmm_results,
        discrete,
        threshold
    )
    
    # Summary of populations
    print("\n=== Population Summary ===")
    print(neuron_data['Population'].value_counts())
    print("\nPopulation Statistics:")
    print(neuron_data.groupby('Population')['Channels_Expressed'].describe())
    
    # 5. Channel expression patterns
    normalized_data = plot_population_channel_patterns(
        neuron_data,
        channel_counts,
        channel_names=channel_names,
        save_path=f"{output_dir}/channel_patterns.png"
    )
    
    # Save neuron classifications to CSV
    neuron_data.to_csv(f"{output_dir}/neuron_classifications.csv", index=False)
    
    # Save channel expression data
    pd.DataFrame(channel_counts).to_csv(f"{output_dir}/channel_expression_levels.csv")
    
    return {
        "statistics": stats,
        "poisson_fit": poisson_fit,
        "gmm_results": gmm_results,
        "neuron_data": neuron_data,
        "channel_counts": channel_counts,
        "normalized_data": normalized_data
    }

if __name__ == "__main__":
    # Example channel names (replace with actual channel names if available)
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
    
    # Run the full analysis
    results = run_full_analysis(channel_names=channel_names, output_dir='bimodal_analysis_results') 