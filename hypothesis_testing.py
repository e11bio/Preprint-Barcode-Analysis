import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.model_selection import KFold
from bimodal_barcode_analysis import load_barcode_data

def compare_distribution_models(expressions_per_object, max_components=4, save_path=None):
    """
    Compare different distribution models (Poisson, 1-4 component GMMs) using BIC and AIC
    
    Parameters:
    -----------
    expressions_per_object : numpy.ndarray
        Array containing the number of channels expressed by each neuron
    max_components : int
        Maximum number of Gaussian components to try
    save_path : str or None
        Path to save the comparison plot
        
    Returns:
    --------
    dict
        Dictionary containing model comparison results
    """
    # Reshape data for sklearn
    X = expressions_per_object.reshape(-1, 1)
    
    # Model scores
    bic_scores = []
    aic_scores = []
    models = []
    model_names = []
    
    # 1. Fit a Poisson distribution
    poisson_lambda = np.mean(expressions_per_object)
    poisson_log_likelihood = np.sum(stats.poisson.logpmf(expressions_per_object, poisson_lambda))
    
    # Calculate BIC and AIC for Poisson (k=1 parameter)
    n = len(expressions_per_object)
    k = 1  # one parameter (lambda)
    poisson_bic = -2 * poisson_log_likelihood + k * np.log(n)
    poisson_aic = -2 * poisson_log_likelihood + 2 * k
    
    bic_scores.append(poisson_bic)
    aic_scores.append(poisson_aic)
    model_names.append("Poisson")
    
    # 2. Fit GMMs with increasing number of components
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
        models.append(gmm)
        model_names.append(f"{n_components}-component GMM")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(model_names, bic_scores, alpha=0.7, color='royalblue')
    plt.title("BIC Scores (lower is better)")
    plt.xticks(rotation=45)
    
    # Add values on top of the bars
    for i, v in enumerate(bic_scores):
        plt.text(i, v + 10, f"{v:.1f}", ha='center')
    
    plt.subplot(2, 1, 2)
    plt.bar(model_names, aic_scores, alpha=0.7, color='indianred')
    plt.title("AIC Scores (lower is better)")
    plt.xticks(rotation=45)
    
    # Add values on top of the bars
    for i, v in enumerate(aic_scores):
        plt.text(i, v + 10, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Return results
    results = {
        "model_names": model_names,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "poisson_lambda": poisson_lambda,
        "models": models
    }
    
    return results

def dip_test_of_unimodality(expressions_per_object, save_path=None):
    """
    Perform Hartigan's dip test for unimodality
    
    Parameters:
    -----------
    expressions_per_object : numpy.ndarray
        Array containing the number of channels expressed by each neuron
    save_path : str or None
        Path to save the comparison plot
        
    Returns:
    --------
    tuple
        (dip_statistic, p_value)
    """
    try:
        from diptest import diptest
    except ImportError:
        print("Please install diptest package: pip install diptest")
        print("Returning placeholder values")
        return (None, None)
    
    # Run the dip test
    dip, pval = diptest(expressions_per_object)
    
    # Create figure showing the distribution and test result
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    bins = np.arange(-0.5, np.max(expressions_per_object) + 1.5)
    plt.hist(
        expressions_per_object,
        bins=bins,
        alpha=0.7,
        color='steelblue',
        edgecolor='black'
    )
    
    # Add test result
    plt.annotate(
        f"Hartigan's Dip Test\nDip statistic: {dip:.4f}\np-value: {pval:.4f}\n" + 
        f"{'Rejects unimodality' if pval < 0.05 else 'Cannot reject unimodality'}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        va='top',
        fontsize=10
    )
    
    plt.xlabel('Number of Channels Expressed')
    plt.ylabel('Frequency')
    plt.title("Hartigan's Dip Test for Unimodality")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return (dip, pval)

def mixture_model_cross_validation(expressions_per_object, max_components=4, n_splits=5, save_path=None):
    """
    Perform cross-validation to find the optimal number of GMM components
    
    Parameters:
    -----------
    expressions_per_object : numpy.ndarray
        Array containing the number of channels expressed by each neuron
    max_components : int
        Maximum number of Gaussian components to try
    n_splits : int
        Number of cross-validation folds
    save_path : str or None
        Path to save the comparison plot
        
    Returns:
    --------
    dict
        Dictionary containing cross-validation results
    """
    # Reshape data for sklearn
    X = expressions_per_object.reshape(-1, 1)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store log-likelihoods
    cv_log_likelihoods = np.zeros((max_components, n_splits))
    
    # Perform cross-validation
    for n_components in range(1, max_components + 1):
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            
            # Fit the model on training data
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(X_train)
            
            # Evaluate on test data
            cv_log_likelihoods[n_components - 1, i] = gmm.score(X_test) * len(X_test)
    
    # Calculate mean log-likelihood for each number of components
    mean_log_likelihoods = np.mean(cv_log_likelihoods, axis=1)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, max_components + 1), mean_log_likelihoods, 'bo-')
    plt.xlabel('Number of GMM Components')
    plt.ylabel('Mean Cross-Validated Log-Likelihood')
    plt.title('GMM Cross-Validation')
    plt.grid(alpha=0.3)
    plt.xticks(range(1, max_components + 1))
    
    # Add optimal number of components
    best_components = np.argmax(mean_log_likelihoods) + 1
    plt.axvline(x=best_components, color='r', linestyle='--', alpha=0.5)
    plt.annotate(
        f'Optimal: {best_components} components',
        xy=(best_components, mean_log_likelihoods[best_components - 1]),
        xytext=(best_components + 0.1, mean_log_likelihoods[best_components - 1]),
        va='center'
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Return results
    results = {
        "cv_log_likelihoods": cv_log_likelihoods,
        "mean_log_likelihoods": mean_log_likelihoods,
        "best_components": best_components
    }
    
    return results

def compare_expression_between_populations(neuron_data, discrete_data, channel_names=None, save_path=None):
    """
    Statistically compare channel expression between the two identified populations
    
    Parameters:
    -----------
    neuron_data : pandas.DataFrame
        DataFrame containing neuron classification data
    discrete_data : numpy.ndarray
        Array containing the raw expression values for each neuron and channel
    channel_names : list or None
        List of channel names
    save_path : str or None
        Path to save the comparison plot
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with statistical comparison results
    """
    # Get indices for each population
    pop1_indices = neuron_data[neuron_data['Population'] == 'Population 1'].index
    pop2_indices = neuron_data[neuron_data['Population'] == 'Population 2'].index
    
    # Extract expression data for each population
    pop1_data = discrete_data[pop1_indices]
    pop2_data = discrete_data[pop2_indices]
    
    # Channel names
    if channel_names is None:
        channel_names = [f"Channel_{i+1}" for i in range(discrete_data.shape[1])]
    
    # Perform statistical tests for each channel
    results = []
    
    for i, channel in enumerate(channel_names):
        # Extract channel data
        channel_pop1 = pop1_data[:, i]
        channel_pop2 = pop2_data[:, i]
        
        # Calculate means
        mean_pop1 = np.mean(channel_pop1)
        mean_pop2 = np.mean(channel_pop2)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(channel_pop1, channel_pop2, equal_var=False)
        
        # Calculate fold change
        fold_change = mean_pop2 / mean_pop1 if mean_pop1 > 0 else float('inf')
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(channel_pop1, ddof=1) + np.var(channel_pop2, ddof=1)) / 2)
        effect_size = (mean_pop2 - mean_pop1) / pooled_std if pooled_std > 0 else float('inf')
        
        # Significance after Bonferroni correction
        sig_bonferroni = "Yes" if p_value < (0.05 / len(channel_names)) else "No"
        
        # Store results
        results.append({
            'Channel': channel,
            'Mean_Pop1': mean_pop1,
            'Mean_Pop2': mean_pop2,
            'Fold_Change': fold_change,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Effect_Size': effect_size,
            'Significant_Bonferroni': sig_bonferroni
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by p-value
    results_df = results_df.sort_values('P_Value')
    
    # Plot comparison with p-values
    plt.figure(figsize=(12, 8))
    
    # Extract top 10 significantly different channels
    top_channels = results_df.head(10)
    
    # Create a bar plot for the means
    channel_indices = np.arange(len(top_channels))
    bar_width = 0.35
    
    # Plot bars
    plt.bar(channel_indices - bar_width/2, top_channels['Mean_Pop1'], bar_width, 
            alpha=0.6, color='blue', label='Population 1')
    plt.bar(channel_indices + bar_width/2, top_channels['Mean_Pop2'], bar_width, 
            alpha=0.6, color='red', label='Population 2')
    
    # Add p-value stars
    for i, p in enumerate(top_channels['P_Value']):
        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        
        if stars:
            max_height = max(top_channels['Mean_Pop1'].iloc[i], top_channels['Mean_Pop2'].iloc[i])
            plt.text(i, max_height * 1.05, stars, ha='center', fontsize=12)
    
    # Set labels
    plt.xlabel('Channel')
    plt.ylabel('Mean Expression Level')
    plt.title('Top 10 Differentially Expressed Channels Between Populations')
    plt.xticks(channel_indices, top_channels['Channel'], rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add p-value legend
    plt.figtext(0.15, 0.01, "* p < 0.05, ** p < 0.01, *** p < 0.001", ha="left")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results_df

def run_all_hypothesis_tests(file_path='neuron_barcodes_fixed_keys.npz', 
                             channel_names=None, 
                             output_dir='hypothesis_tests'):
    """
    Run all hypothesis tests to validate the bimodal distribution
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    channel_names : list or None
        List of channel names
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    dict
        Dictionary containing all test results
    """
    # Create output directory
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    discrete, thresholded, expressions_per_object, total_cells, threshold = load_barcode_data(file_path)
    
    # 1. Model comparison
    model_comparison = compare_distribution_models(
        expressions_per_object, 
        max_components=4, 
        save_path=f"{output_dir}/model_comparison.png"
    )
    
    # 2. Dip test
    dip_results = dip_test_of_unimodality(
        expressions_per_object, 
        save_path=f"{output_dir}/dip_test.png"
    )
    
    # 3. Cross-validation
    cv_results = mixture_model_cross_validation(
        expressions_per_object, 
        max_components=4, 
        save_path=f"{output_dir}/cross_validation.png"
    )
    
    # Get the best model based on cross-validation
    best_components = cv_results["best_components"]
    
    # Fit the best model
    X = expressions_per_object.reshape(-1, 1)
    best_gmm = GaussianMixture(n_components=best_components, random_state=42)
    best_gmm.fit(X)
    
    # Get labels
    labels = best_gmm.predict(X)
    
    # Create neuron data DataFrame
    neuron_data = pd.DataFrame({
        'Neuron_ID': np.arange(len(expressions_per_object)),
        'Channels_Expressed': expressions_per_object,
        'Population': [f"Population {label+1}" for label in labels]
    })
    
    # 4. Compare expression between populations
    if best_components == 2:
        stat_comparison = compare_expression_between_populations(
            neuron_data,
            discrete,
            channel_names=channel_names,
            save_path=f"{output_dir}/population_comparison.png"
        )
        
        # Save comparison results
        stat_comparison.to_csv(f"{output_dir}/channel_statistical_comparison.csv", index=False)
    else:
        print(f"Warning: {best_components} components detected, skipping population comparison (requires exactly 2)")
        stat_comparison = None
    
    # Return all results
    return {
        "model_comparison": model_comparison,
        "dip_results": dip_results,
        "cv_results": cv_results,
        "best_gmm": best_gmm,
        "neuron_data": neuron_data,
        "stat_comparison": stat_comparison
    }

if __name__ == "__main__":
    # Example channel names
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
    
    # Run all hypothesis tests
    results = run_all_hypothesis_tests(
        channel_names=channel_names,
        output_dir='hypothesis_test_results'
    ) 