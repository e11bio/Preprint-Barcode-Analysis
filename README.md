# Barcode Expression Bimodal Distribution Analysis

This repository contains code for analyzing and characterizing the bimodal distribution of barcode expressions in neurons. The analysis is designed for use in scientific publications investigating viral infection patterns in neural barcoding experiments.

## Overview

The analysis pipeline systematically explores and quantifies the presence of two distinct neuronal populations:

1. A **Poisson-like population** with lower channel expression counts following expected random infection dynamics
2. A **High-expressing population** of neurons expressing significantly more channels than expected

## Features

The analysis pipeline provides:

- Statistical characterization of the bimodal distribution
- Gaussian Mixture Model (GMM) fitting to identify the two populations
- Poisson distribution fitting to the lower-expressing population
- Classification of individual neurons into their respective populations
- Analysis of channel-specific expression patterns in each population
- Rigorous statistical hypothesis testing to validate the bimodality
- Publication-quality figures and data exports

## Requirements

- Python 3.7+
- NumPy
- SciPy
- pandas
- matplotlib
- scikit-learn
- seaborn

Install dependencies:
```
pip install numpy scipy pandas matplotlib scikit-learn seaborn diptest
```

Note: The `diptest` package is optional but recommended for the Hartigan's dip test of unimodality.

## Usage

### Basic Analysis

```python
from bimodal_barcode_analysis import run_full_analysis

# List your channel names if available
channel_names = [
    'E2-barcode-R1',
    'S1-barcode-R1',
    'ALFA-barcode-R1',
    # ... other channels
]

# Run the full analysis
results = run_full_analysis(
    file_path='neuron_barcodes_fixed_keys.npz',  # Path to your data file
    channel_names=channel_names,                 # Optional: list of channel names
    output_dir='analysis_results'                # Directory to save output files
)
```

### Advanced Hypothesis Testing

For more rigorous statistical validation of the bimodality:

```python
from hypothesis_testing import run_all_hypothesis_tests

# Run all hypothesis tests
test_results = run_all_hypothesis_tests(
    file_path='neuron_barcodes_fixed_keys.npz',
    channel_names=channel_names,
    output_dir='hypothesis_tests'
)
```

## Analysis Methods

### 1. Initial Data Processing

The raw barcode signal intensities are thresholded to create a binary matrix indicating expression (1) or non-expression (0) for each channel in each neuron.

### 2. Basic Distribution Analysis

Basic statistical measures (mean, median, variance, etc.) are calculated and a histogram is created to visualize the distribution of channel expression counts.

### 3. Poisson Fit Analysis

A Poisson distribution is fitted to the lower-expressing population to test whether these neurons follow a random (Poisson) infection process. Chi-square goodness-of-fit tests quantify the match between the observed data and Poisson expectation.

### 4. Gaussian Mixture Model Analysis

A two-component Gaussian Mixture Model identifies and characterizes the two populations. Model evaluation metrics (BIC, AIC) are calculated to assess the quality of the fit.

### 5. Population Classification and Characterization

Neurons are classified into the two populations, and the expression patterns of each population are analyzed across all channels.

### 6. Hypothesis Testing

Several statistical tests are performed to validate the presence of two distinct populations:

- **Model Comparison**: Different distribution models (Poisson, 1-4 component GMMs) are compared using BIC and AIC metrics.
- **Hartigan's Dip Test**: Tests for multimodality in the distribution.
- **Cross-Validation**: K-fold cross-validation determines the optimal number of GMM components.
- **Channel Expression Comparison**: Statistical comparison of channel expression between the two populations, including t-tests and effect size calculations.

## Output Files

The analysis generates:

### Basic Analysis Outputs

- **PNG figures**:
  - `basic_distribution.png`: Histogram of barcode expressions
  - `poisson_fit.png`: Poisson distribution fit to the lower-expressing population
  - `gmm_analysis.png`: GMM analysis of the bimodal distribution
  - `channel_patterns.png`: Heatmap of channel expression across populations

- **CSV data files**:
  - `neuron_classifications.csv`: Classification of each neuron with expression counts
  - `channel_expression_levels.csv`: Expression levels for each channel in each population

### Hypothesis Testing Outputs

- **PNG figures**:
  - `model_comparison.png`: BIC and AIC comparison of different models
  - `dip_test.png`: Results of Hartigan's dip test for unimodality
  - `cross_validation.png`: Cross-validation results for determining optimal GMM components
  - `population_comparison.png`: Differential expression of channels between populations

- **CSV data files**:
  - `channel_statistical_comparison.csv`: Statistical comparison of channel expression between populations

## Citation

If you use this code in your research, please cite:

```
Publication in preparation
```

## License

MIT License 