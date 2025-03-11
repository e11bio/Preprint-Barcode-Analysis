# Advanced Analysis of AAV Infection Dynamics and Neuron Populations

This project provides comprehensive analyses of AAV infection dynamics and neuron population characteristics, building on previously established bimodal distribution of barcode expressions in neurons.

## Overview

The advanced analyses include:

1. **Co-occurrence Analysis of Epitope Expression** - Reveals which epitopes tend to be expressed together, identifying patterns unique to each population.

2. **Hierarchical Clustering Analysis** - Identifies potential subpopulations within the two main neuronal populations.

3. **Infection Marker Correlation Analysis** - Investigates relationships between viral load and channel expression patterns.

4. **Channel Preference Analysis** - Tests whether different neuron populations have specific channel preferences.

5. **Dimensionality Reduction Analysis** - Uses t-SNE to visualize high-dimensional expression patterns in 2D space.

6. **Network Analysis of Epitope Co-expression** - Creates network diagrams to visualize co-expression patterns.

7. **NGS Correlation Analysis** - Correlates epitope expression patterns with sequencing data.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- pandas
- matplotlib
- scikit-learn
- seaborn
- networkx
- statsmodels

Install dependencies:
```
pip install numpy scipy pandas matplotlib scikit-learn seaborn networkx statsmodels
```

## Quick Start

Run the comprehensive analysis with default parameters:

```bash
python run_advanced_analysis.py
```

By default, this script will:
- Load data from `Data/neuron_barcodes_fixed_keys.npz`
- Use NGS data from `Data/221208-pool-seq-clean.csv`
- Save results to the `advanced_analysis_results/` directory

## Custom Options

You can customize the paths and output directory:

```bash
python run_advanced_analysis.py --data path/to/data.npz --ngs path/to/ngs_data.csv --output custom_output_dir
```

## Output Files

The analysis generates the following output files in the output directory:

- **PNG figures**:
  - `epitope_co_occurrence.png` - Heatmaps of epitope co-expression patterns
  - `hierarchical_clustering.png` - Dendrogram showing neuron clustering
  - `infection_correlation.png` - Correlation between viral load and expression
  - `channel_preference.png` - Channel preference by population
  - `dimensionality_reduction.png` - t-SNE visualization of populations
  - `epitope_networks.png` - Network visualization of co-expression patterns
  - `ngs_correlation.png` - Correlation with NGS sequencing data

- **Data files**:
  - `analysis_summary.csv` - Summary of key findings from each analysis
  - `comprehensive_analysis_report.md` - Detailed report explaining all analyses

## Interpretation Guide

### Co-occurrence Analysis
The heatmaps show which epitopes tend to be expressed together in each population. The differential heatmap highlights combinations that are enriched in the Overexpressors population (red) or the Expected Population (blue).

### Hierarchical Clustering
The dendrogram illustrates how neurons cluster based on epitope expression patterns. The color bar below shows which population each neuron belongs to, helping to identify potential subpopulations.

### Infection Marker Correlation
This plot shows the relationship between total signal intensity (a proxy for viral load) and the number of channels expressed, with separate regression lines for each population.

### Channel Preference Analysis
The upper plot shows preference scores for each channel in each population, where values above 1 indicate preference. The lower plot shows statistical significance of these differences.

### Dimensionality Reduction
The t-SNE plot visualizes neurons in 2D space based on their epitope expression patterns, with colors indicating population membership and point sizes showing channel expression counts.

### Network Analysis
These network diagrams visualize epitope co-expression patterns as graphs. Node size indicates expression frequency, and edge width represents co-expression strength.

### NGS Correlation Analysis
These plots correlate epitope expression frequencies with viral and plasmid frequencies from NGS data, investigating how the input viral pool composition influences expression patterns.

## Advanced Usage

You can import individual analysis functions from `advanced_barcode_analysis.py` to run specific analyses:

```python
from advanced_barcode_analysis import epitope_co_occurrence_analysis, hierarchical_clustering_analysis

# Run specific analyses
results = epitope_co_occurrence_analysis(thresholded, neuron_classifications, channel_names)
``` 