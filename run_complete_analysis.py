#!/usr/bin/env python3
"""
Complete Barcode Expression Analysis Pipeline

This script runs both the basic distribution analysis and the hypothesis testing
to provide a complete analysis of the bimodal distribution in neural barcode data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from bimodal_barcode_analysis import run_full_analysis
from hypothesis_testing import run_all_hypothesis_tests

# Set up output directories
base_output_dir = "barcode_analysis_results"
basic_output_dir = os.path.join(base_output_dir, "basic_analysis")
hypothesis_output_dir = os.path.join(base_output_dir, "hypothesis_tests")

# Create base directory if it doesn't exist
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Channel names
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

def main():
    """Run the complete analysis pipeline"""
    print("Starting neural barcode expression analysis...")
    
    # Step 1: Run the basic distribution analysis
    print("\n=== Running Basic Distribution Analysis ===")
    basic_results = run_full_analysis(
        file_path='neuron_barcodes_fixed_keys.npz',
        channel_names=channel_names,
        output_dir=basic_output_dir
    )
    
    # Step 2: Run the hypothesis testing suite
    print("\n=== Running Statistical Hypothesis Tests ===")
    hypothesis_results = run_all_hypothesis_tests(
        file_path='neuron_barcodes_fixed_keys.npz',
        channel_names=channel_names,
        output_dir=hypothesis_output_dir
    )
    
    # Step 3: Generate a summary report
    generate_summary_report(basic_results, hypothesis_results, base_output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {base_output_dir}")
    print("See summary.txt for a overview of findings.")

def generate_summary_report(basic_results, hypothesis_results, output_dir):
    """Generate a text summary of the analysis results"""
    
    summary_path = os.path.join(output_dir, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== NEURAL BARCODE EXPRESSION ANALYSIS SUMMARY ===\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write("-----------------\n")
        for key, value in basic_results["statistics"].items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n")
        
        # Poisson fit results
        f.write("POISSON FIT RESULTS:\n")
        f.write("-------------------\n")
        poisson_fit = basic_results["poisson_fit"]
        f.write(f"Lambda (mean): {poisson_fit['lambda']:.4f}\n")
        f.write(f"Chi-square: {poisson_fit['chi2_stat']:.4f}\n")
        f.write(f"P-value: {poisson_fit['p_value']:.4f}\n")
        f.write(f"Degrees of freedom: {poisson_fit['dof']}\n")
        
        f.write("\n")
        
        # GMM results
        f.write("GAUSSIAN MIXTURE MODEL RESULTS:\n")
        f.write("-----------------------------\n")
        gmm_results = basic_results["gmm_results"]
        
        for i, (mean, var, weight) in enumerate(zip(
            gmm_results["means"], 
            gmm_results["variances"], 
            gmm_results["weights"]
        )):
            pop_name = "Poisson-like population" if i == 0 else "High-expressing population"
            f.write(f"Population {i+1} ({pop_name}):\n")
            f.write(f"  Mean: {mean:.4f}\n")
            f.write(f"  Variance: {var:.4f}\n")
            f.write(f"  Weight: {weight:.4f} ({weight*100:.1f}%)\n")
            f.write("\n")
        
        # Model comparison results
        f.write("MODEL COMPARISON RESULTS:\n")
        f.write("------------------------\n")
        model_comparison = hypothesis_results["model_comparison"]
        
        f.write("BIC scores (lower is better):\n")
        for name, score in zip(model_comparison["model_names"], model_comparison["bic_scores"]):
            f.write(f"  {name}: {score:.1f}\n")
        
        f.write("\nAIC scores (lower is better):\n")
        for name, score in zip(model_comparison["model_names"], model_comparison["aic_scores"]):
            f.write(f"  {name}: {score:.1f}\n")
        
        f.write("\n")
        
        # Dip test results
        f.write("HARTIGAN'S DIP TEST FOR UNIMODALITY:\n")
        f.write("----------------------------------\n")
        dip, pval = hypothesis_results["dip_results"]
        if dip is not None:
            f.write(f"Dip statistic: {dip:.4f}\n")
            f.write(f"P-value: {pval:.4f}\n")
            f.write(f"Interpretation: {'Rejects unimodality (p < 0.05)' if pval < 0.05 else 'Cannot reject unimodality (p >= 0.05)'}\n")
        else:
            f.write("Dip test not performed (diptest package required)\n")
        
        f.write("\n")
        
        # Cross-validation results
        f.write("GMM CROSS-VALIDATION RESULTS:\n")
        f.write("----------------------------\n")
        cv_results = hypothesis_results["cv_results"]
        f.write(f"Optimal number of components: {cv_results['best_components']}\n")
        
        f.write("\nMean log-likelihoods:\n")
        for i, ll in enumerate(cv_results["mean_log_likelihoods"]):
            f.write(f"  {i+1} components: {ll:.1f}\n")
        
        # Population statistics
        f.write("\nPOPULATION STATISTICS:\n")
        f.write("---------------------\n")
        neuron_data = hypothesis_results["neuron_data"]
        pop_counts = neuron_data["Population"].value_counts()
        
        for pop, count in pop_counts.items():
            f.write(f"{pop}: {count} neurons ({count/len(neuron_data)*100:.1f}%)\n")
        
        f.write("\n")
        
        # Channel comparison
        stat_comparison = hypothesis_results.get("stat_comparison")
        if stat_comparison is not None:
            f.write("TOP 5 DIFFERENTIALLY EXPRESSED CHANNELS:\n")
            f.write("-------------------------------------\n")
            
            for i, row in stat_comparison.head(5).iterrows():
                f.write(f"{row['Channel']}:\n")
                f.write(f"  Mean (Pop1): {row['Mean_Pop1']:.2f}\n")
                f.write(f"  Mean (Pop2): {row['Mean_Pop2']:.2f}\n")
                f.write(f"  Fold Change: {row['Fold_Change']:.2f}\n")
                f.write(f"  P-value: {row['P_Value']:.6f}\n")
                f.write(f"  Significant after correction: {row['Significant_Bonferroni']}\n")
                f.write("\n")
        
        # Conclusion
        f.write("CONCLUSION:\n")
        f.write("-----------\n")
        
        if stat_comparison is not None and cv_results['best_components'] == 2:
            f.write("The analysis strongly supports the presence of two distinct populations of neurons:\n")
            f.write("1. A lower-expressing population that follows a Poisson-like distribution\n")
            f.write("2. A high-expressing population with significantly more channels expressed\n\n")
            
            f.write("These findings suggest a biological phenomenon beyond random infection,\n")
            f.write("potentially indicating different cell types or infection susceptibility.")
        else:
            f.write("The analysis suggests a complex distribution pattern. ")
            f.write(f"The optimal model identified {cv_results['best_components']} distinct components.\n\n")
            
            f.write("Further investigation is recommended to understand the biological factors\n")
            f.write("contributing to this pattern.")

if __name__ == "__main__":
    main() 