#!/usr/bin/env python3
"""
Runner script for streamlined analysis of AAV infection dynamics and neuron populations.
This script executes a focused subset of the advanced analyses defined in advanced_barcode_analysis.py.
"""

import os
import argparse
from advanced_barcode_analysis import run_comprehensive_analysis

def main():
    """
    Main function to parse arguments and run the focused analysis.
    """
    parser = argparse.ArgumentParser(description='Run focused analyses of AAV infection dynamics.')
    
    parser.add_argument('--data', type=str, default='Data/neuron_barcodes_full_roi.npz',
                        help='Path to the neuron barcode data (.npz file)')
    
    parser.add_argument('--ngs', type=str, default='Data/221208-pool-seq-clean.csv',
                        help='Path to the NGS sequencing data (.csv file)')
    
    parser.add_argument('--output', type=str, default='focused_analysis_results',
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found!")
        return 1
    
    if not os.path.exists(args.ngs):
        print(f"Error: NGS data file {args.ngs} not found!")
        return 1
    
    # Run the focused analysis
    print(f"Running focused analysis with the following analyses:")
    print("- Epitope Co-occurrence Analysis")
    print("- Infection Marker Correlation Analysis")
    print("- NGS Correlation Analysis")
    print("- Hamming Distance Analysis")
    print("- Barcode Collision Analysis")
    print(f"\nData file: {args.data}")
    print(f"NGS data file: {args.ngs}")
    print(f"Output directory: {args.output}")
    
    results = run_comprehensive_analysis(
        data_path=args.data,
        ngs_data_path=args.ngs,
        output_dir=args.output
    )
    
    print("\nAnalysis complete!")
    print(f"The following files were generated in {args.output}/:")
    print("- epitope_co_occurrence.png - Shows co-occurrence patterns of epitopes")
    print("- infection_correlation.png - Shows correlation between viral load and expression")
    print("- ngs_correlation.png - Shows correlation with NGS sequencing data")
    print("- hamming_distance_analysis.png - Shows distribution of closest-match distances")
    print("- hamming_distance_by_population.png - Compares distances between populations")
    print("- barcode_collision_analysis.png - Shows cells with unique vs shared barcodes")
    print("- analysis_summary.csv - Summary of key findings")
    print("- comprehensive_analysis_report.md - Detailed report of all analyses")
    
    return 0

if __name__ == "__main__":
    exit(main()) 