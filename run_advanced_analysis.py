#!/usr/bin/env python3
"""
Runner script for advanced analysis of AAV infection dynamics and neuron populations.
This script executes all the advanced analyses defined in advanced_barcode_analysis.py.
"""

import os
import argparse
from advanced_barcode_analysis import run_comprehensive_analysis

def main():
    """
    Main function to parse arguments and run the comprehensive analysis.
    """
    parser = argparse.ArgumentParser(description='Run advanced analyses of AAV infection dynamics.')
    
    parser.add_argument('--data', type=str, default='Data/neuron_barcodes_full_roi.npz',
                        help='Path to the neuron barcode data (.npz file)')
    
    parser.add_argument('--ngs', type=str, default='Data/221208-pool-seq-clean.csv',
                        help='Path to the NGS sequencing data (.csv file)')
    
    parser.add_argument('--output', type=str, default='advanced_analysis_results',
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found!")
        return 1
    
    if not os.path.exists(args.ngs):
        print(f"Error: NGS data file {args.ngs} not found!")
        return 1
    
    # Run the comprehensive analysis
    print(f"Running comprehensive analysis...")
    print(f"Data file: {args.data}")
    print(f"NGS data file: {args.ngs}")
    print(f"Output directory: {args.output}")
    
    results = run_comprehensive_analysis(
        data_path=args.data,
        ngs_data_path=args.ngs,
        output_dir=args.output
    )
    
    print("Analysis complete!")
    print(f"Results saved to {args.output}/")
    
    return 0

if __name__ == "__main__":
    exit(main()) 