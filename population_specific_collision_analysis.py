import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def barcode_collision_analysis_for_population(thresholded, population_mask, population_name):
    """
    Analyze barcode collisions for a specific population
    
    Parameters:
    -----------
    thresholded : numpy.ndarray
        Binary matrix where 1 = expression above threshold, 0 = expression below
    population_mask : numpy.ndarray
        Boolean mask indicating which cells belong to the desired population
    population_name : str
        Name of the population for plot titles
    
    Returns:
    --------
    dict
        Dictionary containing collision analysis results and visualization
    """
    # Filter to include only the specified population
    filtered_thresholded = thresholded[population_mask]
    
    # Convert binary arrays to strings for hashability
    barcode_strings = [''.join(map(str, row)) for row in filtered_thresholded]
    
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
    total_cells = filtered_thresholded.shape[0]
    collision_df['Percentage'] = (collision_df['Number of Cells'] / total_cells) * 100
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Set color based on population
    plot_color = 'royalblue' if population_name == 'Expected Population' else 'firebrick'
    
    # Bar plot of collision counts
    ax1.bar(collision_df['Collisions'], collision_df['Number of Cells'], 
            alpha=0.7, edgecolor='black', color=plot_color)
    ax1.set_xlabel('Number of Collisions')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title(f'Barcode Collision Distribution - {population_name} Only')
    ax1.set_xticks(collision_df['Collisions'])
    ax1.grid(alpha=0.3)
    
    # Add counts to each bar
    for i, v in enumerate(collision_df['Number of Cells']):
        ax1.text(collision_df['Collisions'].iloc[i], v + max(5, v*0.05), str(v), 
                ha='center', va='bottom')
    
    # Pie chart of collision percentages
    ax2.pie(collision_df['Number of Cells'], labels=[f"{c} collisions" for c in collision_df['Collisions']], 
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title(f'Percentage of Cells by Collision Count - {population_name} Only')
    
    plt.tight_layout()
    
    # Create a summary statistics dictionary
    summary_stats = {
        "total_cells": total_cells,
        "unique_patterns": unique_patterns,
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

def process_population(thresholded, population_mask, population_name, output_dir):
    """Process a single population and save results"""
    # Perform collision analysis
    results = barcode_collision_analysis_for_population(thresholded, population_mask, population_name)
    
    # Create a safe filename from the population name
    safe_name = population_name.lower().replace(' ', '_')
    
    # Save the plot
    results['figure'].savefig(f'{output_dir}/{safe_name}_collision_analysis.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    total_cells = results['summary_stats']['total_cells']
    unique_patterns = results['unique_patterns']
    unique_percentage = results['summary_stats']['unique_cells_percentage']
    
    print(f"Analysis of {population_name}:")
    print(f"Total cells: {total_cells}")
    print(f"Number of unique barcode patterns: {unique_patterns}")
    print(f"Percentage of cells with unique barcodes: {unique_percentage:.2f}%")
    print(f"\nCollision Distribution Table:")
    print(results['markdown_table'])
    
    # Create a markdown summary file
    with open(f'{output_dir}/{safe_name}_analysis.md', 'w') as f:
        f.write(f"# Barcode Collision Analysis - {population_name}\n\n")
        f.write(f"Total cells: {total_cells}\n\n")
        f.write(f"Number of unique barcode patterns: {unique_patterns}\n\n")
        f.write(f"Percentage of cells with unique barcodes: {unique_percentage:.2f}%\n\n")
        f.write("## Collision Distribution Table\n\n")
        f.write(results['markdown_table'])
    
    return results

def create_comparison_plot(expected_results, overexp_results, output_dir):
    """Create a plot comparing collision distributions between the two populations"""
    expected_df = expected_results['collision_df'].copy()
    expected_df['Population'] = 'Expected Population'
    
    overexp_df = overexp_results['collision_df'].copy()
    overexp_df['Population'] = 'Overexpressors'
    
    # Combine data for plotting
    combined_df = pd.concat([expected_df, overexp_df])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for a more sophisticated grouped bar chart
    import seaborn as sns
    ax = sns.barplot(x='Collisions', y='Percentage', hue='Population', data=combined_df,
                     palette={'Expected Population': 'royalblue', 'Overexpressors': 'firebrick'})
    
    plt.title('Barcode Collision Distribution Comparison Between Populations')
    plt.xlabel('Number of Collisions')
    plt.ylabel('Percentage of Cells (%)')
    plt.legend(title='Population')
    plt.grid(alpha=0.3)
    
    # Add percentages above the bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/population_comparison_collision_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create a markdown comparison file
    with open(f'{output_dir}/population_comparison.md', 'w') as f:
        f.write("# Barcode Collision Analysis - Population Comparison\n\n")
        
        f.write("## Expected Population\n")
        f.write(f"Total cells: {expected_results['summary_stats']['total_cells']}\n")
        f.write(f"Unique barcode patterns: {expected_results['unique_patterns']}\n")
        f.write(f"Percentage of cells with unique barcodes: {expected_results['summary_stats']['unique_cells_percentage']:.2f}%\n\n")
        
        f.write("## Overexpressors\n")
        f.write(f"Total cells: {overexp_results['summary_stats']['total_cells']}\n")
        f.write(f"Unique barcode patterns: {overexp_results['unique_patterns']}\n")
        f.write(f"Percentage of cells with unique barcodes: {overexp_results['summary_stats']['unique_cells_percentage']:.2f}%\n\n")
        
        f.write("The comparison plot shows the percentage of cells with different collision counts for both populations.\n")
        f.write("This analysis helps understand if one population has more unique barcodes than the other,\n")
        f.write("which could indicate different biological properties or technical artifacts.\n")

def main():
    # Load the neuron classification data
    neuron_data = pd.read_csv('bimodal_analysis_results/neuron_classifications.csv')
    
    # Load the barcode data - UPDATED to use the correct data file
    data = np.load('Data/neuron_barcodes_full_roi.npz', allow_pickle=True)
    barcode_data = data['arr_0'].item()
    
    # Extract discrete values for each object across channels
    discrete = np.array([obj_dict['discrete'] for obj_dict in barcode_data.values()])
    
    # Calculate the threshold for each channel (mean of discrete values)
    threshold = np.mean(discrete, axis=0)
    
    # Create binary matrix where 1 = expression above threshold, 0 = expression below threshold
    thresholded = (discrete > threshold).astype(int)
    
    # Create masks for each population
    expected_pop_mask = neuron_data['Population'] == 'Expected Population'
    overexp_mask = neuron_data['Population'] == 'Overexpressors'
    
    # Create output directory if it doesn't exist
    output_dir = 'population_specific_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process Expected Population
    print("\n" + "="*50)
    print("ANALYZING EXPECTED POPULATION")
    print("="*50)
    expected_results = process_population(thresholded, expected_pop_mask, 'Expected Population', output_dir)
    
    # Process Overexpressors
    print("\n" + "="*50)
    print("ANALYZING OVEREXPRESSORS")
    print("="*50)
    overexp_results = process_population(thresholded, overexp_mask, 'Overexpressors', output_dir)
    
    # Create comparison visualization
    print("\n" + "="*50)
    print("CREATING POPULATION COMPARISON")
    print("="*50)
    create_comparison_plot(expected_results, overexp_results, output_dir)
    
    print(f"\nAnalysis complete. All results saved to {output_dir} directory.")

if __name__ == "__main__":
    main() 