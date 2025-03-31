# this script generates a plot for epitope distribution 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy

# Import plotting settings
from plot_settings import MAIN_COLOR, SECONDARY_COLOR, FIG_SIZE, set_style, DPI

# Import functions from soma-preprocessing.py
from soma_preprocessing import (
    generate_barcode_array,
    target_channels
)

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()

epitope_counts = np.sum(soma_barcodes, axis=0)  # Sum each column
total_cells = len(soma_barcodes)
epitope_percentages = (epitope_counts / total_cells) * 100

# Calculate the average percentage
mean_percentage = np.mean(epitope_percentages)
median_percentage = np.median(epitope_percentages)

# Create a DataFrame for easier plotting with the target names
# Extract just the first part of each epitope name (before "-barcode")
simplified_epitopes = [name.split('-')[0] for name in target_channels]
epitope_df = pd.DataFrame({
    'Epitope': simplified_epitopes,
    'Percentage': epitope_percentages
})

# Sort from highest to lowest percentage for better visualization
epitope_df = epitope_df.sort_values('Percentage', ascending=False)

# Create the plot
plt.figure(figsize=FIG_SIZE, dpi=DPI)
set_style()  # Apply the standard plotting style
bars = sns.barplot(x='Epitope', y='Percentage', data=epitope_df, color=MAIN_COLOR)

# Remove grid lines
plt.grid(False)
ax = plt.gca()
ax.grid(False)

# Despine the plot (remove top and right spines)
sns.despine()

# Set x-axis label font size to 8
plt.xticks(fontsize=8)

# Add percentage labels only for the lowest and highest bars
min_idx = epitope_df['Percentage'].idxmin()
max_idx = epitope_df['Percentage'].idxmax()

for i, p in enumerate(bars.patches):
    if i == 0 or i == len(bars.patches)-1:  # Only label first (highest) and last (lowest) bars
        # Adjust horizontal position for the first bar (highest) to avoid overlap with y-axis
        h_align = 'center'
        x_pos = p.get_x() + p.get_width() / 2.
        
        # Push the highest bar's label slightly to the right
        if i == 0:  # This is the highest bar
            x_pos += 0.1  # Shift to the right
        
        bars.annotate(f'{p.get_height():.1f}%', 
                     (x_pos, p.get_height()),
                     ha=h_align, va='bottom')

# Add labels and title
plt.xlabel('Epitope')
plt.ylabel('Percentage of Cells (%)')
plt.title(f'Percentage of Somas Containing Each Epitope (n={len(soma_barcodes)})')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

output_dir = "/home/aashir/repos/barcode_analysis/Preprint-Barcode-Analysis/Plotting-scripts/Output"

plot_filename = os.path.join(output_dir, "epitope_distribution.png")
plt.savefig(plot_filename, dpi=500)




# Create markdown file with context
md_content = f"""# Epitope Distribution Analysis

## Overview
This plot shows the distribution of epitopes across 147 somas in the barcode analysis, sorted in descending order.

## Details
- **Total somas analyzed**: {len(soma_barcodes)}
- **Highest epitope presence**: {epitope_df['Percentage'].max():.1f}%
- **Lowest epitope presence**: {epitope_df['Percentage'].min():.1f}%

The histogram showcases the distribution of each epitope of the barcode, indicating the percentage of cells containing each epitope.

![Epitope Distribution](epitope_distribution.png)
"""

md_filename = os.path.join(output_dir, "epitope_distribution.md")
with open(md_filename, 'w') as md_file:
    md_file.write(md_content)

print(f"Plot saved to {plot_filename}")
print(f"Markdown documentation saved to {md_filename}")



# lowest and the highest, no other percentages in between
# get rid of mean

