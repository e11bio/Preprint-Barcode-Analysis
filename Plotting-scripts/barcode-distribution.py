# This script generates a plot for barcode distribution given a numpy array of binary barcodes. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy

# Import plotting settings
from plot_settings import MAIN_COLOR, SECONDARY_COLOR, FIG_SIZE, set_style

# Import functions from soma-preprocessing.py
from soma_preprocessing import (
    generate_barcode_array
)

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()


# creating a plot of barcode length 

barcode_lengths = np.sum(soma_barcodes, axis=1)

plt.figure(figsize=FIG_SIZE)

ax = sns.histplot(barcode_lengths, kde=False, bins=range(19), discrete=True, color=MAIN_COLOR, stat='percent')

ax.grid(False) # turning off the grid

# Add labels and title
plt.xlabel('Barcode Length')
plt.ylabel('Frequency (%)')
plt.title(f'Distribution of Barcode Lengths Across {len(barcode_lengths)} Somas')

# Set x-axis ticks to include all possible barcode lengths (0 to 18)
plt.xticks(range(19))

# Add a grid for better readability
# plt.grid(axis='y', alpha=0.3)

# Calculate some statistics to add as text
mean_length = np.mean(barcode_lengths)
median_length = np.median(barcode_lengths)
max_length = np.max(barcode_lengths)

# Add statistics as text
stats_text = f"Mean: {mean_length:.2f}"
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=12)

# Create output directory if it doesn't exist
output_dir = "/home/aashir/repos/barcode_analysis/Preprint-Barcode-Analysis/Plotting-scripts/Output"

# Save the plot
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"{output_dir}/somas_barcode_length_distr.png"
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)
plt.close()

# Also print basic statistics
print(f"Total cells: {len(barcode_lengths)}")
print(f"Mean barcode length: {mean_length:.2f}")
print(f"Median barcode length: {median_length}")
print(f"Min barcode length: {np.min(barcode_lengths)}")
print(f"Max barcode length: {max_length}")

# Create documentation file
doc_filename = f"{output_dir}/somas_barcode_length_distr.md"
with open(doc_filename, 'w') as doc_file:
    doc_file.write(f"# Barcode Length Distribution Analysis\n\n")
    doc_file.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    doc_file.write(f"## Description\n")
    doc_file.write(f"This analysis shows the distribution of barcode lengths (number of positive markers) across {len(barcode_lengths)} soma cells.\n\n")
    doc_file.write(f"## Statistics\n")
    doc_file.write(f"- **Total cells analyzed:** {len(barcode_lengths)}\n")
    doc_file.write(f"- **Mean barcode length:** {mean_length:.2f}\n")
    doc_file.write(f"- **Median barcode length:** {median_length}\n")
    doc_file.write(f"- **Minimum barcode length:** {np.min(barcode_lengths)}\n")
    doc_file.write(f"- **Maximum barcode length:** {max_length}\n\n")
    doc_file.write(f"## Files\n")
    doc_file.write(f"- Plot image: `{os.path.basename(plot_filename)}`\n\n")
    doc_file.write(f"## Methods\n")
    doc_file.write(f"The barcode length for each cell was calculated by summing the number of positive markers (1s) in each cell's barcode array.\n")
    doc_file.write(f"The distribution was visualized using a histogram with discrete bins for each possible barcode length (0-18).\n")

print(f"Plot saved to: {plot_filename}")
print(f"Documentation saved to: {doc_filename}")
