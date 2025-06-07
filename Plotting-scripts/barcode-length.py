# This script generates a plot for barcode distribution given a numpy array of binary barcodes.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from soma_preprocessing import generate_barcode_array

# Import plotting settings
from plot_settings import (
    MAIN_COLOR,
    FIG_SIZE_HISTOGRAM_barcode_lengths,
    set_style,
    DPI,
    OUTPUT_DIR,
)

FIG_SIZE = FIG_SIZE_HISTOGRAM_barcode_lengths
# Import functions from soma-preprocessing.py

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()

# load the soma_barcode_info.csv file

# creating a plot of barcode length
set_style()
sns.set_palette(palette="Greys")
sns.set_style("ticks")
barcode_lengths = np.sum(soma_barcodes, axis=1)

plt.figure(figsize=FIG_SIZE, dpi=DPI)

ax = sns.histplot(barcode_lengths, kde=False, bins=range(19), discrete=True)
# , color="grey")

# Set the bar color to #1f77b4
for patch in ax.patches:
    patch.set_facecolor("#1f77b4")
    # patch.set_edgecolor('black')
    patch.set_linewidth(0.1)


# sns.despine(left=True, bottom=True)
ax.grid(False)
sns.despine()  # Only remove left, top, and right spines


# Add labels and title
# set the x and y labels to empty
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
# plt.xlabel("Hamming Weight")
# plt.ylabel("Frequency (counts)")
# plt.title(f'Distribution of Barcode Lengths Across {len(barcode_lengths)} Somas')

# Set x-axis ticks to include all possible barcode lengths (0 to 18)
# get the max length
# max_length = np.max(barcode_lengths)
plt.xticks(range(1, 18 + 1, 2))


# Add a grid for better readability
# plt.grid(axis='y', alpha=0.3)

# Calculate some statistics to add as text
mean_length = np.mean(barcode_lengths)
median_length = np.median(barcode_lengths)
max_length = np.max(barcode_lengths)

# Add statistics as text
# stats_text = f"Mean: {mean_length:.2f}"
# plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
#          verticalalignment='top', horizontalalignment='right',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=12)

# Create output directory if it doesn't exist
output_dir = OUTPUT_DIR

# Save the plot
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"{output_dir}/somas_barcode_length_distr.png"

plt.tight_layout()
plt.savefig(plot_filename, dpi=500)
plt.close()

# Also print basic statistics
print(f"Total cells: {len(barcode_lengths)}")
print(f"Mean barcode length: {mean_length:.2f}")
print(f"Median barcode length: {median_length}")
print(f"Min barcode length: {np.min(barcode_lengths)}")
print(f"Max barcode length: {max_length}")

# Create documentation file
doc_filename = f"{output_dir}/somas_barcode_length_distr.md"
with open(doc_filename, "w") as doc_file:
    doc_file.write(f"# Barcode Length Distribution Analysis\n\n")
    doc_file.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    doc_file.write(f"## Description\n")
    doc_file.write(
        f"This analysis shows the distribution of barcode lengths (number of positive markers) across {len(barcode_lengths)} soma cells.\n\n"
    )
    doc_file.write(f"## Statistics\n")
    doc_file.write(f"- **Total cells analyzed:** {len(barcode_lengths)}\n")
    doc_file.write(f"- **Mean barcode length:** {mean_length:.2f}\n")
    doc_file.write(f"- **Median barcode length:** {median_length}\n")
    doc_file.write(f"- **Minimum barcode length:** {np.min(barcode_lengths)}\n")
    doc_file.write(f"- **Maximum barcode length:** {max_length}\n\n")
    doc_file.write(
        f"- **Standard deviation of barcode length:** {np.std(barcode_lengths):.2f}\n\n"
    )
    doc_file.write(f"## Files\n")
    doc_file.write(f"- Plot image: `{os.path.basename(plot_filename)}`\n\n")
    doc_file.write(f"## Methods\n")
    doc_file.write(
        f"The barcode length for each cell was calculated by summing the number of positive markers (1s) in each cell's barcode array.\n"
    )
    doc_file.write(
        f"The distribution was visualized using a histogram with discrete bins for each possible barcode length (0-18).\n"
    )

print(f"Plot saved to: {plot_filename}")
print(f"Documentation saved to: {doc_filename}")
