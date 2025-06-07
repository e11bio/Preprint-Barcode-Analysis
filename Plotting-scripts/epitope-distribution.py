# this script generates a plot for epitope distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy

# Import plotting settings
from plot_settings import (
    MAIN_COLOR,
    FIG_SIZE_HISTOGRAM_epitope_dist,
    set_style,
    DPI,
    OUTPUT_DIR,
)

FIG_SIZE = FIG_SIZE_HISTOGRAM_epitope_dist
# Import functions from soma-preprocessing.py
from soma_preprocessing import generate_barcode_array, target_channels

# sns.color_palette(palette="Greys")

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()

epitope_counts = np.sum(soma_barcodes, axis=0)  # Sum each column
total_cells = len(soma_barcodes)
epitope_percentages = (epitope_counts / total_cells) * 100

# Calculate the average percentage
mean_percentage = np.mean(epitope_percentages)
median_percentage = np.median(epitope_percentages)

# Create a DataFrame for easier plotting with the target names
simplified_epitopes = [name.split("-")[0] for name in target_channels]
epitope_df = pd.DataFrame(
    {"Epitope": simplified_epitopes, "Percentage": epitope_percentages}
)

# Sort from highest to lowest percentage for better visualization
epitope_df = epitope_df.sort_values("Percentage", ascending=True)

# Create the plot
# modify fig size to be
# FIG_SIZE = (3, 2.5)

plt.figure(figsize=FIG_SIZE, dpi=DPI)
set_style()  # Apply the standard plotting style
sns.set_style("ticks")
bars = sns.barplot(
    x="Epitope",
    y="Percentage",
    data=epitope_df,
    color="#1f77b4",
    fill=True,
    # edgecolor="grey",
)

# Remove grid lines
plt.grid(False)
ax = plt.gca()
ax.grid(False)

# Ensure all text is black
ax.title.set_color("black")
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
# for text in ax.get_xticklabels() + ax.get_yticklabels():
#     text.set_color('black')
#     text.set_fontfamily('Arial')
#     text.set_fontsize(8)

sns.despine()
plt.xticks(fontsize=8)

# Add a horizontal line for the mean percentage
mean_line = plt.axhline(y=mean_percentage, color="grey", linestyle="--", alpha=0.5)

# Add a legend for the mean line with dashed line symbol
# plt.legend([mean_line], ['Mean ({:.1f}%)'.format(mean_percentage)], loc='upper left', frameon=False)


# Add percentage labels only for the lowest and highest bars
min_idx = epitope_df["Percentage"].idxmin()
max_idx = epitope_df["Percentage"].idxmax()

for i, p in enumerate(bars.patches):
    if (
        i == 0 or i == len(bars.patches) - 1
    ):  # Only label first (highest) and last (lowest) bars
        # Adjust horizontal position for the first bar (highest) to avoid overlap with y-axis
        h_align = "center"
        x_pos = p.get_x() + p.get_width() / 2.0

        # Push the highest bar's label slightly to the right
        if i == 0:  # This is the highest bar
            x_pos += 0.8  # Shift to the right

        bars.annotate(
            f"{p.get_height():.1f}%", (x_pos, p.get_height()), ha=h_align, va="bottom"
        )

# Add labels and title
# remove the x and y labels
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
# plt.title(f'Percentage of somas containing epitope (n={len(soma_barcodes)})')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha="center")

# set the y axis to 0-40
plt.ylim(0, 50)
# Adjust layout
plt.tight_layout()

output_dir = "./out"
# make the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

plot_filename = os.path.join(output_dir, "epitope_distribution.png")
plt.savefig(plot_filename, dpi=500)


# Create markdown file with context
md_content = f"""# Epitope Distribution Analysis

## Overview
This plot shows the distribution of epitopes across 147 somas in the barcode analysis, sorted in descending order.

## Details
- **Total somas analyzed**: {len(soma_barcodes)}
- **Highest epitope presence**: {epitope_df["Percentage"].max():.1f}%
- **Lowest epitope presence**: {epitope_df["Percentage"].min():.1f}%
- **Mean epitope presence**: {mean_percentage:.1f}%
- **Median epitope presence**: {median_percentage:.1f}%
- **Standard deviation of epitope presence**: {np.std(epitope_percentages):.1f}%

The histogram showcases the distribution of each epitope of the barcode, indicating the percentage of cells containing each epitope.

The values for each epitope are as follows:
{epitope_df.to_markdown()}

![Epitope Distribution](epitope_distribution.png)
"""

md_filename = os.path.join(output_dir, "epitope_distribution.md")
with open(md_filename, "w") as md_file:
    md_file.write(md_content)

print(f"Plot saved to {plot_filename}")
print(f"Markdown documentation saved to {md_filename}")
