# load the data
import os
import textwrap

import matplotlib.pyplot as plt
import monte_carlo_interpretation as mc
import pandas as pd
import seaborn as sns

# Import plotting settings
from plot_settings import (
    DPI,
    PlotStyle,
    apply_style,
    get_script_output_dir,
    set_plot_style,
)

from matplotlib import font_manager as fm

# Use DejaVu Sans for superscripts
dejavu = fm.FontProperties(family="DejaVu Sans")
# --------------------------------#
# where to save the data and how big the plots are
# Define plot sizes
PLOT_SIZES = {
    "fraction_unique_cells": (3, 2.5),
    "unique_count": (3, 2.5),
    "labeling_limit_table": (3, 2.5),
    "summary_table": (3.5, 2.5),
}

# Define experimental data point style
EXPERIMENTAL_POINT_STYLE = {
    "color": "black",
    "s": 80,
    "zorder": 15,
    "marker": (4, 1, 0),  # "",  # Diamond shape for better visibility
    "edgecolors": "white",  # White outline for contrast
    "linewidth": 0.5,  # Thicker outline
    "label": "Observed data",
}

# Define output directory
OUTPUT_DIR = get_script_output_dir("monte_carlo_modeling_plots")

MIN_SD_DISPLAY = 0.5  # SD below this will be considered negligible


# --------------------------------#
# hard coded variables about the dataset
n_somas = 146
n_unique_codes = 130
fraction_unique_cells = 0.80
filter_nums = [2, 5, 3, 8, 15]

# --------------------------------#
# importing the data and doing some light clean up etc

df = pd.read_csv("out/monte_carlo_modeling_data.csv")

df["fraction_unique_cells"] = df["num_objects_appearing_once"] / df["num_samples"]
# filter out samples of low n for readability
df = df[~df["num_samples"].isin(filter_nums)]
# keep only single level data for clean comparaison
df = df[~df["type"].str.contains("2 level")]

# --------------------------------#
# labels and legend info

legend_labels = {
    "simulation - E11 - ideal binary": "Ideal 18ch",
    "simulation - Brainbow - ideal binary": "Ideal 3ch",
    "simulation - true soma data": "Extrapolated 18ch",
    "simulation - Tetbow - ideal binary": "Ideal 7ch",
}

# Define custom sort order
sort_order = [
    "simulation - true soma data",
    "simulation - E11 - ideal binary",
    "simulation - Tetbow - ideal binary",
    "simulation - Brainbow - ideal binary",
]


# --------------------------------#
# calculating the per group maximmus
# getting maximums
# # keep the first occurrence of each type-length combination to preserve order
# unique_combinations = df[["type", "length"]].drop_duplicates()
# counts = (
#     df.groupby(["type", "length"]).size().rename("count")
# )  # count the number of rows per type-length (for capping)
# unique_combinations = unique_combinations.merge(
#     counts, on=["type", "length"]
# )  # merge counts back
# unique_combinations["unique_count"] = (2 ** unique_combinations["length"]) - 1

# # optional: drop 'count' if not needed
# theoretical_values = unique_combinations.drop(columns="count")


# --------------------------------#
# then the ec50 from gpt

# 1) Compute f(n) = mean unique_fraction by method
summ = (
    df.groupby(["type", "length", "num_samples"])
    .agg(mean_frac=("fraction_unique_cells", "mean"))
    .reset_index()
    .sort_values("num_samples")
)


metrics = []
for (t, L), g in df.groupby(["type", "length"]):
    # Get all 4 values from n_at_fraction: n_median, n_iqr, n_mean, n_sd
    n75_median, n75_iqr, n75_mean, n75_sd = mc.n_at_fraction(g, p=0.75)
    n50_median, n50_iqr, n50_mean, n50_sd = mc.n_at_fraction(g, p=0.5)
    n25_median, n25_iqr, n25_mean, n25_sd = mc.n_at_fraction(g, p=0.25)

    # Use summ for auc_log which expects mean_frac column
    g_summ = summ[(summ["type"] == t) & (summ["length"] == L)]
    alog = mc.auc_log(g_summ)

    # Store all statistics for comprehensive analysis
    metrics.append(
        {
            "type": t,
            "length": L,
            "n75_median": n75_median,
            "n75_iqr": n75_iqr,
            "n75_mean": n75_mean,
            "n75_sd": n75_sd,
            "n50_median": n50_median,
            "n50_iqr": n50_iqr,
            "n50_mean": n50_mean,
            "n50_sd": n50_sd,
            "n25_median": n25_median,
            "n25_iqr": n25_iqr,
            "n25_mean": n25_mean,
            "n25_sd": n25_sd,
            "AUC_log": alog,
        }
    )


metrics = pd.DataFrame(metrics)
# Sort metrics by custom order
metrics["sort_key"] = metrics["type"].apply(lambda x: mc.get_sort_key(x, sort_order))
metrics = metrics.sort_values(["sort_key", "length"]).drop("sort_key", axis=1)
# print(metrics)


def configure_plot_style(settings):
    """Configure matplotlib and seaborn styling"""
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.family": settings["font_family"],
            "font.size": settings["tick_size"],
            "axes.labelsize": settings["label_size"],
            "axes.titlesize": settings["label_size"],
            "xtick.labelsize": settings["tick_size"],
            "ytick.labelsize": settings["tick_size"],
            "legend.fontsize": settings["tick_size"],
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


if __name__ == "__main__":
    # Setup
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        choices=["paper", "poster"],
        default="paper",
        help="Plot style to use (paper or poster)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        help="Font size to use for poster mode",
    )
    args = parser.parse_args()

    # Get plot settings
    style = PlotStyle.POSTER if args.style == "poster" else PlotStyle.PAPER
    settings = set_plot_style(style, font_size=args.font_size)

    # Apply settings
    apply_style(settings)
    configure_plot_style(settings)

    # # Ensure all types have data at the maximum n_samples
    # max_n_samples = df["num_samples"].max()

    # # Find types that don't have data at max_n_samples
    # types_at_max = df[df["num_samples"] == max_n_samples]["type"].unique()
    # all_types = df["type"].unique()
    # missing_types = set(all_types) - set(types_at_max)

    # # For missing types, add a data point at max_n_samples using their theoretical maximum
    # if missing_types:
    #     missing_data = []
    #     for missing_type in missing_types:
    #         # Get the length for this type
    #         type_length = df[df["type"] == missing_type]["length"].iloc[0]
    #         theoretical_max = (2**type_length) - 1

    #         # Create a data point at max_n_samples
    #         missing_data.append(
    #             {
    #                 "num_samples": max_n_samples,
    #                 "iteration": 0,
    #                 "unique_count": theoretical_max,
    #                 "unique_fraction": theoretical_max / max_n_samples,
    #                 "length": type_length,
    #                 "type": missing_type,
    #                 "cells_appearing_once": theoretical_max,  # Assuming all are unique at saturation
    #             }
    #         )

    #     # Add missing data to dataframe
    #     missing_df = pd.DataFrame(missing_data)
    #     df = pd.concat([df, missing_df], ignore_index=True)

    # Plot and save figures
    fig, ax = plt.subplots(figsize=PLOT_SIZES["fraction_unique_cells"])
    df_plot = df.copy()
    # Sort dataframe by custom order for consistent legend ordering
    df_plot["sort_key"] = df_plot["type"].apply(
        lambda x: mc.get_sort_key(x, sort_order)
    )
    df_plot = df_plot.sort_values("sort_key").drop("sort_key", axis=1)
    df_plot["type"] = df_plot["type"].map(legend_labels).fillna(df_plot["type"])

    sns.lineplot(
        data=df_plot,
        x="num_samples",
        y="fraction_unique_cells",
        hue="type",
        # style="type",
        errorbar=("pi", 95),
        ax=ax,
        legend=False,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of cells modeled")
    ax.set_ylabel("Fraction of cells with a unique barcode")
    # Set consistent x-axis ticks for both plots
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(
        [r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"]
    )
    # Remove minor ticks to match the second plot
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.hlines(
        0.5, 1, 1000000, color="black", linestyle="--", linewidth=0.5, label="$LL_{50}$"
    )

    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    ax.scatter(n_somas, fraction_unique_cells, **EXPERIMENTAL_POINT_STYLE)

    sns.despine()
    plt.tight_layout()
    # plt.subplots_adjust(right=0.7, bottom=0.2)
    fig.savefig(
        os.path.join(OUTPUT_DIR, "fraction_unique_cells.pdf"),
        dpi=DPI,
        bbox_inches=None,
    )

    # -------------------------------- #
    fig, ax = plt.subplots(figsize=PLOT_SIZES["unique_count"])
    sns.lineplot(
        data=df_plot,
        x="num_samples",
        y="unique_count",
        hue="type",
        # style="type",
        errorbar=("pi", 95),
        ax=ax,
        # legend=False,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of cells modeled")
    ax.set_ylabel("Number of codes generated")
    # Set consistent x-axis ticks for both plots
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(
        [r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"]
    )
    # Add experimental data point
    ax.scatter(n_somas, n_unique_codes, **EXPERIMENTAL_POINT_STYLE)

    # aesthetics
    sns.despine()
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)

    # plt.subplots_adjust(right=0.7, bottom=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "unique_count.pdf"), dpi=DPI, bbox_inches=None)

    # Save metrics to markdown
    mc.save_metrics_to_markdown(metrics, OUTPUT_DIR, df)

    # -------------------------------- #
    # Create a separate table as its own image
    table_data = []
    for _, row in metrics.iterrows():
        if (
            not pd.isna(row["n75_mean"])
            and not pd.isna(row["n50_mean"])
            and not pd.isna(row["n25_mean"])
        ):
            # Apply legend mapping if available
            display_type = legend_labels.get(row["type"], row["type"])
            table_data.append(
                [
                    display_type,
                    row["n75_mean"],
                    row["n50_mean"],
                    row["n25_mean"],
                    row["n75_sd"],
                    row["n50_sd"],
                    row["n25_sd"],
                ]
            )

    if table_data:
        # Format data with mean ± SD style
        formatted_table_data = []
        for row in table_data:
            # Wrap type name
            type_name = textwrap.fill(row[0], width=15)
            n75 = mc.format_mean_sd(
                row[1],
                row[4],
                compact=True,
            )  # median ± IQR format
            n50 = mc.format_mean_sd(
                row[2],
                row[5],
                compact=True,
            )  # median ± IQR format
            n25 = mc.format_mean_sd(row[3], row[6], compact=True)  # median ± IQR format
            formatted_table_data.append([type_name, n75, n50, n25])

        fig_table, ax_table = plt.subplots(figsize=PLOT_SIZES["labeling_limit_table"])
        ax_table.axis("tight")
        ax_table.axis("off")

        table = ax_table.table(
            cellText=formatted_table_data,
            colLabels=["Type", "$LL_{75}$", "$LL_{50}$", "$LL_{25}$"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Smart column sizing based on content

        # Calculate relative column widths
        col_lengths = [
            mc.get_max_content_length(j, formatted_table_data) for j in range(4)
        ]
        total_length = sum(col_lengths)

        # Set minimum and maximum widths
        min_width, max_width = 0.15, 0.5
        base_widths = [
            max(min_width, min(max_width, length / total_length))
            for length in col_lengths
        ]

        # Ensure first column (Type) has reasonable width for text wrapping
        base_widths[0] = max(0.25, base_widths[0])

        # Normalize to sum to reasonable total
        width_sum = sum(base_widths)
        target_sum = 1.2  # Slightly wider table
        widths = [w * target_sum / width_sum for w in base_widths]

        # Style the table
        for i in range(len(formatted_table_data) + 1):  # +1 for header
            for j in range(4):  # 4 columns
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor("#E6E6E6")
                    cell.set_text_props(weight="bold")
                else:
                    cell.set_facecolor("white")
                cell.set_edgecolor("black")
                cell.set_linewidth(0.5)

                # Apply smart column widths
                cell.set_width(widths[j])

        fig_table.savefig(
            os.path.join(OUTPUT_DIR, "labeling_limit_table.pdf"),
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close(fig_table)

    # -------------------------------- #
    # Create summary table as its own image
    cell_thresholds = [10, 100, 1000, 10000, 100000]

    # Use the same sorting as the LL table
    available_types = df["type"].unique().tolist()
    type_order_with_sort_key = [
        (t, mc.get_sort_key(t, sort_order)) for t in available_types
    ]
    type_order_with_sort_key.sort(key=lambda x: x[1])  # Sort by sort key
    type_order = [t[0] for t in type_order_with_sort_key]  # Extract just the type names

    summary_table_data = []

    # Create header row with 10^n notation for cell counts
    import textwrap

    # Convert cell thresholds to 10^n notation
    def format_cell_count_header(count):
        if count == 10:
            return r"$10^1$"
        elif count == 100:
            return r"$10^2$"
        elif count == 1000:
            return r"$10^3$"
        elif count == 10000:
            return r"$10^4$"
        elif count == 100000:
            return r"$10^5$"
        else:
            return str(count)

    # Swapped layout: Types as rows, Cell counts as columns
    cell_count_headers = [
        format_cell_count_header(threshold) for threshold in cell_thresholds
    ]
    wrapped_header_row = ["Type"] + cell_count_headers
    header_row = ["Type"] + [
        str(threshold) for threshold in cell_thresholds
    ]  # Keep original for data processing

    # Create data rows - one row per barcode type
    for barcode_type in type_order:
        # Apply legend mapping and text wrapping to type name
        mapped_type_name = legend_labels.get(barcode_type, barcode_type)
        wrapped_type_name = textwrap.fill(mapped_type_name, width=15)
        row_data = [wrapped_type_name]

        for threshold in cell_thresholds:
            # Filter data for this type and cell count
            type_data = df[
                (df["type"] == barcode_type) & (df["num_samples"] == threshold)
            ]

            if not type_data.empty:
                mean_codes = type_data["unique_count"].mean()
                sd_codes = type_data["unique_count"].std()
                formatted_value = mc.format_mean_sd(mean_codes, sd_codes, compact=True)
                row_data.append(formatted_value)
            else:
                row_data.append("–")

        summary_table_data.append(row_data)

    if summary_table_data:
        fig_summary, ax_summary = plt.subplots(figsize=PLOT_SIZES["summary_table"])
        ax_summary.axis("tight")
        ax_summary.axis("off")

        table_summary = ax_summary.table(
            cellText=summary_table_data,
            colLabels=wrapped_header_row,
            cellLoc="center",
            loc="center",
        )
        table_summary.auto_set_font_size(False)
        table_summary.set_fontsize(8)  # Keep font size at 8
        table_summary.scale(1, 2)  # Restore vertical scaling for readability

        # Smart column sizing for summary table
        def get_max_content_length_summary(col_index):
            """Get the maximum character length for a column in summary table"""
            # Use the wrapped header for width calculation
            header_lines = wrapped_header_row[col_index].split("\n")
            max_len = max(
                len(line) for line in header_lines
            )  # Longest line in wrapped header
            for row in summary_table_data:
                # Handle multi-line content (like compact mean±SD format)
                content_lines = str(row[col_index]).split("\n")
                content_max_len = max(len(line) for line in content_lines)
                max_len = max(max_len, content_max_len)
            return max_len

        col_lengths_summary = [
            get_max_content_length_summary(j) for j in range(len(header_row))
        ]
        total_length_summary = sum(col_lengths_summary)

        # Set minimum and maximum widths - more aggressive for 3-inch constraint
        min_width, max_width = 0.03, 0.18  # Much smaller max width
        base_widths_summary = [
            max(min_width, min(max_width, length / total_length_summary))
            for length in col_lengths_summary
        ]

        # Ensure first column (Type) has more breathing room for text
        base_widths_summary[0] = max(
            0.15, min(0.35, base_widths_summary[0])
        )  # More room for type names

        # Make the numeric columns (columns 1-5) even narrower
        for i in range(1, len(base_widths_summary)):
            base_widths_summary[i] = min(
                0.12, base_widths_summary[i]
            )  # Cap numeric columns at 0.12

        # Normalize to fit within 3-inch constraint with 8pt font
        width_sum_summary = sum(base_widths_summary)
        target_sum_summary = 1.1  # Slightly more room for 8pt font and line breaks
        widths_summary = [
            w * target_sum_summary / width_sum_summary for w in base_widths_summary
        ]

        for i in range(len(summary_table_data) + 1):  # +1 for header
            for j in range(len(header_row)):
                cell = table_summary[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor("#E6E6E6")
                    cell.set_text_props(weight="bold")
                else:
                    cell.set_facecolor("white")
                cell.set_edgecolor("black")
                cell.set_linewidth(0.5)

                # Apply smart column widths
                cell.set_width(widths_summary[j])

        fig_summary.savefig(
            os.path.join(OUTPUT_DIR, "summary_table.pdf"),
            dpi=DPI,
            bbox_inches=None,  # Respect the original figsize
        )
        plt.close(fig_summary)

    # plt.show()
