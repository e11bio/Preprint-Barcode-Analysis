# load the data
import logging
import os
import textwrap

import matplotlib.pyplot as plt
import monte_carlo_interpretation as mc
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext, LogLocator

# Import plotting settings
from plot_settings import (
    DPI,
    PlotStyle,
    apply_style,
    get_script_output_dir,
    set_plot_style,
)

# set up logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_fraction_unique_cells(
    df,
    sort_order,
    legend_labels,
    n_somas,
    fraction_unique_cells,
    output_dir,
    dpi,
    plot_sizes,
    experimental_point_style,
    mc,  # keep this global dependency explicit
):
    """Plot fraction of unique barcodes vs number of cells."""
    fig, ax = plt.subplots(figsize=plot_sizes["fraction_unique_cells"])
    df_plot = df.copy()

    # sort for consistent legend ordering
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
        errorbar=("pi", 95),
        ax=ax,
        legend=False,
    )

    # log scale + clean ticks
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Number of cells modeled")
    ax.set_ylabel("Fraction of cells with a unique barcode")

    # reference line + data points
    ax.hlines(
        0.5,
        df["num_samples"].min(),
        df["num_samples"].max(),
        color="black",
        linestyle="--",
        linewidth=0.5,
        label="$SR_{50}$",
    )
    # remove the legend for the experimental data
    experimental_point_style["label"] = None
    ax.scatter(n_somas, fraction_unique_cells, **experimental_point_style)

    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fraction_unique_cells.pdf")
    fig.savefig(save_path, dpi=dpi)
    print(f"Fraction unique cells plot saved to {save_path}")
    # logging.INFO(f"Fraction unique cells plot saved to {output_dir}")


def plot_unique_count(
    df,
    sort_order,
    legend_labels,
    n_somas,
    n_unique_codes,
    output_dir,
    dpi,
    plot_sizes,
    experimental_point_style,
    mc,
):
    """Plot number of unique codes vs number of cells."""
    fig, ax = plt.subplots(figsize=plot_sizes["unique_count"])
    df_plot = df.copy()

    # sort for consistent legend ordering
    df_plot["sort_key"] = df_plot["type"].apply(
        lambda x: mc.get_sort_key(x, sort_order)
    )
    df_plot = df_plot.sort_values("sort_key").drop("sort_key", axis=1)
    df_plot["type"] = df_plot["type"].map(legend_labels).fillna(df_plot["type"])

    sns.lineplot(
        data=df_plot,
        x="num_samples",
        y="unique_count",
        hue="type",
        errorbar=("pi", 95),
        ax=ax,
    )

    # log scale + clean ticks
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Number of cells modeled")
    ax.set_ylabel("Number of codes generated")

    # experimental data
    ax.scatter(n_somas, n_unique_codes, **experimental_point_style)

    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)

    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "unique_count.pdf")
    fig.savefig(save_path, dpi=dpi)
    print(f"Unique count plot saved to {save_path}")


def create_singlet_rate_table(metrics, legend_labels, output_path, plot_sizes, dpi):
    """
    Create and save a singlet rate table figure.
    Rows are included only if all *_mean columns are non-NaN.
    SD columns are dynamically paired with mean columns.
    Column labels are automatically derived from *_mean column names.
    """
    # Identify mean and corresponding sd columns
    mean_cols = sorted([c for c in metrics.columns if c.endswith("_mean")])
    sd_cols = [c.replace("_mean", "_sd") for c in mean_cols]

    # Collect rows
    table_data = []
    for _, row in metrics.iterrows():
        if row[mean_cols].notna().all():
            display_type = legend_labels.get(row["type"], row["type"])
            row_data = [display_type]  # first column
            for m_col, s_col in zip(mean_cols, sd_cols):
                row_data.extend([row[m_col], row[s_col]])
            table_data.append(row_data)

    if not table_data:
        return None  # Nothing to plot

    # Format table: first col = type, then mean±sd for each
    formatted_table_data = []
    for row in table_data:
        type_name = textwrap.fill(row[0], width=8)
        formatted_values = [
            mc.format_mean_sd(mean, sd, compact=True)
            for mean, sd in zip(row[1::2], row[2::2])
        ]
        formatted_table_data.append([type_name] + formatted_values)

    # Build column labels dynamically from *_mean columns
    col_labels = ["Type"]
    for m_col in mean_cols:
        # Extract numeric part from column name if possible
        label_num = "".join([c for c in m_col if c.isdigit()])
        col_labels.append(f"$SR_{{{label_num}}}$" if label_num else m_col)

    # Build figure
    fig_table, ax_table = plt.subplots(figsize=plot_sizes["singlet_rate_table"])
    ax_table.axis("tight")
    ax_table.axis("off")

    table = ax_table.table(
        cellText=formatted_table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Smart column widths
    col_lengths = [
        mc.get_max_content_length(j, formatted_table_data)
        for j in range(len(col_labels))
    ]
    total_length = sum(col_lengths)
    min_width, max_width = 0.15, 0.5
    base_widths = [
        max(min_width, min(max_width, length / total_length)) for length in col_lengths
    ]
    base_widths[0] = max(0.25, base_widths[0])  # widen first col
    width_sum = sum(base_widths)
    target_sum = 1.2
    widths = [w * target_sum / width_sum for w in base_widths]

    for i in range(len(formatted_table_data) + 1):  # +1 for header
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#E6E6E6")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.5)
            cell.set_width(widths[j])

    save_path = os.path.join(output_path, "singlet_rate_table.pdf")
    fig_table.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig_table)
    print(f"Singlet rate table saved to {save_path}")
    # return output_path


def plot_summary_table(
    df,
    legend_labels=None,
    sort_order=None,
    output_dir=".",
    plot_sizes=None,
    dpi=300,
    cell_thresholds=None,
    font_size=8,
):
    """
    Generate a summary table as its own figure showing number of unique barcodes
    at various sample sizes with saturation indicated by "–".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'type', 'num_samples', 'length', 'unique_count'.
    legend_labels : dict, optional
        Mapping from type to display name.
    sort_order : list, optional
        Custom sort order for types.
    output_dir : str
        Directory to save the table PDF.
    plot_sizes : dict
        Dictionary with key "summary_table" for figure size (e.g., (6,4)).
    dpi : int
        Figure DPI.
    cell_thresholds : list of int
        Cell counts to display as columns. Defaults to 10, 100, ..., 1e7.
    font_size : int
        Font size for table.
    """
    if legend_labels is None:
        legend_labels = {}
    if sort_order is None:
        sort_order = df["type"].unique().tolist()
    if plot_sizes is None:
        plot_sizes = {"summary_table": (6, 4)}
    if cell_thresholds is None:
        cell_thresholds = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    # Sort types
    available_types = df["type"].unique().tolist()
    type_order_with_sort_key = [
        (t, mc.get_sort_key(t, sort_order)) for t in available_types
    ]
    type_order_with_sort_key.sort(key=lambda x: x[1])
    type_order = [t[0] for t in type_order_with_sort_key]

    # Format header in 10^n notation
    power_map = {10**i: rf"$\mathrm{{10}}^{i}$" for i in range(1, 8)}
    header_row = ["Type"] + [power_map.get(th, str(th)) for th in cell_thresholds]

    # Build table data
    summary_table_data = []
    for barcode_type in type_order:
        mapped_type_name = legend_labels.get(barcode_type, barcode_type)
        wrapped_type_name = textwrap.fill(mapped_type_name, width=8)
        row_data = [wrapped_type_name]

        # Theoretical max
        type_rows = df[df["type"] == barcode_type]
        type_length = type_rows["length"].iloc[0] if not type_rows.empty else None
        theoretical_max = (
            (2**type_length - 1) if type_length is not None else float("inf")
        )
        saturated = False

        for threshold in cell_thresholds:
            if saturated:
                row_data.append("–")
                continue

            type_data = type_rows[type_rows["num_samples"] == threshold]
            if not type_data.empty:
                mean_codes = type_data["unique_count"].mean()
                sd_codes = type_data["unique_count"].std()

                if mean_codes >= theoretical_max * 0.99:
                    # Show the maximum value this time, but mark as saturated for next columns
                    row_data.append(
                        mc.format_mean_sd(mean_codes, sd_codes, compact=True)
                    )
                    saturated = True  # Mark saturated for subsequent columns
                else:
                    row_data.append(
                        mc.format_mean_sd(mean_codes, sd_codes, compact=True)
                    )
            else:
                row_data.append("–")

        summary_table_data.append(row_data)

    # Create figure and table
    fig, ax = plt.subplots(figsize=plot_sizes["summary_table"])
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=summary_table_data,
        colLabels=header_row,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 2)

    # Smart column widths based on formatted strings
    def get_max_content_length_summary(col_index):
        # Header
        header_len = max(len(line) for line in header_row[col_index].split("\n"))
        # Data content
        content_len = max(
            len(str(row[col_index]).replace("\n", "")) for row in summary_table_data
        )
        return max(header_len, content_len)

    col_lengths_summary = [
        get_max_content_length_summary(j) for j in range(len(header_row))
    ]
    total_length_summary = sum(col_lengths_summary)
    min_width, max_width = 0.03, 0.18
    base_widths_summary = [
        max(min_width, min(max_width, length / total_length_summary))
        for length in col_lengths_summary
    ]
    base_widths_summary[0] = max(0.15, min(0.35, base_widths_summary[0]))
    width_sum_summary = sum(base_widths_summary)
    target_sum_summary = 1.1
    widths_summary = [
        w * target_sum_summary / width_sum_summary for w in base_widths_summary
    ]

    # Apply widths and style
    for i in range(len(summary_table_data) + 1):
        for j in range(len(header_row)):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#E6E6E6")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.5)
            cell.set_width(widths_summary[j])

    # Save figure
    save_path = os.path.join(output_dir, "summary_table.pdf")
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches=None,
    )
    plt.close(fig)
    print(f"Summary table saved to {save_path}")


# --------------------------------#
# where to save the data and how big the plots are
# Define plot sizes
PLOT_SIZES = {
    "fraction_unique_cells": (3, 2.5),
    "unique_count": (3, 2.5),
    "singlet_rate_table": (2.5, 2.6),
    "summary_table": (4, 2.4),
}

# Define experimental data point style
EXPERIMENTAL_POINT_STYLE = {
    "color": "black",
    "s": 80,
    "zorder": 15,
    "marker": (4, 1, 0),  # "",  # Diamond shape for better visibility
    "edgecolors": "white",  # White outline for contrast
    "linewidth": 0.5,  # Thicker outline
    "label": "Soma sample",
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

# Data-fit 18ch ↔ Poisson-modeled 18ch

legend_labels = {
    "simulation - E11 - ideal binary": "Poisson 18 bit",
    "simulation - Brainbow - ideal binary": "Poisson 3 bit",
    "simulation - true soma data": "Data-fit 18 bit",
    "simulation - Tetbow - ideal binary": "Poisson 7 bit",
}

# Define custom sort order
sort_order = [
    "simulation - true soma data",
    "simulation - E11 - ideal binary",
    "simulation - Tetbow - ideal binary",
    "simulation - Brainbow - ideal binary",
]


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
# --------------------------------#
# now we will do some clean up functions...


def add_missing_max_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all types have a datapoint at the maximum num_samples.
    If missing, add a row at max_n_samples with the theoretical maximum
    barcode count for that type.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['num_samples', 'iteration', 'unique_count',
        'unique_fraction', 'length', 'type'].

    Returns
    -------
    pd.DataFrame
        Original df with missing max points added.
    """
    max_n_samples = df["num_samples"].max()

    types_at_max = df[df["num_samples"] == max_n_samples]["type"].unique()
    all_types = df["type"].unique()
    missing_types = set(all_types) - set(types_at_max)

    if not missing_types:
        return df  # nothing to do

    missing_data = []
    for missing_type in missing_types:
        type_length = df.loc[df["type"] == missing_type, "length"].iloc[0]
        theoretical_max = (2**type_length) - 1

        missing_data.append(
            {
                "num_samples": max_n_samples,
                "iteration": 0,
                "unique_count": theoretical_max,
                "unique_fraction": theoretical_max / max_n_samples,
                "length": type_length,
                "type": missing_type,
                "cells_appearing_once": theoretical_max,
            }
        )

    return pd.concat([df, pd.DataFrame(missing_data)], ignore_index=True)


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

    # Ensure all types have data at the maximum n_samples
    df = add_missing_max_points(df)

    plot_fraction_unique_cells(
        df=df,
        sort_order=sort_order,
        legend_labels=legend_labels,
        n_somas=n_somas,
        fraction_unique_cells=fraction_unique_cells,
        output_dir=OUTPUT_DIR,
        dpi=DPI,
        plot_sizes=PLOT_SIZES,
        experimental_point_style=EXPERIMENTAL_POINT_STYLE,
        mc=mc,
    )

    plot_unique_count(
        df=df,
        sort_order=sort_order,
        legend_labels=legend_labels,
        n_somas=n_somas,
        n_unique_codes=n_unique_codes,
        output_dir=OUTPUT_DIR,
        dpi=DPI,
        plot_sizes=PLOT_SIZES,
        experimental_point_style=EXPERIMENTAL_POINT_STYLE,
        mc=mc,
    )

    # Save metrics to markdown
    mc.save_metrics_to_markdown(metrics, OUTPUT_DIR, df)

    create_singlet_rate_table(
        metrics=metrics,
        legend_labels=legend_labels,
        output_path=OUTPUT_DIR,
        plot_sizes=PLOT_SIZES,
        dpi=DPI,
    )
    # -------------------------------- #
    # Create summary table as its own image

    plot_summary_table(
        df=df,
        legend_labels=legend_labels,
        sort_order=sort_order,
        output_dir=OUTPUT_DIR,
        plot_sizes=PLOT_SIZES,
        dpi=DPI,
        cell_thresholds=[10, 100, 1000, 10000, 100000, 1000000],
        font_size=8,
    )

    # cell_thresholds = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    # # Use the same sorting as the SR table
    # available_types = df["type"].unique().tolist()
    # type_order_with_sort_key = [
    #     (t, mc.get_sort_key(t, sort_order)) for t in available_types
    # ]
    # type_order_with_sort_key.sort(key=lambda x: x[1])  # Sort by sort key
    # type_order = [t[0] for t in type_order_with_sort_key]  # Extract just the type names

    # summary_table_data = []

    # # Create header row with 10^n notation for cell counts

    # # Convert cell thresholds to 10^n notation
    # def format_cell_count_header(count):
    #     # Use a cleaner approach with a dictionary lookup
    #     power_map = {
    #         10: r"$10^1$",
    #         100: r"$10^2$",
    #         1000: r"$10^3$",
    #         10000: r"$10^4$",
    #         100000: r"$10^5$",
    #         1000000: r"$10^6$",
    #         10000000: r"$10^7$",
    #     }
    #     return power_map.get(count, str(count))

    # # Swapped layout: Types as rows, Cell counts as columns
    # cell_count_headers = [
    #     format_cell_count_header(threshold) for threshold in cell_thresholds
    # ]
    # wrapped_header_row = ["Type"] + cell_count_headers
    # header_row = ["Type"] + [
    #     str(threshold) for threshold in cell_thresholds
    # ]  # Keep original for data processing

    # # Create data rows - one row per barcode type
    # for barcode_type in type_order:
    #     # Apply legend mapping and text wrapping to type name
    #     mapped_type_name = legend_labels.get(barcode_type, barcode_type)
    #     wrapped_type_name = textwrap.fill(mapped_type_name, width=15)
    #     row_data = [wrapped_type_name]

    #     # Calculate theoretical maximum for this barcode type
    #     type_length = (
    #         df[df["type"] == barcode_type]["length"].iloc[0]
    #         if not df[df["type"] == barcode_type].empty
    #         else None
    #     )
    #     theoretical_max = (
    #         (2**type_length) - 1 if type_length is not None else float("inf")
    #     )

    #     saturated = False  # Track if we've hit saturation

    #     for threshold in cell_thresholds:
    #         if saturated:
    #             # Once saturated, all subsequent columns get "–"
    #             row_data.append("–")
    #             continue

    #         # Filter data for this type and cell count
    #         type_data = df[
    #             (df["type"] == barcode_type) & (df["num_samples"] == threshold)
    #         ]

    #         if not type_data.empty:
    #             mean_codes = type_data["unique_count"].mean()
    #             sd_codes = type_data["unique_count"].std()

    #             # Check if we've reached theoretical saturation (within 1% tolerance)
    #             if mean_codes >= theoretical_max * 0.99:
    #                 saturated = True
    #                 row_data.append("–")  # Use dash to indicate saturation
    #             else:
    #                 formatted_value = mc.format_mean_sd(
    #                     mean_codes, sd_codes, compact=True
    #                 )
    #                 row_data.append(formatted_value)
    #         else:
    #             row_data.append("–")

    #     summary_table_data.append(row_data)

    # if summary_table_data:
    #     fig_summary, ax_summary = plt.subplots(figsize=PLOT_SIZES["summary_table"])
    #     ax_summary.axis("tight")
    #     ax_summary.axis("off")

    #     table_summary = ax_summary.table(
    #         cellText=summary_table_data,
    #         colLabels=wrapped_header_row,
    #         cellLoc="center",
    #         loc="center",
    #     )
    #     table_summary.auto_set_font_size(False)
    #     table_summary.set_fontsize(8)  # Keep font size at 8
    #     table_summary.scale(1, 2)  # Restore vertical scaling for readability

    #     # Smart column sizing for summary table
    #     def get_max_content_length_summary(col_index):
    #         """Get the maximum character length for a column in summary table"""
    #         # Use the wrapped header for width calculation
    #         header_lines = wrapped_header_row[col_index].split("\n")
    #         max_len = max(
    #             len(line) for line in header_lines
    #         )  # Longest line in wrapped header
    #         for row in summary_table_data:
    #             # Handle multi-line content (like compact mean±SD format)
    #             content_lines = str(row[col_index]).split("\n")
    #             content_max_len = max(len(line) for line in content_lines)
    #             max_len = max(max_len, content_max_len)
    #         return max_len

    #     col_lengths_summary = [
    #         get_max_content_length_summary(j) for j in range(len(header_row))
    #     ]
    #     total_length_summary = sum(col_lengths_summary)

    #     # Set minimum and maximum widths - more aggressive for 3-inch constraint
    #     min_width, max_width = 0.03, 0.18  # Much smaller max width
    #     base_widths_summary = [
    #         max(min_width, min(max_width, length / total_length_summary))
    #         for length in col_lengths_summary
    #     ]

    #     # Ensure first column (Type) has more breathing room for text
    #     base_widths_summary[0] = max(
    #         0.15, min(0.35, base_widths_summary[0])
    #     )  # More room for type names

    #     # Make the numeric columns (columns 1-5) even narrower
    #     for i in range(1, len(base_widths_summary)):
    #         base_widths_summary[i] = min(
    #             0.12, base_widths_summary[i]
    #         )  # Cap numeric columns at 0.12

    #     # Normalize to fit within 3-inch constraint with 8pt font
    #     width_sum_summary = sum(base_widths_summary)
    #     target_sum_summary = 1.1  # Slightly more room for 8pt font and line breaks
    #     widths_summary = [
    #         w * target_sum_summary / width_sum_summary for w in base_widths_summary
    #     ]

    #     for i in range(len(summary_table_data) + 1):  # +1 for header
    #         for j in range(len(header_row)):
    #             cell = table_summary[(i, j)]
    #             if i == 0:  # Header row
    #                 cell.set_facecolor("#E6E6E6")
    #                 cell.set_text_props(weight="bold")
    #             else:
    #                 cell.set_facecolor("white")
    #             cell.set_edgecolor("black")
    #             cell.set_linewidth(0.5)

    #             # Apply smart column widths
    #             cell.set_width(widths_summary[j])

    #     fig_summary.savefig(
    #         os.path.join(OUTPUT_DIR, "summary_table.pdf"),
    #         dpi=DPI,
    #         bbox_inches=None,  # Respect the original figsize
    #     )
    #     plt.close(fig_summary)

    # plt.show()
