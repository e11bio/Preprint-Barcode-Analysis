import concurrent.futures
import csv
import gc
import logging
import os
from collections import Counter
from multiprocessing import Manager

# import matplotlib.pyplot as plt
import numpy as np
from binary_analyses import load_barcode_data
from scipy.stats import poisson
from soma_preprocessing import generate_barcode_array
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

desired_number_of_workers = 6

# Custom mapping for epitope names for Sanity reasons.
EPITOPE_NAMES = {
    "E2-barcode-R1": "E2",
    "S1-barcode-R1": "S1",
    "ALFA-barcode-R1": "ALFA",
    "Ty1-barcode-R2": "Ty1",
    "HA-barcode-R3": "HA",
    "T7-barcode-R6": "T7",
    "VSVG-barcode-R6": "VSVG",
    "AU5-barcode-R8": "AU5",
    "NWS-barcode-R9": "NWS",
    "SunTag-barcode-R9": "SUN",
    "ETAG-barcode-R9": "ETAG",
    "SPOT-barcode-R10": "SPOT",
    "MoonTag-barcode-R10": "MOON",
    "HSV Tag-barcode-R10": "HSV",
    "Protein C-barcode-R11": "PRTC",
    "Tag100-barcode-R11": "TG100",
    "CMyc-barcode-R11": "MYC",
    "OLLAS-barcode-R12": "OLLAS",
}
# these are pulled from the epitope distribution work we did previously. obviously could be un-hardcoded.
epitope_percentages = {
    "E2": 14.2857,
    "AU5": 14.966,
    "S1": 16.3265,
    "Ty1": 17.0068,
    "SPOT": 17.6871,
    "ALFA": 18.3673,
    "VSVG": 19.0476,
    "HA": 19.7279,
    "SunTag": 20.4082,
    "OLLAS": 21.7687,
    "Myc": 22.449,
    "HSV": 23.1293,
    "Tag100": 25.1701,
    "NWS": 25.1701,
    "T7": 27.2109,
    "Protein-C": 27.8912,
    "ETAG": 28.5714,
    "MOON": 36.0544,
}


def duplicate_probabilities(original_probs):
    extended_probs = {}
    for key, prob in original_probs.items():
        # Create two entries for each original key
        extended_probs[f"{key}_1"] = prob / 2
        extended_probs[f"{key}_2"] = prob / 2
    return extended_probs


def convert_percents(epitope_dict):
    """Convert raw percentages to probabilities that sum to 1."""
    vals = np.array(list(epitope_dict.values()), dtype=float)
    return vals / vals.sum()


def generate_random_barcode(hamming_weight, length=18, p=None):
    """
    Generate a binary barcode of given length and Hamming weight.
    If `p` is provided, it should be a probability vector of length `length`.
    """
    barcode = np.zeros(length, dtype=np.uint8)
    if p is not None:
        indices = np.random.choice(length, hamming_weight, replace=False, p=p)
    else:
        indices = np.random.choice(length, hamming_weight, replace=False)
    barcode[indices] = 1
    return barcode


def monte_carlo_generate_cells(
    hamming_weight_distribution, num_samples=1000, length=18, epitope_percentages=None
):
    """
    Simulate many barcodes:
      - sample Hamming weights from a distribution
      - assign epitopes based on probabilities
    """
    # normalize hamming weight distribution
    weights = np.array(list(hamming_weight_distribution.keys()), dtype=np.uint8)
    probabilities = np.array(list(hamming_weight_distribution.values()), dtype=float)
    probabilities /= probabilities.sum()

    # normalize epitope percentages if provided
    epitope_probs = None
    if epitope_percentages is not None:
        epitope_probs = convert_percents(epitope_percentages)

    # generate many barcodes
    return np.array(
        [
            generate_random_barcode(
                np.random.choice(weights, p=probabilities),
                length,
                p=epitope_probs,
            )
            for _ in range(num_samples)
        ]
    )


def count_unique(barcodes):
    """
    Count number of unique barcodes (rows) in the barcode matrix.
    """
    return np.unique(barcodes, axis=0, return_counts=True)


def run_simulation(
    hamming_weight_distribution,
    sample_size,
    i,
    length,
    epitope_percentages,
    kind,
    progress_list,
):
    logging.debug(f"Running simulation for sample size {sample_size}, iteration {i}")
    new_cells = monte_carlo_generate_cells(
        hamming_weight_distribution,
        num_samples=sample_size,
        length=length,
        epitope_percentages=epitope_percentages,
    )
    unique_elements, counts = count_unique(new_cells)
    num_items_appearing_once = unique_elements[counts == 1].shape[0]
    num_codes = unique_elements.shape[0]

    record = {
        "num_samples": sample_size,
        "iteration": i,
        "cells_appearing_once": num_items_appearing_once,
        "unique_count": num_codes,
        "unique_fraction": num_items_appearing_once / sample_size,
        "length": length,
        "type": kind,
    }
    logging.debug(f"Record created: {record}")
    del new_cells
    gc.collect()
    progress_list.append(1)  # Update progress
    return record


def get_existing_iteration_counts(csv_file, kind, num_samples_list):
    """Check existing CSV file and return counts of iterations for each sample size."""
    if not os.path.exists(csv_file):
        return {sample_size: 0 for sample_size in num_samples_list}

    import pandas as pd

    try:
        df = pd.read_csv(csv_file)
        # Filter for this specific type
        type_data = df[df["type"] == kind]
        # Count iterations for each sample size
        counts = type_data.groupby("num_samples").size().to_dict()
        # Return counts, defaulting to 0 for missing sample sizes
        return {
            sample_size: counts.get(sample_size, 0) for sample_size in num_samples_list
        }
    except Exception as e:
        logging.warning(f"Could not read existing CSV file: {e}")
        return {sample_size: 0 for sample_size in num_samples_list}


def run_monte_carlo_model(
    hamming_weight_distribution,
    num_samples,
    iterations,
    length=18,
    epitope_percentages=None,
    kind=None,
):
    logging.debug("Starting Monte Carlo model")

    # Check existing iterations and adjust accordingly
    csv_file = "./out/monte_carlo_modeling_data.csv"
    existing_counts = get_existing_iteration_counts(csv_file, kind, num_samples)
    logging.info(f"Existing iteration counts for {kind}: {existing_counts}")

    with open(csv_file, mode="a", newline="") as csvfile:
        logging.debug("Opening CSV file")
        fieldnames = [
            "num_samples",
            "iteration",
            "unique_count",
            "unique_fraction",
            "length",
            "type",
            "cells_appearing_once",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is empty or doesn't exist
        if (
            not os.path.exists("./out/monte_carlo_modeling_data.csv")
            or os.stat("./out/monte_carlo_modeling_data.csv").st_size == 0
        ):
            writer.writeheader()
        if epitope_percentages is not None:
            logging.debug(f"Length of epitope_percentages: {len(epitope_percentages)}")
        else:
            logging.debug("No epitope percentages provided")
        logging.debug(
            f"Length of hamming_weight_distribution: {len(hamming_weight_distribution)}"
        )

        # Verify the sizes of arrays and probability distributions
        for sample_size in tqdm(num_samples, desc="Simulating cell ranges"):
            # Calculate how many more iterations are needed for this sample size
            existing_count = existing_counts.get(sample_size, 0)
            remaining_iterations = max(0, iterations - existing_count)

            if remaining_iterations == 0:
                logging.info(
                    f"Sample size {sample_size} already has {existing_count} iterations, skipping"
                )
                continue

            logging.info(
                f"Processing sample size: {sample_size}, running {remaining_iterations} additional iterations (existing: {existing_count})"
            )

            with Manager() as manager:
                progress_list = manager.list()
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=desired_number_of_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            run_simulation,
                            hamming_weight_distribution,
                            sample_size,
                            existing_count
                            + i,  # Continue numbering from existing count
                            length,
                            epitope_percentages,
                            kind,
                            progress_list,
                        )
                        for i in range(remaining_iterations)
                    ]
                    logging.debug(f"Submitted {len(futures)} futures")
                    with tqdm(total=remaining_iterations, desc="Iterations") as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                                logging.debug(f"Future completed with result: {result}")
                                writer.writerow(result)
                                csvfile.flush()  # Ensure data is written to disk
                                logging.debug(f"Written to CSV: {result}")
                                pbar.update(len(progress_list))
                                progress_list[:] = []  # Clear the list after updating
                            except Exception as e:
                                logging.error(f"Error processing future: {e}")


def trim_barcodes(barcodes, new_length, seed=None):
    """
    Trim the length of barcodes by sampling a set of channels.

    Parameters:
    - barcodes: np.ndarray, original array of barcodes (2D array)
    - new_length: int, desired length of the trimmed barcodes

    Returns:
    - np.ndarray, new array of barcodes with the specified length
    """
    original_length = barcodes.shape[1]
    if new_length > original_length:
        raise ValueError(
            "New length must be less than or equal to the original length."
        )

    # Randomly select indices to keep
    selected_indices = np.random.choice(original_length, new_length, replace=False)

    # Trim the barcodes
    trimmed_barcodes = barcodes[:, selected_indices]

    return trimmed_barcodes


def compute_hamming_weight_probabilities(barcodes):
    """
    Generate a barcode array and compute the Hamming weight distribution
    as probabilities.

    Parameters
    ----------
    generate_barcode_array : callable
        Function that returns an array of barcodes, shape (n_cells, n_channels)

    Returns
    -------
    dict
        Keys are Hamming weights, values are probabilities
    """
    # Step 1: generate barcodes

    # Step 2: compute Hamming weights
    hamming_weights = np.sum(barcodes, axis=1)

    # Step 3: count occurrences
    weight_counts = Counter(hamming_weights)

    # Step 4: convert to probabilities
    total = sum(weight_counts.values())
    weight_probabilities = {
        weight: count / total for weight, count in weight_counts.items()
    }

    return weight_probabilities


def compute_epitope_probabilities(barcodes):
    """
    Compute per-epitope probabilities (frequency of 1's per channel).

    Parameters
    ----------
    barcodes : np.ndarray
        Array of shape (n_cells, n_channels), with 0/1 entries.

    Returns
    -------
    dict
        Keys = channel index (or epitope name if provided),
        Values = probabilities (fraction of cells where epitope is present).
    """
    n_cells, n_channels = barcodes.shape

    # mean along rows gives fraction of times each channel is "on"
    freqs = np.mean(barcodes, axis=0)

    return {i: freqs[i] for i in range(n_channels)}


def generate_perfect_distributions(lam, k_values):
    probs = poisson.pmf(k_values, mu=lam)
    return {k: float(p) for k, p in zip(k_values, probs)}  # noqa: B905


## --- import the hand calls ---
soma_barcodes = generate_barcode_array()
hamming_weight_soma_probabilities = compute_hamming_weight_probabilities(soma_barcodes)
epitope_probabilities_soma = compute_epitope_probabilities(soma_barcodes)


## --- import and threshold the binary masks ---
data_file = "../Data/neuron_barcodes_full_roi.npz"
data = np.load(data_file, allow_pickle=True)

# Get the barcode data
barcode_data = data["arr_0"].item()
discrete, thresholded, expressions_per_object, total_cells, threshold = (
    load_barcode_data(data_file, threshold_method="value", threshold_value=15)
)
hamming_weight_neurites_probabilities = compute_hamming_weight_probabilities(
    thresholded
)
epitope_probabilities_neurites = compute_epitope_probabilities(thresholded)


if __name__ == "__main__":
    # plot the ideal hamming weight distribution

    print("starting simulation")
    n_trials = 5000
    n_samples = [
        # 1,
        # 2,
        # 3,
        # 5,
        # 8,
        10,
        # 15,
        # 20,
        # 30,
        # 40,
        # 50,
        # 60,
        # 70,
        # 80,
        # 90,
        100,
        # 120,
        # 146,
        # 150,
        # 175,
        # 200,
        # 300,
        # 400,
        # 800,
        # 1600,
        1000,
        # 3200,
        # 6400,
        # 12800,
        # 25600,
        # 51200,
        10000,
        # 102400,
        # 204800,
        # 409600,
        100000,
        # 819200,
        # 1638400,
        # 3276800,
        # 1e6,
    ]
    n_samples = [int(sample) for sample in n_samples]

    e11_distribution_binary = generate_perfect_distributions(
        soma_barcodes.shape[1] / 2, np.arange(1, soma_barcodes.shape[1] + 1)
    )
    e11_distribution_binary_2level = generate_perfect_distributions(
        soma_barcodes.shape[1], np.arange(1, soma_barcodes.shape[1] * 2 + 1)
    )
    brainbow_distribution_binary = generate_perfect_distributions(2, np.arange(1, 4))
    brainbow_distribution_2level_binary = generate_perfect_distributions(
        4, np.arange(1, 7)
    )

    tetbow_distribution_binary = generate_perfect_distributions(4, np.arange(1, 8))
    tetbow_distribution_2level_binary = generate_perfect_distributions(
        8, np.arange(1, 15)
    )

    distributions = {
        "E11 - ideal binary": (
            e11_distribution_binary,
            len(e11_distribution_binary),
        ),
        # "Tetbow - 2 level binary": (
        #     tetbow_distribution_2level_binary,
        #     len(tetbow_distribution_2level_binary),
        # ),
        # "E11 - 2 level binary": (
        #     e11_distribution_binary_2level,
        #     len(e11_distribution_binary_2level),
        # ),
        # "Brainbow - 2 level binary": (
        #     brainbow_distribution_2level_binary,
        #     len(brainbow_distribution_2level_binary),
        # ),
        # "Brainbow - ideal binary": (
        #     brainbow_distribution_binary,
        #     len(brainbow_distribution_binary),
        # ),
        "Tetbow - ideal binary": (
            tetbow_distribution_binary,
            len(tetbow_distribution_binary),
        ),
    }

    for name, (distribution, length) in distributions.items():
        print(f"Running simulation for {name} with length {length}")
        run_monte_carlo_model(
            distribution,
            n_samples,
            n_trials,
            length=length,
            epitope_percentages=None,
            kind=f"simulation - {name}",
        )

    # Double the length for levels
    # doubled_length = 18 * 2
    # duplicated_epitope_percentages = duplicate_probabilities(epitope_probabilities_soma)
    # # Use true epitope percentages for levels
    # run_monte_carlo_model(
    #     hamming_weight_soma_probabilities,
    #     n_samples,
    #     n_trials,
    #     length=doubled_length,  # Double the length
    #     epitope_percentages=duplicated_epitope_percentages,  # Use true epitope percentages
    #     kind="simulation - true soma data with 2 levels",
    # )

    run_monte_carlo_model(
        hamming_weight_soma_probabilities,
        n_samples,
        n_trials,
        length=18,  # Double the length
        epitope_percentages=epitope_probabilities_soma,  # Use true epitope percentages
        kind="simulation - true soma data",
    )
