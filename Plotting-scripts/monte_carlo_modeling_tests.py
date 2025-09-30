import numpy as np
import pytest
from monte_carlo_modeling import (
    count_observed_barcode_types,
    generate_random_barcode,
    monte_carlo_generate_cells,
    run_monte_carlo_model,
)


def test_generate_random_barcode_length_and_weight():
    length = 18
    weight = 5
    barcode = generate_random_barcode(weight, length)
    assert len(barcode) == length
    assert barcode.sum() == weight
    assert set(np.unique(barcode)).issubset({0, 1})


def test_generate_random_barcode_zero_weight():
    barcode = generate_random_barcode(0, 10)
    assert np.all(barcode == 0)


def test_generate_random_barcode_full_weight():
    length = 8
    barcode = generate_random_barcode(length, length)
    assert np.all(barcode == 1)


# --- Tests for monte_carlo_generate_cells ---


def test_monte_carlo_generate_cells_shape():
    dist = {2: 1.0}
    cells = monte_carlo_generate_cells(dist, num_samples=50, length=10)
    assert cells.shape == (50, 10)
    # each barcode should have exactly 2 ones
    assert np.all(cells.sum(axis=1) == 2)


def test_monte_carlo_generate_cells_distribution():
    dist = {1: 0.2, 3: 0.8}
    cells = monte_carlo_generate_cells(dist, num_samples=200, length=10)
    weights = cells.sum(axis=1)
    # should roughly follow distribution
    frac_1 = np.mean(weights == 1)
    frac_3 = np.mean(weights == 3)
    assert 0.1 < frac_1 < 0.4
    assert 0.6 < frac_3 < 0.9


# --- Tests for count_unique_barcodes ---


def test_count_unique_barcodes_basic():
    barcodes = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    assert count_observed_barcode_types(barcodes) == 2


def test_count_unique_barcodes_full_set():
    # For length 3, maximum unique barcodes is 2^3 = 8
    barcodes = np.array([[int(x) for x in f"{i:03b}"] for i in range(8)])
    assert count_observed_barcode_types(barcodes) == 8
    # add duplicates, still should be 8
    extended = np.vstack([barcodes, barcodes])
    assert count_observed_barcode_types(extended) == 8


# --- Tests for run_monte_carlo_model ---

# 2. Put your tests inline
# import numpy as np


def test_run_monte_carlo_model_output_structure():
    dist = {2: 1.0}
    num_samples = [5, 10]
    iterations = 3

    records = run_monte_carlo_model(dist, num_samples, iterations)

    # total records = len(num_samples) * iterations
    assert len(records) == len(num_samples) * iterations

    # each record should have the right keys
    for rec in records:
        assert set(rec.keys()) == {
            "num_samples",
            "iteration",
            "unique_count",
            "unique_fraction",
        }
        assert 0 <= rec["unique_count"] <= rec["num_samples"]
        assert 0.0 <= rec["unique_fraction"] <= 1.0


def test_run_monte_carlo_model_unique_fraction_bounds():
    dist = {1: 1.0}
    num_samples = [4]
    iterations = 2
    records = run_monte_carlo_model(dist, num_samples, iterations)

    for rec in records:
        assert rec["unique_fraction"] <= 1.0
        assert rec["unique_count"] <= rec["num_samples"]


@pytest.mark.parametrize("monkeypatch", [pytest.MonkeyPatch()], indirect=True)
def test_run_monte_carlo_model_reproducibility(monkeypatch):
    # Replace barcode generator with deterministic version
    def fake_generate_random_barcode(weight, length=18):
        return np.zeros(length, dtype=np.uint8)

    monkeypatch.setattr(
        "monte_carlo_modeling.generate_random_barcode", fake_generate_random_barcode
    )

    dist = {0: 1.0}
    num_samples = [3]
    iterations = 1
    records = run_monte_carlo_model(dist, num_samples, iterations)

    assert records[0]["unique_count"] == 1
    assert records[0]["unique_fraction"] == 1 / 3
