import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split
import math

from config import *

from matplotlib.figure import Figure

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

import sys


def set_seed(seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all CUDA devices, if available


set_seed()


def sanity_check(machine_csv_compare):
    machine_csv_compare["rr_interval_seconds"] = (
            machine_csv_compare["rr_interval"] / 1000
    )
    machine_csv_compare["ventricular_rate"] = (
            60 / machine_csv_compare["rr_interval_seconds"]
    )
    machine_csv_compare["pr_interval"] = (
            machine_csv_compare["qrs_onset"] - machine_csv_compare["p_onset"]
    )
    # Calculate P Wave Duration
    machine_csv_compare["p_wave_duration"] = (
            machine_csv_compare["p_end"] - machine_csv_compare["p_onset"]
    )
    # Calculate QRS Duration
    machine_csv_compare["qrs_duration"] = (
            machine_csv_compare["qrs_end"] - machine_csv_compare["qrs_onset"]
    )
    # Calculate QT Interval
    machine_csv_compare["qt_interval"] = (
            machine_csv_compare["t_end"] - machine_csv_compare["qrs_onset"]
    )

    # Checks for plausibility of values
    # Replace negative values and values above 500 ms with NaN
    machine_csv_compare["pr_interval"] = machine_csv_compare["pr_interval"].apply(
        lambda x: np.nan if x < 20 or x > 500 else x
    )
    # Replace zero, negative values, and values above 300 ms with NaN
    machine_csv_compare["p_wave_duration"] = machine_csv_compare[
        "p_wave_duration"
    ].apply(lambda x: np.nan if x <= 20 or x > 300 else x)
    # Replace negative values and values above 300 ms with NaN
    machine_csv_compare["qrs_duration"] = machine_csv_compare["qrs_duration"].apply(
        lambda x: np.nan if x < 50 or x > 300 else x
    )
    # Replace negative values and values above 1000 ms with NaN
    machine_csv_compare["qt_interval"] = machine_csv_compare["qt_interval"].apply(
        lambda x: np.nan if x < 200 or x > 700 else x
    )

    # Calculate qtc_interval using Bazett's formula
    machine_csv_compare["qtc_interval"] = machine_csv_compare["qt_interval"] / np.sqrt(
        60 / machine_csv_compare["ventricular_rate"]
    )

    machine_csv_compare.drop(columns=["rr_interval_seconds"], inplace=True)

    return machine_csv_compare.dropna()


def first_n_digits(num, n):
    return num // 10 ** (int(math.log(num, 10)) - n + 1)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Model checkpoint epoch {epoch} saved to {filename}")


def load_checkpoint(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, scheduler, start_epoch, loss


def plot_real_vs_predicted(true_values, predicted_values, epoch):
    fig = Figure(figsize=(6, 6))  # Create a Figure object directly
    ax = fig.add_subplot(1, 1, 1)  # Add a subplot

    ax.scatter(true_values, predicted_values, alpha=0.5)

    # Plot the 45-degree line for reference
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"True vs Predicted Values - Epoch {epoch}")
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    return fig  # Return the Figure object


def check_accuracy(model, dataloader: torch.utils.data.DataLoader, device):
    # Collect predictions and ground-truths
    predictions = []
    ground_truths = []
    model.eval()
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Forward pass
            predictions.append(outputs.cpu())
            ground_truths.append(targets.cpu())

    # Concatenate all predictions and ground-truths
    predictions = torch.cat(predictions).numpy()
    ground_truths = torch.cat(ground_truths).numpy()

    # Calculate MAE
    mae = mean_absolute_error(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    rmse = root_mean_squared_error(ground_truths, predictions)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (MSE): {rmse:.4f}")
    return mae, mse, rmse


# check_accuracy(model, hand_loader)

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()  # Ensure immediate writing

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_train_test_loader(dataset: torch.utils.data.Dataset, test_split=TEST_SPLIT):
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    train_subset, test_subset = random_split(dataset, [train_size, test_size])

    print("Verifying dataset split...")

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, test_loader


def get_untouched_indices(dataset: torch.utils.data.Dataset, touched_indices: list, amount=500):
    """
    Returns a list of indices from the dataset that have not yet been used.

    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset to sample from.
    - touched_indices (list): List of indices that have already been used.
    - amount (int): Number of untouched indices to return.

    Returns:
    - list: A list of untouched indices.
    """

    total_indices = set(range(len(dataset)))  # All possible indices
    untouched_set = total_indices - set(touched_indices)  # Remove used indices

    untouched_list = list(untouched_set)

    # Ensure we don't sample more than available
    if amount > len(untouched_list):
        print(f"Take less because amount = {amount}, len list = {len(untouched_list)}")
    amount = min(amount, len(untouched_list))

    return random.sample(untouched_list, amount) if amount > 0 else []
