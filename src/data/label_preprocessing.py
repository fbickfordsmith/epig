from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data.utils import is_float, replace_zero_with_one


def preprocess_labels_for_zero_mean_and_unit_variance(
    dataset: Dataset, train_labels: np.ndarray, **kwargs: Any
) -> Dataset:
    """
    Here we could use sklearn.preprocessing.StandardScaler but instead go for consistency with
    preprocess_inputs_for_zero_mean_and_unit_variance().
    """
    assert is_float(train_labels)

    dataset.targets_mean = np.mean(train_labels, **kwargs)
    dataset.targets_std = np.std(train_labels, **kwargs)
    dataset.targets_std = replace_zero_with_one(dataset.targets_std)  # Avoid dividing by zero
    dataset.targets = (dataset.targets - dataset.targets_mean) / dataset.targets_std

    return dataset


def preprocess_labels_for_unit_max_abs_value(
    dataset: Dataset, train_labels: np.ndarray, **kwargs: Any
) -> Dataset:
    """
    Here we could use sklearn.preprocessing.MaxAbsScaler but instead go for consistency with
    preprocess_inputs_for_unit_max_abs_value().
    """
    assert is_float(train_labels)

    dataset.targets_max_abs = np.max(np.abs(train_labels), **kwargs)
    dataset.targets /= dataset.targets_max_abs

    return dataset


def preprocess_labels_for_value_range(
    dataset: Dataset,
    train_labels: np.ndarray,
    min_value: float = -1.0,
    max_value: float = 1.0,
    **kwargs: Any,
) -> Dataset:
    """
    Here we could use sklearn.preprocessing.MinMaxScaler but instead go for consistency with
    preprocess_inputs_for_value_range().
    """
    assert is_float(train_labels)

    dataset.targets_min = np.min(train_labels, **kwargs)
    dataset.targets_max = np.max(train_labels, **kwargs)
    dataset.targets = (dataset.targets - dataset.targets_min) / (
        dataset.targets_max - dataset.targets_min
    )
    dataset.targets = dataset.targets * (max_value - min_value) + min_value

    return dataset


def preprocess_1d_labels(dataset: Dataset, train_labels: np.ndarray, mode: str) -> None:
    assert train_labels.ndim == 1

    if mode == "zero_mean_and_unit_variance":
        dataset = preprocess_labels_for_zero_mean_and_unit_variance(dataset, train_labels)

    elif mode == "unit_max_abs_value":
        dataset = preprocess_labels_for_unit_max_abs_value(dataset, train_labels)

    elif isinstance(mode, str) and ("value_range" in mode):
        min_value, max_value = mode.split("_")[-2:]
        min_value, max_value = float(min_value), float(max_value)
        dataset = preprocess_labels_for_value_range(dataset, train_labels, min_value, max_value)

    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    return dataset
