from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data.utils import replace_zero_with_one


def preprocess_inputs_for_zero_mean_and_unit_variance(
    dataset: Dataset, train_inputs: np.ndarray, **kwargs: Any
) -> Dataset:
    """
    Avoid sklearn.preprocessing.StandardScaler because it is limited to two-dimensional inputs.

    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    dataset.data_mean = np.mean(train_inputs, **kwargs)
    dataset.data_std = np.std(train_inputs, **kwargs)
    dataset.data_std = replace_zero_with_one(dataset.data_std)  # Avoid dividing by zero
    dataset.data = (dataset.data - dataset.data_mean) / dataset.data_std

    return dataset


def preprocess_inputs_for_unit_max_abs_value(
    dataset: Dataset, train_inputs: np.ndarray, **kwargs: Any
) -> Dataset:
    """
    Avoid sklearn.preprocessing.MaxAbsScaler because it is limited to two-dimensional inputs.

    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
    """
    dataset.data_max_abs = np.max(np.abs(train_inputs), **kwargs)
    dataset.data /= dataset.data_max_abs

    return dataset


def preprocess_inputs_for_value_range(
    dataset: Dataset,
    train_inputs: np.ndarray,
    min_value: float = -1.0,
    max_value: float = 1.0,
    **kwargs: Any,
) -> Dataset:
    """
    Avoid sklearn.preprocessing.MinMaxScaler because it is limited to two-dimensional inputs.

    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    dataset.data_min = np.min(train_inputs, **kwargs)
    dataset.data_max = np.max(train_inputs, **kwargs)
    dataset.data = (dataset.data - dataset.data_min) / (dataset.data_max - dataset.data_min)
    dataset.data = dataset.data * (max_value - min_value) + min_value

    return dataset


def preprocess_inputs_for_unit_norm(dataset: Dataset, **kwargs: Any) -> Dataset:
    """
    Avoid sklearn.preprocessing.Normalizer because it is limited to two-dimensional inputs.

    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
    """
    dataset.data_norm = np.linalg.norm(dataset.data, **kwargs)
    dataset.data /= dataset.data_norm

    return dataset


def preprocess_2d_inputs(dataset: Dataset, train_inputs: np.ndarray, mode: str) -> Dataset:
    assert train_inputs.ndim == 2

    if mode == "zero_mean_and_unit_variance":
        dataset = preprocess_inputs_for_zero_mean_and_unit_variance(
            dataset, train_inputs, axis=0, keepdims=True
        )

    elif mode == "unit_max_abs_value":
        dataset = preprocess_inputs_for_unit_max_abs_value(
            dataset, train_inputs, axis=0, keepdims=True
        )

    elif isinstance(mode, str) and ("value_range" in mode):
        min_value, max_value = mode.split("_")[-2:]
        min_value, max_value = float(min_value), float(max_value)

        dataset = preprocess_inputs_for_value_range(
            dataset, train_inputs, min_value=min_value, max_value=max_value, axis=0, keepdims=True
        )

    elif mode == "unit_norm":
        dataset = preprocess_inputs_for_unit_norm(dataset, axis=-1, keepdims=True)

    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    return dataset