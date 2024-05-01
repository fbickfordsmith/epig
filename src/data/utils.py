from typing import Tuple, Union

import numpy as np
from numpy.random import Generator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def split_indices_using_class_labels(
    labels: np.ndarray,
    test_size: Union[dict, float, int],
    rng: Generator,
    balance_test_set: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sklearn.model_selection.train_test_split() doesn't produce exact stratification.
    """
    if isinstance(test_size, float) and balance_test_set:
        class_counts = np.bincount(labels)
        test_size = int(test_size * min(class_counts))
        test_size = max(test_size, 1)

    train_inds, test_inds = [], []

    for _class in np.unique(labels):
        class_inds = np.flatnonzero(labels == _class)
        class_inds = rng.permutation(class_inds)

        if isinstance(test_size, dict):
            class_test_size = test_size[_class]
        else:
            class_test_size = test_size

        if isinstance(class_test_size, float):
            class_test_size = int(class_test_size * len(class_inds))
            class_test_size = max(class_test_size, 1)

        train_inds += [class_inds[:-class_test_size]]
        test_inds += [class_inds[-class_test_size:]]

    train_inds = np.concatenate(train_inds)
    train_inds = rng.permutation(train_inds)

    test_inds = np.concatenate(test_inds)
    test_inds = rng.permutation(test_inds)

    return train_inds, test_inds


def compute_mean_and_std(
    x: np.ndarray, axis: int = 0, keepdims: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    To avoid dividing by zero we set the standard deviation to one if it is less than epsilon.
    """
    mean = np.mean(x, axis=axis, keepdims=keepdims)
    std = np.std(x, axis=axis, keepdims=keepdims)

    eps = np.finfo(x.dtype).eps

    if isinstance(std, np.ndarray):
        std[std < eps] = 1
    elif std < eps:
        std = 1

    return mean, std


def preprocess_inputs_for_unit_norm(
    dataset: Dataset, axis: int = -1, keepdims: bool = True
) -> Dataset:
    dataset.data_norm = np.linalg.norm(dataset.data, axis=axis, keepdims=keepdims)
    dataset.data /= dataset.data_norm

    return dataset


def preprocess_inputs_for_unit_variance(
    dataset: Dataset, train_inputs: np.ndarray, axis: int = 0, keepdims: bool = True
) -> Dataset:
    mean, std = compute_mean_and_std(train_inputs, axis=axis, keepdims=keepdims)

    dataset.data = (dataset.data - mean) / std
    dataset.data_mean = mean
    dataset.data_std = std

    return dataset


def preprocess_labels_for_unit_variance(
    dataset: Dataset, train_labels: np.ndarray, axis: int = 0, keepdims: bool = True
) -> Dataset:
    mean, std = compute_mean_and_std(train_labels, axis=axis, keepdims=keepdims)

    dataset.targets = (dataset.targets - mean) / std
    dataset.targets_mean = mean
    dataset.targets_std = std

    return dataset


def get_next(dataloader: DataLoader) -> Union[Tensor, Tuple]:
    try:
        return next(dataloader)
    except:
        dataloader = iter(dataloader)
        return next(dataloader)
