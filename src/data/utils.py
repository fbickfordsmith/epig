import sys
from typing import Dict, Tuple

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor
from torch.utils.data import DataLoader

from src.typing import Array


def get_next_batch(dataloader: DataLoader) -> Tensor | Tuple:
    try:
        return next(dataloader)
    except:
        dataloader = iter(dataloader)
        return next(dataloader)


def is_float(x: Array) -> bool:
    if isinstance(x, np.ndarray):
        return x.dtype.kind == "f"
    else:
        return torch.is_floating_point(x)


def replace_zero_with_one(x: Array | float) -> Array | float:
    if isinstance(x, (float, np.floating)):
        return 1.0 if x < sys.float_info.epsilon else x

    elif isinstance(x, np.ndarray):
        x[x < np.finfo(x.dtype).eps] = 1.0
        return x

    elif isinstance(x, Tensor):
        x[x < torch.finfo(x.dtype).eps] = 1.0
        return x

    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def split_indices_using_class_labels(
    labels: np.ndarray,
    test_size: Dict[int, float | int] | float | int,
    rng: Generator,
    balance_test_set: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Avoid sklearn.model_selection.train_test_split() because it doesn't give exact stratification.
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
