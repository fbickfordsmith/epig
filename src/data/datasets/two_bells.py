from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import logistic, multivariate_t

from src.data.datasets.base import BaseDataset


class TwoBells(BaseDataset):
    """
    Synthetic dataset with two-dimensional inputs and binary class labels.

    For p_2(x) in Figure 1 in [1], use TwoBells(..., input_dim_0=(np.pi / 4), input_dist_shape=0.2).

    References:
    [1] https://arxiv.org/abs/2304.08151
    """

    def __init__(
        self,
        data_dir: Path | str,
        seed: int,
        n_train: int,
        n_test: int,
        input_dim_0: float = 0.0,
        input_dist_shape: float = 0.8,
        input_dist_ndof: int = 5,
        input_scale: float = 2.0,
        label_scale: float = 0.05,
        train: bool = True,
    ) -> None:
        def boundary(inputs_dim_0: float | np.ndarray, input_scale: float) -> float | np.ndarray:
            return np.tanh(input_scale * inputs_dim_0)

        def label_dist(inputs: np.ndarray, boundary: Callable, label_scale: float) -> np.ndarray:
            # Compute the probability of class 1.
            boundary_distance_in_dim_1 = inputs[:, 1] - boundary(inputs[:, 0])
            return logistic.cdf(boundary_distance_in_dim_1 / label_scale)

        rng = np.random.default_rng(seed=seed)

        boundary = partial(boundary, input_scale=input_scale)

        input_dist = multivariate_t(
            loc=[input_dim_0, boundary(input_dim_0)], shape=input_dist_shape, df=input_dist_ndof
        )
        label_dist = partial(label_dist, boundary=boundary, label_scale=label_scale)

        n_examples = n_train if train else n_test

        inputs = input_dist.rvs(size=n_examples, random_state=rng).astype(np.float32)
        labels = rng.binomial(n=np.ones(n_examples, dtype=int), p=label_dist(inputs))

        self.input_dist = input_dist
        self.data = inputs
        self.targets = labels
