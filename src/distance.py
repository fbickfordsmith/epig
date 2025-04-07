"""
M = number of examples 0
N = number of examples 1
F = number of features
k = number of neighbors
"""

from typing import Any

import numpy as np
import torch
from faiss import StandardGpuResources, knn, knn_gpu, pairwise_distance_gpu, pairwise_distances
from faiss.contrib.torch_utils import torch_replacement_knn_gpu as knn_gpu_torch
from faiss.contrib.torch_utils import (
    torch_replacement_pairwise_distance_gpu as pairwise_distance_gpu_torch,
)
from scipy.spatial.distance import pdist, squareform
from torch import Tensor

from src.typing import Array


def scipy_knn_dist(inputs: np.ndarray, n_neighbors: int, **kwargs: Any) -> np.ndarray:
    """
    Arguments:
        inputs: np.ndarray[float], [N, F]
        n_neighbors: int

    Returns:
        np.ndarray[float], [N, k]
    """
    distances = squareform(pdist(inputs, **kwargs))  # [N, N]
    distances = np.sort(distances, axis=-1)  # [N, N]
    distances = distances[:, 1 : (n_neighbors + 1)]  # [N, k]

    return distances  # [N, k]


def torch_cdist(inputs_0: Tensor, inputs_1: Tensor, **kwargs: Any) -> Tensor:
    """
    Numerically stable implementation of torch.cdist().

    References:
        https://pytorch.org/docs/stable/generated/torch.cdist.html
        https://github.com/pytorch/pytorch/issues/42479
        https://github.com/pytorch/pytorch/issues/57690

    Arguments:
        inputs_0: Tensor[float], [N, F]
        inputs_1: Tensor[float], [M, F]

    Returns:
        Tensor[float], [N, M]
    """
    return torch.cdist(inputs_0, inputs_1, compute_mode="donot_use_mm_for_euclid_dist", **kwargs)


def torch_pdist_squareform(inputs: Tensor, **kwargs: Any) -> Tensor:
    """
    Arguments:
        inputs: Tensor[float], [N, F]

    Returns:
        Tensor[float], [N, N]

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
    """
    distances = torch.zeros(len(inputs), len(inputs), device=inputs.device)  # [N, N]

    row_inds, col_inds = torch.triu_indices(*distances.shape, offset=1)  # [N',], [N',]
    distances[row_inds, col_inds] = torch.pdist(inputs, **kwargs)  # [N',]

    return distances + distances.T  # [N, N]


def torch_knn_dist(inputs: Tensor, n_neighbors: int, **kwargs: Any) -> Tensor:
    """
    Arguments:
        inputs: Tensor[float], [N, F]
        n_neighbors: int

    Returns:
        Tensor[float], [N, k]
    """
    distances = torch_pdist_squareform(inputs, **kwargs)  # [N, N]
    distances, _ = torch.sort(distances, dim=-1)  # [N, N]
    distances = distances[:, 1 : (n_neighbors + 1)]  # [N, k]

    return distances  # [N, k]


def faiss_cdist(inputs_0: Array, inputs_1: Array, use_gpu: bool = True) -> Array:
    """
    From the Faiss docs:
        Faiss reports squared Euclidean (L2) distance, avoiding the square root. This is still
        monotonic as the Euclidean distance, but if exact distances are needed, an additional
        square root of the result is needed.

    References:
        https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
        https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py#L496

    Arguments:
        inputs_0: Array[float], [N, F]
        inputs_1: Array[float], [M, F]
        use_gpu: bool

    Returns:
        Array[float], [N, M]
    """
    assert inputs_0.dtype == inputs_1.dtype

    if use_gpu:
        gpu = StandardGpuResources()

        if isinstance(inputs_0, np.ndarray):
            distances = pairwise_distance_gpu(gpu, inputs_0, inputs_1)  # [N, M]
            distances = np.sqrt(distances)  # [N, M]
        else:
            distances = pairwise_distance_gpu_torch(gpu, inputs_0, inputs_1)  # [N, M]
            distances = torch.sqrt(distances)  # [N, M]

    else:
        distances = pairwise_distances(inputs_0, inputs_1)  # [N, M]
        distances = np.sqrt(distances)  # [N, M]

    return distances  # [N, M]


def faiss_knn_dist(inputs: Array, n_neighbors: int, use_gpu: bool = True) -> Array:
    """
    From the Faiss docs:
        Faiss reports squared Euclidean (L2) distance, avoiding the square root. This is still
        monotonic as the Euclidean distance, but if exact distances are needed, an additional
        square root of the result is needed.

    References:
        https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
        https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py#L496

    Arguments:
        inputs: Array[float], [N, F]
        n_neighbors: int
        use_gpu: bool

    Returns:
        Array[float], [N, k]
    """
    n_neighbors += 1

    if use_gpu:
        gpu = StandardGpuResources()

        if isinstance(inputs, np.ndarray):
            distances, _ = knn_gpu(gpu, inputs, inputs, n_neighbors)  # [N, k + 1]
            distances = np.sqrt(distances)  # [N, k + 1]
        else:
            distances, _ = knn_gpu_torch(gpu, inputs, inputs, n_neighbors, device=0)  # [N, k + 1]
            distances = torch.sqrt(distances)  # [N, k + 1]

    else:
        distances, _ = knn(inputs, inputs, n_neighbors)  # [N, k + 1]
        distances = np.sqrt(distances)  # [N, k + 1]

    return distances[:, 1:]  # [N, k]
