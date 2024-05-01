from typing import List

import numpy as np
import pandas as pd
import torch
from faiss import StandardGpuResources, knn
from faiss.contrib.torch_utils import torch_replacement_knn_gpu
from numpy.random import Generator
from torch import Tensor

from src.coresets.k_means import k_means_cluster


def acquire_using_typiclust(
    inputs: Tensor,
    train_inds: List[int],
    n_acquire: int,
    rng: Generator,
    max_n_clusters: int = 1_000,
    max_n_clusters_for_full_batch_k_means: int = 50,
    min_cluster_size: int = 5,
    k_means_batch_size: int = 5_000,
    n_neighbors: int = 20,
    match_paper: bool = True,
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.

    References:
        https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py
    """
    n_clusters = min(len(train_inds) + n_acquire, max_n_clusters)
    seed = rng.choice(int(1e6))

    if n_clusters <= max_n_clusters_for_full_batch_k_means:
        k_means_batch_size = None

    cluster_assignments, _ = k_means_cluster(inputs.cpu(), n_clusters, seed, k_means_batch_size)
    cluster_ids, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
    cluster_sizes_train = np.bincount(cluster_assignments[train_inds], minlength=len(cluster_ids))

    clusters = {
        "id": cluster_ids,
        "neg_size": -cluster_sizes,
        "size": cluster_sizes,
        "size_train": cluster_sizes_train,
    }
    clusters = pd.DataFrame(clusters)
    clusters = clusters[clusters["size"] > min_cluster_size]
    clusters = clusters.sort_values(by=["size_train", "neg_size"])

    selected_inds = []

    for i in range(n_acquire):
        cluster_assignments[train_inds + selected_inds] = -1

        cluster_label = clusters["id"].iloc[i % len(clusters)]
        cluster_inds = np.flatnonzero(cluster_assignments == cluster_label)

        if match_paper:
            n_neighbors_i = min(n_neighbors, len(cluster_inds))
            use_squared_distance = False
        else:
            n_neighbors_i = min(n_neighbors, len(cluster_inds) // 2)
            use_squared_distance = True

        typicality = compute_typicality(inputs[cluster_inds], n_neighbors_i, use_squared_distance)

        selected_ind = cluster_inds[torch.argmax(typicality)]
        selected_inds += [selected_ind]

    return selected_inds


def compute_pairwise_distances(inputs: Tensor, p: int = 2) -> Tensor:
    """
    Validation:
    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists_scipy = squareform(pdist(inputs.numpy()))
    >>> dists_torch = compute_pairwise_distances(inputs)
    >>> assert np.allclose(dists_scipy, dists_torch.numpy())

    Arguments:
        inputs: Tensor[float], [N, F]
        p: int

    Returns:
        distances: Tensor[float], [N, N]

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
    """
    row_inds, col_inds = torch.triu_indices(len(inputs), len(inputs), offset=1)  # [N',], [N',]

    distances = torch.zeros(len(inputs), len(inputs), device=inputs.device)  # [N, N]
    distances[row_inds, col_inds] = torch.pdist(inputs, p=p)  # [N',]
    distances += torch.clone(distances.T)  # [N, N]

    return distances  # [N, N]


def compute_typicality(
    inputs: Tensor, n_neighbors: int, use_squared_distance: bool = False, mode: str = "standard"
) -> Tensor:
    """
    From the Faiss docs:
        Faiss reports squared Euclidean (L2) distance, avoiding the square root. This is still
        monotonic as the Euclidean distance, but if exact distances are needed, an additional
        square root of the result is needed.

    References:
        https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
        https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py#L496
    """
    if mode == "standard":
        distances = compute_pairwise_distances(inputs)  # [N, N]
        distances, _ = torch.sort(distances, dim=-1)  # [N, N]
        distances = distances[:, 1 : (n_neighbors + 1)]  # [N, k]
        distances = torch.square(distances) if use_squared_distance else distances  # [N, k]

    elif "faiss" in mode:
        device = inputs.device
        inputs = inputs.cpu()  # [N, F]

        if "gpu" in mode:
            distances, _ = torch_replacement_knn_gpu(
                StandardGpuResources(), inputs, inputs, n_neighbors + 1, device=0
            )  # [N, k + 1]
            distances = distances.to(device)  # [N, k + 1]

        else:
            distances, _ = knn(inputs.numpy(), inputs.numpy(), n_neighbors + 1)  # [N, k + 1]
            distances = torch.tensor(distances, device=device)  # [N, k + 1]

        distances = distances[:, 1:]  # [N, k]
        distances = distances if use_squared_distance else torch.sqrt(distances)  # [N, k]

    return 1 / torch.mean(distances, dim=-1)  # [N,]
