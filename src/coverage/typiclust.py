from typing import List

import numpy as np
import pandas as pd
import torch
from numpy.random import Generator

from src.coverage.k_means import k_means_cluster
from src.distance import scipy_knn_dist, torch_knn_dist
from src.typing import Array


def acquire_using_typiclust(
    inputs: Array,
    train_inds: List[int],
    n_acquire: int,
    rng: Generator | None = None,
    seed: int | None = None,
    max_n_clusters: int = 1_000,
    max_n_clusters_for_full_batch_k_means: int = 50,
    min_cluster_size: int = 5,
    k_means_batch_size: int = 5_000,
    n_neighbors: int = 20,
    match_official_repo: bool = True,
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.

    References:
        https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py
    """
    knn_dist_fn = scipy_knn_dist if isinstance(inputs, np.ndarray) else torch_knn_dist

    n_clusters = min(len(train_inds) + n_acquire, max_n_clusters)

    if n_clusters <= max_n_clusters_for_full_batch_k_means:
        k_means_batch_size = None

    if seed is None:
        seed = rng.choice(int(1e6))

    cluster_assignments, _ = k_means_cluster(inputs, n_clusters, seed, k_means_batch_size)
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

        if match_official_repo:
            n_neighbors_i = min(n_neighbors, len(cluster_inds) // 2)
            distance_transform = np.square if isinstance(inputs, np.ndarray) else torch.square
        else:
            n_neighbors_i = min(n_neighbors, len(cluster_inds))
            distance_transform = lambda x: x

        distances = knn_dist_fn(inputs[cluster_inds], n_neighbors_i)
        distances = distance_transform(distances).mean(-1)

        # Directly use the argmin of distances rather than computing typicality = 1 / distances
        # and then taking the argmax of typicality.
        selected_ind = cluster_inds[distances.argmin()]
        selected_inds += [selected_ind]

    return selected_inds
