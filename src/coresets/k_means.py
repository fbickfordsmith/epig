from typing import List, Sequence, Tuple

import numpy as np
import torch
from faiss import Kmeans as FaissKmeans
from numpy.random import Generator
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
from torch import Tensor

from src.typing import Array


def acquire_using_k_means_plusplus(
    inputs: Tensor, train_inds: Sequence[int], n_acquire: int, rng: Generator
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.
    """
    non_train_inds = np.setdiff1d(range(len(inputs)), train_inds)

    inputs = inputs[non_train_inds]

    _, selected_inds = kmeans_plusplus(
        inputs.cpu().numpy(), n_clusters=n_acquire, random_state=rng.choice(int(1e6))
    )

    return selected_inds


def acquire_using_k_means(
    inputs: Tensor, train_inds: Sequence[int], n_acquire: int, rng: Generator
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.

    References:
        https://github.com/JordanAsh/badge/blob/master/query_strategies/kmeans_sampling.py
    """
    non_train_inds = np.setdiff1d(range(len(inputs)), train_inds)

    inputs = inputs[non_train_inds]

    cluster_assignments, cluster_centers = k_means_cluster(
        inputs.cpu(), n_clusters=n_acquire, seed=rng.choice(int(1e6))
    )
    cluster_centers = torch.tensor(cluster_centers, dtype=inputs.dtype, device=inputs.device)

    selected_inds = []

    for i, cluster_center in enumerate(cluster_centers):
        cluster_inds = np.flatnonzero(cluster_assignments == i)

        distances = torch.cdist(inputs[cluster_inds], cluster_center[None, :])
        ind_min_distance = torch.argmin(distances).item()

        selected_ind = cluster_inds[ind_min_distance]
        selected_inds += [selected_ind]

    selected_inds = non_train_inds[selected_inds]

    return selected_inds.tolist()


def k_means_cluster(
    inputs: Array,
    n_clusters: int,
    seed: int,
    batch_size: int = None,
    verbose: bool = False,
    use_faiss: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    References:
        https://github.com/facebookresearch/faiss/blob/main/faiss/python/extra_wrappers.py#L434
    """
    if use_faiss:
        k_means = FaissKmeans(
            d=inputs.shape[-1],
            k=n_clusters,
            gpu=1,
            niter=100,
            nredo=10,
            verbose=verbose,
            seed=seed,
        )
        k_means.train(inputs)

        _, cluster_assignments = k_means.assign(inputs)  # [N,]
        cluster_centers = k_means.centroids  # [k, F]

    else:
        # When init="k-means++" (default), n_init="auto" -> n_init=1.
        k_means_kwargs = dict(n_init="auto", verbose=verbose, random_state=seed)

        if batch_size is None:
            k_means = SKLearnKMeans(n_clusters, **k_means_kwargs)
        else:
            k_means = MiniBatchKMeans(n_clusters, batch_size=batch_size, **k_means_kwargs)

        cluster_assignments = k_means.fit_predict(inputs)  # [N,]
        cluster_centers = k_means.cluster_centers_  # [k, F]

    return cluster_assignments, cluster_centers  # [N,], [k, F]
