from copy import deepcopy
from typing import List

import numpy as np
import torch
from faiss import StandardGpuResources
from faiss.contrib.torch_utils import torch_replacement_pairwise_distance_gpu
from numpy.random import Generator
from torch import Tensor


def acquire_using_probcover(
    inputs: Tensor,
    train_inds: List[int],
    n_acquire: int,
    rng: Generator = None,
    precomputed_graph: dict = None,
    ball_radius: float = None,
    graph_batch_size: int = None,
    match_paper: bool = True,
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.

    References:
        https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/prob_cover.py
    """
    if precomputed_graph is None:
        graph = construct_graph(inputs, ball_radius, graph_batch_size)
    else:
        graph = deepcopy(precomputed_graph)

    selected_inds = []

    for _ in range(n_acquire):
        # Only keep "uncovered" nodes that don't have parents in train_inds or selected_inds.
        has_parent_in_train = np.isin(graph["parent_ind"], [train_inds + selected_inds])
        covered_child_inds = np.unique(graph["child_ind"][has_parent_in_train])
        graph = graph[~np.isin(graph["parent_ind"], covered_child_inds)]

        # Select a node based on its coverage of other nodes (= degree = number of outward edges).
        degrees = np.bincount(graph["parent_ind"], minlength=len(inputs))

        if match_paper:
            # See Appendix C1 in the ProbCover paper.
            selected_ind = rng.choice(np.argsort(degrees)[-5:])
        else:
            selected_ind = np.argmax(degrees)

        selected_inds += [selected_ind]

    return selected_inds


def construct_graph(inputs: Tensor, ball_radius: float, batch_size: int) -> dict:
    """
    Construct a directed graph where x -> x' if distance(x, x') < ball_radius.
    """
    parent_inds, child_inds, distances = [], [], []

    for i in range(0, len(inputs), batch_size):
        distances_i = torch.cdist(inputs[i : i + batch_size], inputs)  # [B, N]
        parent_inds_i, child_inds_i = torch.nonzero(distances_i < ball_radius).T

        parent_inds_i += i
        distances_i = distances_i[distances_i < ball_radius]

        parent_inds += [parent_inds_i.cpu()]
        child_inds += [child_inds_i.cpu()]
        distances += [distances_i.cpu()]

    parent_inds = torch.cat(parent_inds).numpy()
    child_inds = torch.cat(child_inds).numpy()
    distances = torch.cat(distances).numpy()

    return {"parent_ind": parent_inds, "child_ind": child_inds, "distance": distances}


def estimate_purity(
    inputs: Tensor,
    cluster_assignments: Tensor,
    ball_radius: float,
    batch_size: int,
    use_faiss: bool = False,
) -> float:
    """
    From the Faiss docs:
        Faiss reports squared Euclidean (L2) distance, avoiding the square root. This is still
        monotonic as the Euclidean distance, but if exact distances are needed, an additional
        square root of the result is needed.

    References:
        https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    """
    n_pure = 0

    for i in range(0, len(inputs), batch_size):
        if use_faiss:
            distances_i = torch_replacement_pairwise_distance_gpu(
                StandardGpuResources(), inputs[i : i + batch_size], inputs
            )  # [B, N]
            distances_i = torch.sqrt(distances_i)  # [B, N]

        else:
            distances_i = torch.cdist(inputs[i : i + batch_size], inputs)  # [B, N]

        cluster_assignments_i = cluster_assignments[i : i + batch_size, None]  # [B, 1]

        lies_in_ball = distances_i < ball_radius  # [B, N]
        does_not_match = cluster_assignments_i != cluster_assignments[None, :]  # [B, N]
        is_pure = ~torch.any(lies_in_ball & does_not_match, axis=-1)  # [B,]

        n_pure += torch.sum(is_pure).item()

    return n_pure / len(inputs)
