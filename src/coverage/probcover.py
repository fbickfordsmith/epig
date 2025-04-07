from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.distance import faiss_cdist, torch_cdist


def acquire_using_probcover(
    inputs: Tensor,
    train_inds: List[int],
    n_acquire: int,
    precomputed_graph: pd.DataFrame | None = None,
    ball_radius: float | None = None,
    graph_batch_size: int | None = None,
    match_official_repo: bool = True,
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.

    References:
        https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/prob_cover.py
    """
    if precomputed_graph is None:
        graph = construct_graph(inputs, ball_radius, graph_batch_size, match_official_repo)
    else:
        graph = precomputed_graph.copy()

    selected_inds = []

    for _ in range(n_acquire):
        # Only keep "uncovered" nodes that don't have parents in train_inds or selected_inds.
        parent_ind_is_in_train = np.isin(graph["parent_ind"], [train_inds + selected_inds])
        covered_child_inds = np.unique(graph["child_ind"][parent_ind_is_in_train])
        child_ind_is_covered = np.isin(graph["child_ind"], covered_child_inds)
        graph = graph[~child_ind_is_covered]

        # Select the node with greatest coverage of other nodes (= degree = number of outward edges).
        degrees = np.bincount(graph["parent_ind"], minlength=len(inputs))
        selected_inds += [np.argmax(degrees)]

    return selected_inds


def construct_graph(
    inputs: Tensor,
    ball_radius: float,
    batch_size: int,
    match_official_repo: bool = True,
) -> pd.DataFrame:
    """
    Construct a directed graph where x -> x' if distance(x, x') < ball_radius.
    """
    cdist_fn = torch.cdist if match_official_repo else torch_cdist

    parent_inds, child_inds, distances = [], [], []

    for i in range(0, len(inputs), batch_size):
        distances_i = cdist_fn(inputs[i : i + batch_size], inputs)  # [B, N]
        parent_inds_i, child_inds_i = torch.nonzero(distances_i < ball_radius).T

        parent_inds_i += i
        distances_i = distances_i[distances_i < ball_radius]

        parent_inds += [parent_inds_i.cpu()]
        child_inds += [child_inds_i.cpu()]
        distances += [distances_i.cpu()]

    parent_inds = torch.cat(parent_inds).numpy()
    child_inds = torch.cat(child_inds).numpy()
    distances = torch.cat(distances).numpy()

    return pd.DataFrame({"parent_ind": parent_inds, "child_ind": child_inds, "distance": distances})


def estimate_purity(
    inputs: Tensor,
    cluster_assignments: Tensor,
    ball_radius: float,
    batch_size: int,
    use_faiss: bool = False,
) -> float:
    cdist_fn = faiss_cdist if use_faiss else torch_cdist

    n_pure = 0

    for i in range(0, len(inputs), batch_size):
        distances_i = cdist_fn(inputs[i : i + batch_size], inputs)  # [B, N]
        cluster_assignments_i = cluster_assignments[i : i + batch_size, None]  # [B, 1]

        lies_in_ball = distances_i < ball_radius  # [B, N]
        does_not_match = cluster_assignments_i != cluster_assignments[None, :]  # [B, N]

        is_pure = ~torch.any(lies_in_ball & does_not_match, axis=-1)  # [B,]
        n_pure += torch.sum(is_pure).item()

    return n_pure / len(inputs)


def estimate_ball_radius_for_threshold_purity(
    purities: np.ndarray | Sequence[float], threshold_purity: float, ball_radii: Sequence[float]
) -> float:
    if not isinstance(purities, np.ndarray):
        purities = np.array(purities)

    ind_above = np.flatnonzero(purities > threshold_purity)[-1]
    ind_above = int(ind_above)
    ind_below = ind_above + 1

    purity_above = purities[ind_above]
    purity_below = purities[ind_below]

    ball_radius_above = ball_radii[ind_above]
    ball_radius_below = ball_radii[ind_below]

    purity_gap_above = (purity_above - threshold_purity) / (purity_above - purity_below)

    ball_radius_for_threshold_purity = (
        ball_radius_above + (ball_radius_below - ball_radius_above) * purity_gap_above
    )

    return ball_radius_for_threshold_purity
