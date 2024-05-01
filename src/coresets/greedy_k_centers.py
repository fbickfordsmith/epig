from typing import List, Sequence

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor


def acquire_using_greedy_k_centers(
    inputs: Tensor, train_inds: Sequence[int], n_acquire: int, rng: Generator
) -> List[int]:
    """
    The inputs passed to this function are assumed to be the combined labelled and unlabelled
    inputs, where train_inds indicates which of them are labelled.

    References:
        https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
        https://github.com/AminParvaneh/alpha_mix_active_learning/blob/main/query_strategies/core_set.py
    """
    non_train_inds = np.setdiff1d(range(len(inputs)), train_inds)

    if len(train_inds) == 0:
        min_distances = float("inf") * torch.ones(len(non_train_inds))  # [N_u,]
        selected_ind = rng.choice(non_train_inds)
    else:
        distances = torch.cdist(inputs[non_train_inds], inputs[train_inds])  # [N_u, N_l]
        min_distances, _ = torch.min(distances, dim=-1)  # [N_u,]
        selected_ind = torch.argmax(min_distances).item()

    selected_inds = [selected_ind]

    for _ in range(n_acquire - 1):
        distances = torch.cdist(inputs[non_train_inds], inputs[[selected_ind]])  # [N_u, 1]
        min_distances = torch.minimum(min_distances, distances.flatten())  # [N_u,]

        selected_ind = torch.argmax(min_distances).item()
        selected_ind = non_train_inds[selected_ind]

        selected_inds += [selected_ind]

    return selected_inds
