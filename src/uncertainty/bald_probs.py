"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import math

import torch
from torch import Tensor

from src.uncertainty.utils import check


def entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    If p(y_i|x) is 0, we make sure p(y_i|x) log p(y_i|x) evaluates to 0, not NaN.

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.xlogy(probs, probs), dim=-1)


def marginal_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    probs = torch.mean(probs, dim=1)  # [N, Cl]

    scores = entropy_from_probs(probs)  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="ME")  # [N,]

    return scores  # [N,]


def conditional_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    E_{p(θ)}[H[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    scores = entropy_from_probs(probs)  # [N, K]
    scores = torch.mean(scores, dim=-1)  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="CE")  # [N,]

    return scores  # [N,]


def bald_from_probs(probs: Tensor) -> Tensor:
    """
    BALD(x) = E_{p(θ)}[H[p(y|x)] - H[p(y|x,θ)]]
            = H[p(y|x)] - E_{p(θ)}[H[p(y|x,θ)]]
            = H[E_{p(θ)}[p(y|x,θ)]] - E_{p(θ)}[H[p(y|x,θ)]]

    BALD(x) = 0 for a deterministic model but numerical instabilities can lead to nonzero scores.

    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/01_batchbald.ipynb

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    marg_entropy = marginal_entropy_from_probs(probs)  # [N,]
    cond_entropy = conditional_entropy_from_probs(probs)  # [N,]

    scores = marg_entropy - cond_entropy  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="BALD")  # [N,]

    return scores  # [N,]
