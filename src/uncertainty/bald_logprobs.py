"""
BALD(x) = 0 for a deterministic model but numerical instabilities can lead to nonzero scores.

Cl = number of classes
K = number of model samples
N = number of examples

References:
    https://github.com/BlackHC/batchbald_redux/blob/master/01_batchbald.ipynb
"""

import math

import torch
from torch import Tensor

from src.math import logmeanexp
from src.uncertainty.utils import check


def entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    Arguments:
        logprobs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.exp(logprobs) * logprobs, dim=-1)


def marginal_entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim == 3

    logprobs = logmeanexp(logprobs, dim=1)  # [N, Cl]

    scores = entropy_from_logprobs(logprobs)  # [N,]
    scores = check(scores, max_value=math.log(logprobs.shape[-1]), score_type="ME")  # [N,]

    return scores  # [N,]


def conditional_entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    E_{p(θ)}[H[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim == 3

    scores = entropy_from_logprobs(logprobs)  # [N, K]
    scores = torch.mean(scores, dim=-1)  # [N,]
    scores = check(scores, max_value=math.log(logprobs.shape[-1]), score_type="CE")  # [N,]

    return scores  # [N,]


def bald_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    BALD(x) = E_{p(θ)}[H[p(y|x)] - H[p(y|x,θ)]]
            = H[p(y|x)] - E_{p(θ)}[H[p(y|x,θ)]]
            = H[E_{p(θ)}[p(y|x,θ)]] - E_{p(θ)}[H[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    marg_entropy = marginal_entropy_from_logprobs(logprobs)  # [N,]
    cond_entropy = conditional_entropy_from_logprobs(logprobs)  # [N,]

    scores = marg_entropy - cond_entropy  # [N,]
    scores = check(scores, max_value=math.log(logprobs.shape[-1]), score_type="BALD")  # [N,]

    return scores  # [N,]
