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
from src.math import logmeanexp
from src.uncertainty.utils import check
from torch import Tensor


def entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but it uses lots of memory.

    Arguments:
        logprobs: Tensor[float], [*N, Cl]

    Returns:
        Tensor[float], [*N,]
    """
    return -torch.sum(torch.exp(logprobs) * logprobs, dim=-1)  # [*N,]


def entropy_from_probs(probs: Tensor) -> Tensor:
    """
    See entropy_from_logprobs.

    If p(y=y'|x) is 0, we make sure p(y=y'|x) log p(y=y'|x) evaluates to 0, not NaN.

    Arguments:
        probs: Tensor[float], [*N, Cl]

    Returns:
        Tensor[float], [*N,]
    """
    logprobs = torch.clone(probs)  #  [*N, Cl]
    logprobs[probs > 0] = torch.log(probs[probs > 0])  #  [*N, Cl]
    return -torch.sum(probs * logprobs, dim=-1)  # [*N,]


def marginal_entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    logprobs = logmeanexp(logprobs, dim=1)  # [N, Cl]
    scores = entropy_from_logprobs(logprobs)  # [N,]
    scores = check(scores, math.log(logprobs.shape[-1]), score_type="ME")  # [N,]
    return scores  # [N,]


def marginal_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    See marginal_entropy_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    probs = torch.mean(probs, dim=1)  # [N, Cl]
    scores = entropy_from_probs(probs)  # [N,]
    scores = check(scores, math.log(probs.shape[-1]), score_type="ME")  # [N,]
    return scores  # [N,]


def conditional_entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    E_{p(θ)}[H[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    scores = entropy_from_logprobs(logprobs)  # [N, K]
    scores = torch.mean(scores, dim=-1)  # [N,]
    scores = check(scores, math.log(logprobs.shape[-1]), score_type="CE")  # [N,]
    return scores  # [N,]


def conditional_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    See conditional_entropy_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    scores = entropy_from_probs(probs)  # [N, K]
    scores = torch.mean(scores, dim=-1)  # [N,]
    scores = check(scores, math.log(probs.shape[-1]), score_type="CE")  # [N,]
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
    scores = check(scores, math.log(logprobs.shape[-1]), score_type="BALD")  # [N,]
    return scores  # [N,]


def bald_from_probs(probs: Tensor) -> Tensor:
    """
    See bald_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    marg_entropy = marginal_entropy_from_probs(probs)  # [N,]
    cond_entropy = conditional_entropy_from_probs(probs)  # [N,]
    scores = marg_entropy - cond_entropy  # [N,]
    scores = check(scores, math.log(probs.shape[-1]), score_type="BALD")  # [N,]
    return scores  # [N,]
