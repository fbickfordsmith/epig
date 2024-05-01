"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import torch
from torch import Tensor

from src.math import logmeanexp
from src.uncertainty.utils import check


def mean_standard_deviation_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    mean_std[p(y|x)] = 1/Cl ∑_{y} sqrt( E_{p(θ)}[ p(y|x,θ)^2 ] - E_{p(θ)}[ p(y|x,θ) ]^2 )
                     = 1/Cl ∑_{y} sqrt( E_{p(θ)}[ p(y|x,θ)^2 ] - p(y|x)^2 )

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim == 3

    log_mean_squared_cond_probs = logmeanexp(2 * logprobs, dim=1)  # [N, Cl]
    log_squared_marg_probs = 2 * logmeanexp(logprobs, dim=1)  # [N, Cl]

    variance = torch.exp(log_mean_squared_cond_probs) - torch.exp(log_squared_marg_probs)  # [N, Cl]

    scores = torch.mean(torch.sqrt(variance), dim=-1)  # [N,]
    scores = check(scores, score_type="mean_std")  # [N,]

    return scores  # [N,]


def predictive_margin_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    margin[p(y|x)] = p(y=y_2|x) - p(y=y_1|x)

    where y_1 and y_2 are the classes with most and second most mass under the predictive
    distribution.

    Arguments:
        logprobs: Tensor[float], [N, Cl] or [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim in {2, 3}

    if logprobs.ndim == 3:
        logprobs = logmeanexp(logprobs, dim=1)  # [N, Cl]

    logprobs, _ = torch.sort(logprobs, dim=-1, descending=True)  # [N, Cl]

    scores = torch.exp(logprobs[:, 1]) - torch.exp(logprobs[:, 0])  # [N,]
    scores = check(scores, min_value=-1, max_value=0, score_type="margin")  # [N,]

    return scores  # [N,]


def variation_ratio_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    variation_ratio[p(y|x)] = 1 - max_{y} p(y|x)

    Arguments:
        logprobs: Tensor[float], [N, Cl] or [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim in {2, 3}

    if logprobs.ndim == 3:
        logprobs = logmeanexp(logprobs, dim=1)  # [N, Cl]

    max_logprobs, _ = torch.max(logprobs, dim=-1)  # [N,]

    scores = 1 - torch.exp(max_logprobs)  # [N,]
    scores = check(scores, max_value=(1 - 1 / logprobs.shape[-1]), score_type="var_ratio")  # [N,]

    return scores  # [N,]
