"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import torch
from torch import Tensor

from src.uncertainty.utils import check


def mean_standard_deviation_from_probs(probs: Tensor) -> Tensor:
    """
    mean_std[p(y|x)] = 1/Cl ∑_{y} sqrt( E_{p(θ)}[ p(y|x,θ)^2 ] - E_{p(θ)}[ p(y|x,θ) ]^2 )
                     = 1/Cl ∑_{y} sqrt( E_{p(θ)}[ p(y|x,θ)^2 ] - p(y|x)^2 )

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    variance = torch.mean(torch.square(probs), dim=1) - torch.mean(probs, dim=1) ** 2  # [N, Cl]

    scores = torch.mean(torch.sqrt(variance), dim=-1)  # [N,]
    scores = check(scores, score_type="mean_std")  # [N,]

    return scores  # [N,]


def predictive_margin_from_probs(probs: Tensor) -> Tensor:
    """
    margin[p(y|x)] = p(y=y_2|x) - p(y=y_1|x)

    where y_1 and y_2 are the classes with most and second most mass under the predictive
    distribution.

    Arguments:
        probs: Tensor[float], [N, Cl] or [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim in {2, 3}

    if probs.ndim == 3:
        probs = torch.mean(probs, dim=1)  # [N, Cl]

    probs, _ = torch.sort(probs, dim=-1, descending=True)  # [N, Cl]

    scores = probs[:, 1] - probs[:, 0]  # [N,]
    scores = check(scores, min_value=-1, max_value=0, score_type="margin")  # [N,]

    return scores  # [N,]


def variation_ratio_from_probs(probs: Tensor) -> Tensor:
    """
    variation_ratio[p(y|x)] = 1 - max_{y} p(y|x)

    Arguments:
        probs: Tensor[float], [N, Cl] or [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim in {2, 3}

    if probs.ndim == 3:
        probs = torch.mean(probs, dim=1)  # [N, Cl]

    max_probs, _ = torch.max(probs, dim=-1)  # [N,]

    scores = 1 - max_probs  # [N,]
    scores = check(scores, max_value=(1 - 1 / probs.shape[-1]), score_type="var_ratio")  # [N,]

    return scores  # [N,]
