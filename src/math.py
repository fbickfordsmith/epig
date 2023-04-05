"""
Cl = number of classes
N = number of examples
"""

import math
import torch
from torch import Tensor
from torch.nn.functional import nll_loss


def accuracy(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    return count_correct(predictions, labels) / len(predictions)  # [1,]


def count_correct(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    return torch.sum(torch.argmax(predictions, dim=-1) == labels)  #  [1,]


def logmeanexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """
    Arguments:
        x: Tensor[float]
        dim: int
        keepdim: bool

    Returns:
        Tensor[float]
    """
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])


def nll_loss_from_probs(probs: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        probs: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    probs = torch.clamp(probs, min=torch.finfo(probs.dtype).eps)  # [N, Cl]
    return nll_loss(torch.log(probs), labels)  # [1,]
