"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import torch
from torch import Tensor


def accuracy_from_conditionals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    return count_correct_from_conditionals(predictions, labels) / len(predictions)  # [K,]


def accuracy_from_marginals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    return count_correct_from_marginals(predictions, labels) / len(predictions)  # [1,]


def count_correct_from_conditionals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    is_correct = torch.argmax(predictions, dim=-1) == labels[:, None]  # [N, K]
    return torch.sum(is_correct, dim=0)  #  [K,]


def count_correct_from_marginals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    is_correct = torch.argmax(predictions, dim=-1) == labels  # [N,]
    return torch.sum(is_correct)  #  [1,]


def nll_loss_from_probs(probs: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """
    Arguments:
        probs: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float]
    """
    reduce_fns = {"none": lambda x: x, "mean": torch.mean, "sum": torch.sum}
    reduce_fn = reduce_fns[reduction]

    liks = torch.gather(probs, dim=-1, index=labels[:, None]).flatten()  # [N,]
    liks = torch.clamp(liks, min=torch.finfo(liks.dtype).eps)  # [N,]
    nlls = -torch.log(liks)  # [N,]

    return reduce_fn(nlls)
