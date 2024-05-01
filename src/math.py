import math
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import relu

from src.typing import Array


def logmeanexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """
    Numerically stable implementation of log(mean(exp(x))).
    """
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])


def log1pexp(x: Tensor) -> Tensor:
    """
    Numerically stable implementation of log(1 + exp(x)) which is equivalent to softplus with
    PyTorch's default scaling factor of 1.

    References:
        https://arxiv.org/abs/2301.08297
        https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
    """
    return torch.log1p(torch.exp(-torch.abs(x))) + relu(x)


def logexpm1(x: Tensor) -> Tensor:
    """
    Numerically stable implementation of log(exp(x) - 1) which is the inverse of log1pexp(x).

    References:
        https://arxiv.org/abs/2301.08297
        https://github.com/pytorch/pytorch/issues/72759#issuecomment-1236496693
    """
    return x + torch.log(-torch.expm1(-x))


def gaussian_entropy(variance: Union[float, Array]) -> Union[float, Array]:
    if isinstance(variance, float):
        log = math.log
    elif isinstance(variance, np.ndarray):
        log = np.log
    elif isinstance(variance, Tensor):
        log = torch.log
    else:
        raise ValueError

    return 0.5 * log(2 * math.pi * math.e * variance)
