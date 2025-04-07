"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

import math

import torch
from torch import Tensor

from src.math import logmeanexp
from src.uncertainty.epig_probs import epig_from_probs_using_matmul
from src.uncertainty.utils import check


def conditional_epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert logprobs_pool.ndim == logprobs_targ.ndim == 3

    _, _, Cl = logprobs_pool.shape

    # Estimate the log of the joint predictive distribution.
    logprobs_pool = logprobs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    logprobs_targ = logprobs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    logprobs_joint = logprobs_pool + logprobs_targ  # [N_p, N_t, K, Cl, Cl]
    logprobs_joint = logmeanexp(logprobs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the log of the marginal predictive distributions.
    logprobs_pool = logmeanexp(logprobs_pool, dim=2)  # [N_p, 1, Cl, 1]
    logprobs_targ = logmeanexp(logprobs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the log of the product of the marginal predictive distributions.
    logprobs_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    log_term = logprobs_joint - logprobs_joint_indep  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(torch.exp(logprobs_joint) * log_term, dim=(-2, -1))  # [N_p, N_t]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p, N_t]

    return scores  # [N_p, N_t]


def epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]

    return scores  # [N_p,]


def epig_from_logprobs_using_matmul(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    probs_pool = torch.exp(logprobs_pool)  # [N_p, K, Cl]
    probs_targ = torch.exp(logprobs_targ)  # [N_t, K, Cl]

    return epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]


def epig_from_logprobs_using_weights(
    logprobs_pool: Tensor, logprobs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
            = ∫ p_*(x_*) EPIG(x|x_*) dx_*
            ~= ∫ p_{pool}(x_*) w(x_*) EPIG(x|x_*) dx_*
            ~= (1 / M) ∑_{i=1}^M w(x_*^i) EPIG(x|x_*^i) where x_*^i in D_{pool}

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl], preds on proxy target inputs from the pool
        weights: Tensor[float], [N_t,], weight on each proxy target input

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]

    return scores  # [N_p,]
