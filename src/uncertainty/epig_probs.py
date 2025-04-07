"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

import math

import torch
from torch import Tensor

from src.uncertainty.bald_probs import marginal_entropy_from_probs
from src.uncertainty.utils import check


def expected_joint_entropy_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    N_t, K, Cl = probs_targ.shape

    probs_pool = probs_pool.permute(0, 2, 1)  # [N_p, Cl, K]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_targ = probs_targ.reshape(K, N_t * Cl)  # [K, N_t * Cl]
    probs_joint = probs_pool @ probs_targ / K  # [N_p, Cl, N_t * Cl]

    scores = -torch.sum(torch.xlogy(probs_joint, probs_joint), dim=(-2, -1)) / N_t  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="JE")  # [N_p,]

    return scores  # [N_p,]


def conditional_epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log p(y,y_*|x,x_*) -
                  ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log p(y|x)p(y_*|x_*)

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    _, _, Cl = probs_pool.shape

    # Estimate the joint predictive distribution.
    probs_pool = probs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    probs_targ = probs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    probs_joint = probs_pool * probs_targ  # [N_p, N_t, K, Cl, Cl]
    probs_joint = torch.mean(probs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=2)  # [N_p, 1, Cl, 1]
    probs_targ = torch.mean(probs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    scores = torch.sum(torch.xlogy(probs_joint, probs_joint), dim=(-2, -1))  # [N_p, N_t]
    scores -= torch.sum(torch.xlogy(probs_joint, probs_pool_targ_indep), dim=(-2, -1))  # [N_p, N_t]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p, N_t]

    return scores  # [N_p, N_t]


def epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]

    return scores  # [N_p,]


def epig_from_probs_using_matmul(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x) = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = H[p(y|x)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]

    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).

    References:
        https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    _, _, Cl = probs_targ.shape

    entropy_pool = marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = marginal_entropy_from_probs(probs_targ)  # [N_t,]
    entropy_joint = expected_joint_entropy_from_probs(probs_pool, probs_targ)  # [N_p,]

    scores = entropy_pool + torch.mean(entropy_targ) - entropy_joint  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]


def epig_from_probs_using_weights(
    probs_pool: Tensor, probs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
            = ∫ p_*(x_*) EPIG(x|x_*) dx_*
            ~= ∫ p_{pool}(x_*) w(x_*) EPIG(x|x_*) dx_*
            ~= (1 / M) ∑_{i=1}^M w(x_*^i) EPIG(x|x_*^i) where x_*^i in D_{pool}

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        weights: Tensor[float], [N_t,]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]

    return scores  # [N_p,]
