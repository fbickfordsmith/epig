"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

import torch
from src.math import logmeanexp
from src.uncertainty.bald import marginal_entropy_from_probs
from src.uncertainty.utils import check
from torch import Tensor


def conditional_epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the log of the joint predictive distribution.
    logprobs_pool = logprobs_pool.permute(1, 0, 2)  # [K, N_p, Cl]
    logprobs_targ = logprobs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    logprobs_pool = logprobs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
    logprobs_targ = logprobs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
    logprobs_pool_targ_joint = logprobs_pool + logprobs_targ  # [K, N_p, N_t, Cl, Cl]
    logprobs_pool_targ_joint = logmeanexp(logprobs_pool_targ_joint, dim=0)  # [N_p, N_t, Cl, Cl]

    # Estimate the log of the marginal predictive distributions.
    logprobs_pool = logmeanexp(logprobs_pool, dim=0)  # [N_p, 1, Cl, 1]
    logprobs_targ = logmeanexp(logprobs_targ, dim=0)  # [1, N_t, 1, Cl]

    # Estimate the log of the product of the marginal predictive distributions.
    logprobs_pool_targ_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    probs_pool_targ_joint = torch.exp(logprobs_pool_targ_joint)  # [N_p, N_t, Cl, Cl]
    log_term = logprobs_pool_targ_joint - logprobs_pool_targ_joint_indep  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    return scores  # [N_p, N_t]


def conditional_epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the joint predictive distribution.
    probs_pool = probs_pool.permute(1, 0, 2)  # [K, N_p, Cl]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_pool = probs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
    probs_targ = probs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
    probs_pool_targ_joint = probs_pool * probs_targ  # [K, N_p, N_t, Cl, Cl]
    probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)  # [N_p, N_t, Cl, Cl]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=0)  # [N_p, 1, Cl, 1]
    probs_targ = torch.mean(probs_targ, dim=0)  # [1, N_t, 1, Cl]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    nonzero_joint = probs_pool_targ_joint > 0  # [N_p, N_t, Cl, Cl]
    log_term = torch.clone(probs_pool_targ_joint)  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    return scores  # [N_p, N_t]


def epig_from_conditional_scores(scores: Tensor) -> Tensor:
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]


def epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


def epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


def epig_from_logprobs_using_matmul(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    See epig_from_probs_using_matmul.

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    probs_pool = torch.exp(logprobs_pool)  # [N_p, K, Cl]
    probs_targ = torch.exp(logprobs_targ)  # [N_t, K, Cl]
    return epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]


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
    N_t, K, C = probs_targ.shape

    entropy_pool = marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = marginal_entropy_from_probs(probs_targ)  # [N_t,]
    entropy_targ = torch.mean(entropy_targ)  # [1,]

    probs_pool = probs_pool.permute(0, 2, 1)  # [N_p, Cl, K]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_targ = probs_targ.reshape(K, N_t * C)  # [K, N_t * Cl]
    probs_pool_targ_joint = probs_pool @ probs_targ / K  # [N_p, Cl, N_t * Cl]

    entropy_pool_targ = (
        -torch.sum(probs_pool_targ_joint * torch.log(probs_pool_targ_joint), dim=(-2, -1)) / N_t
    )  # [N_p,]
    entropy_pool_targ[torch.isnan(entropy_pool_targ)] = 0.0

    scores = entropy_pool + entropy_targ - entropy_pool_targ  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]


def epig_from_logprobs_using_weights(
    logprobs_pool: Tensor, logprobs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
            = ∫ p_*(x_*) EPIG(x|x_*) dx_*
            ~= ∫ p_{pool}(x_*) w(x_*) EPIG(x|x_*) dx_*
            ~= (1 / M) ∑_{i=1}^M w(x_*^i) EPIG(x|x_*^i)  where  x_*^i in D_{pool}

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl], preds on proxy target inputs from the pool
        weights: Tensor[float], [N_t,], weight on each proxy target input

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


def epig_from_probs_using_weights(
    probs_pool: Tensor, probs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    See epig_from_logprobs_using_weights.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        weights: Tensor[float], [N_t,]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]