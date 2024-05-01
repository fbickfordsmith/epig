import random
from typing import Union

import numpy as np
import torch
from numpy.random import Generator
from torch import Generator, Tensor
from torch.distributions import Categorical, Laplace, MultivariateNormal, Normal
from torch.distributions.multivariate_normal import _batch_mv
from torch.nn.init import trunc_normal_

from src.typing import Shape


def get_rng(seed: int = -1) -> Generator:
    """
    References:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed == -1:
        seed = random.randint(0, int(1e6))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    return np.random.default_rng(seed)


def sample_categorical(
    distribution: Categorical = None,
    logprobs: Tensor = None,
    probs: Tensor = None,
    sample_shape: Shape = [],
    torch_rng: Generator = None,
) -> Tensor:
    """
    Sample from a categorical distribution with optional control over the random-number generator,
    which is not possible using the sample() method of a torch.distributions.Categorical object.

    References:
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/categorical.py#L128
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/distribution.py#L12
    """
    # Ensure we have only one of {distribution, logprobs, probs} specified.
    assert np.sum([distribution is not None, logprobs is not None, probs is not None]) == 1

    sample_shape = torch.Size(sample_shape)

    if distribution is not None:
        probs = distribution.probs.reshape(-1, distribution._num_events)
        extended_sample_shape = distribution._extended_shape(sample_shape)

    else:
        # torch.multinomial() normalizes its inputs so we only need to handle normalization
        # ourselves if we have logprobs.
        if logprobs is not None:
            probs = torch.exp(logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True))

        assert probs.ndim >= 1

        # _extended_shape(sample_shape) = sample_shape + _batch_shape + _event_shape where for a
        # categorical distribution we have _batch_shape = probs.shape[:-1] and _event_shape = [].
        extended_sample_shape = sample_shape + probs.shape[:-1]

        probs = probs.reshape(-1, probs.shape[-1])

    sample = torch.multinomial(
        probs,
        num_samples=sample_shape.numel(),
        replacement=True,
        generator=torch_rng,
    )

    return sample.T.reshape(extended_sample_shape)


def sample_laplace(
    distribution: Laplace, sample_shape: Shape = [], torch_rng: Generator = None
) -> Tensor:
    """
    Sample from a Laplace distribution with optional control over the random-number generator, which
    is not possible using torch.distributions.Laplace().rsample().
    """
    sample_shape = distribution._extended_shape(sample_shape)
    finfo = torch.finfo(distribution.loc.dtype)

    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .uniform_()
        unif = torch.rand(
            sample_shape,
            generator=torch_rng,
            dtype=distribution.loc.dtype,
            device=distribution.loc.device,
        )
        unif = unif * 2 - 1
        log_1_minus_abs_unif = torch.log1p(-unif.abs().clamp(min=finfo.tiny))

    else:
        unif = distribution.loc.new(sample_shape).uniform_(-1 + finfo.eps, 1, generator=torch_rng)
        log_1_minus_abs_unif = torch.log1p(-unif.abs())

    return distribution.loc - distribution.scale * unif.sign() * log_1_minus_abs_unif


def sample_gaussian(
    distribution: Union[MultivariateNormal, Normal],
    sample_shape: Shape = [],
    torch_rng: Generator = None,
) -> Tensor:
    """
    Sample from a Gaussian distribution with optional control over the random-number generator,
    which is not possible using the rsample() methods of MultivariateNormal or Normal.

    If we wanted to take mean and std as arguments, we would use the following code:
    >>> sample_shape = torch.Size(sample_shape) + mean.shape
    >>> mean = mean.expand(sample_shape)
    >>> std = std.expand(sample_shape)
    """
    sample_shape = distribution._extended_shape(sample_shape)

    mean = distribution.mean.expand(sample_shape)
    std = distribution.stddev.expand(sample_shape)

    sample = torch.normal(torch.zeros_like(mean), torch.ones_like(std), generator=torch_rng)

    if isinstance(distribution, MultivariateNormal):
        return distribution.mean + _batch_mv(distribution._unbroadcasted_scale_tril, sample)
    else:
        return mean + std * sample


def sample_truncated_gaussian(
    tensor: Tensor, limit: float, torch_rng: Generator, mean: float = 0.0, std: float = 1.0
) -> Tensor:
    """
    Sample from a truncated Gaussian distribution, matching the shape and dtype of an input tensor.
    """
    sample = torch.empty_like(tensor)

    trunc_normal_(sample, mean, std, -limit, limit, torch_rng)

    return sample
