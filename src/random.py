import random

import numpy as np
import torch
from numpy.random import Generator
from torch import Generator, Tensor
from torch.distributions import Categorical, Gumbel, Laplace, MultivariateNormal, Normal, Uniform
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
    distribution: Categorical | None = None,
    logprobs: Tensor | None = None,
    probs: Tensor | None = None,
    sample_shape: Shape = [],
    torch_rng: Generator | None = None,
) -> Tensor:
    """
    Sample from a categorical distribution with optional control over the random-number generator,
    which is not possible using torch.distributions.Categorical().sample().

    References:
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/categorical.py
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/distribution.py
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


def sample_gaussian(
    distribution: MultivariateNormal | Normal,
    sample_shape: Shape = [],
    torch_rng: Generator | None = None,
) -> Tensor:
    """
    Sample from a Gaussian distribution with optional control over the random-number generator,
    which is not possible using torch.distributions.MultivariateNormal().sample() or
    torch.distributions.Normal().sample().

    If we wanted to take mean and std as arguments, we would use the following code:
    >>> sample_shape = torch.Size(sample_shape) + mean.shape
    >>> mean = mean.expand(sample_shape)
    >>> std = std.expand(sample_shape)

    References:
        https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/distributions/multivariate_normal.py
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/multivariate_normal.py
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/normal.py
    """
    sample_shape = distribution._extended_shape(sample_shape)
    mean = distribution.mean.expand(sample_shape)
    sample = torch.normal(torch.zeros_like(mean), torch.ones_like(mean), generator=torch_rng)

    if isinstance(distribution, MultivariateNormal):
        # Calling distribution._unbroadcasted_scale_tril can cause an error if distribution is a
        # lazily instantiated instance of gpytorch.distributions.MultivariateNormal.
        return distribution.mean + _batch_mv(distribution._unbroadcasted_scale_tril, sample)
    else:
        return distribution.mean + distribution.stddev * sample


def sample_gumbel(
    distribution: Gumbel, sample_shape: Shape = [], torch_rng: Generator | None = None
) -> Tensor:
    """
    Sample from a Gumbel distribution with optional control over the random-number generator, which
    is not possible using torch.distributions.Gumbel().sample().

    References:
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/gumbel.py
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/transformed_distribution.py
    """
    sample = sample_uniform(distribution.base_dist, sample_shape, torch_rng)

    for transform in distribution.transforms:
        sample = transform(sample)

    return sample


def sample_laplace(
    distribution: Laplace, sample_shape: Shape = [], torch_rng: Generator | None = None
) -> Tensor:
    """
    Sample from a Laplace distribution with optional control over the random-number generator, which
    is not possible using torch.distributions.Laplace().sample().

    References:
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/laplace.py
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


def sample_uniform(
    distribution: Uniform, sample_shape: Shape = [], torch_rng: Generator | None = None
) -> Tensor:
    """
    Sample from a uniform distribution with optional control over the random-number generator, which
    is not possible using torch.distributions.Uniform().sample().

    References:
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/uniform.py
    """
    sample_shape = distribution._extended_shape(sample_shape)

    sample = torch.rand(
        sample_shape,
        generator=torch_rng,
        dtype=distribution.low.dtype,
        device=distribution.low.device,
    )

    return distribution.low + sample * (distribution.high - distribution.low)


def sample_truncated_gaussian_like(
    x: Tensor, limit: float, mean: float = 0.0, std: float = 1.0, torch_rng: Generator | None = None
) -> Tensor:
    """
    Sample from a truncated Gaussian distribution, matching the shape, dtype, layout and device of
    the input tensor.
    """
    sample = torch.empty_like(x)

    trunc_normal_(sample, mean, std, -limit, limit, torch_rng)

    return sample
