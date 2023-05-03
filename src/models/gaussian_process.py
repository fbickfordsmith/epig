"""
F = number of features
N = number of examples
O = number of outputs
"""

import torch
from functools import partial
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    DeltaVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    MeanFieldVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from numpy.random import Generator
from scipy.stats import truncnorm
from torch import Size, Tensor


class VariationalGaussianProcess(ApproximateGP):
    def __init__(
        self,
        inputs: Tensor,
        output_size: int,
        mean_fn: str = "constant",
        covariance_fn: str = "rbf",
        variational_form: str = "full",
        using_train_inputs: bool = True,
        learn_inducing_locations: bool = False,
        pdist_init: bool = False,
        ard: bool = False,
        rng: Generator = None,
    ) -> None:
        # Here "batch" corresponds to a "batch of GPs" so batch_shape = output_size.
        batch_shape = Size([output_size]) if output_size > 1 else Size([])

        # Set the variational distribution q(u) over the latent-function values at the inputs.
        variational_distributions = {
            "full": CholeskyVariationalDistribution,  # Full-covariance Gaussian
            "diag": MeanFieldVariationalDistribution,  # Diagonal-covariance Gaussian
            "delta": DeltaVariationalDistribution,  # Delta (equivalent to MAP estimation)
        }
        variational_distribution = variational_distributions[variational_form](
            num_inducing_points=len(inputs), batch_shape=batch_shape
        )

        # Skip whitening the inputs in order to speed things up, unless using ARD.
        if using_train_inputs and not ard:
            variational_strategy = UnwhitenedVariationalStrategy
        else:
            variational_strategy = VariationalStrategy

        # Don't optimize if using_train_inputs. We can't beat the training inputs themselves (see
        # Section 2.2.1 of http://www.gatsby.ucl.ac.uk/~snelson/thesis.pdf).
        learn_inducing_locations &= not using_train_inputs

        variational_strategy = variational_strategy(
            model=self,
            inducing_points=inputs,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        if output_size > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy, num_tasks=output_size
            )

        super().__init__(variational_strategy=variational_strategy)

        batch_shape = batch_shape if ard else Size([])

        self.set_mean_module(mean_fn, batch_shape, rng=rng)
        self.set_covariance_module(covariance_fn, inputs, batch_shape, ard, pdist_init, rng=rng)

    def set_mean_module(
        self, name: str, batch_shape: Size, limit: float = 0.1, rng: Generator = None
    ) -> None:
        """
        The random-initialization procedure is based on the following.
        >>> ConstantMean().constant == torch.zeros(n_output_dims, 1)  # -> True
        """
        means = {"constant": ConstantMean, "zero": ZeroMean}
        mean = means[name](batch_shape=batch_shape)

        if (name == "constant") and rng:
            mean.constant += self.sample_truncated_gaussian(mean.constant, limit=limit, rng=rng)

        self.mean_module = mean

    def set_covariance_module(
        self,
        name: str,
        inputs: Tensor,
        batch_shape: Size,
        ard: bool,
        pdist_init: bool,
        limit: float = 0.1,
        rng: Generator = None,
    ) -> None:
        """
        The pdist_init option is based on code by Joost van Amersfoort and John Bradshaw (see
        references below).

        The random-initialization procedure is based on the following.
        >>> covariance = ScaleKernel(RBFKernel(), batch_shape=batch_shape)
        >>> covariance.base_kernel.lengthscale == softplus(covariance.base_kernel.raw_lengthscale)  # -> True
        >>> covariance.base_kernel.raw_lengthscale == torch.zeros(n_output_dims, 1, n_features)  # -> True
        >>> covariance.outputscale == softplus(covariance.raw_outputscale)  # -> True
        >>> covariance.raw_outputscale == torch.zeros(n_output_dims)  # -> True

        References:
            https://github.com/y0ast/DUE/blob/main/due/dkl.py
            https://gist.github.com/john-bradshaw/e6784db56f8ae2cf13bb51eec51e9057#file-gpdnns-py-L93
        """
        if ard:
            assert inputs.ndim == 2
            covariance_kwargs = dict(ard_num_dims=inputs.shape[-1])
        else:
            covariance_kwargs = {}

        covariances = {
            "matern12": partial(MaternKernel, nu=0.5),
            "matern32": partial(MaternKernel, nu=1.5),
            "matern52": partial(MaternKernel, nu=2.5),
            "rbf": RBFKernel,
            "rq": RQKernel,
        }
        covariance = covariances[name](batch_shape=batch_shape, **covariance_kwargs)

        if pdist_init:
            assert len(inputs) > 1
            covariance.lengthscale = torch.mean(torch.pdist(inputs))

        covariance = ScaleKernel(covariance, batch_shape=batch_shape)

        if rng:
            covariance.base_kernel.lengthscale += self.sample_truncated_gaussian(
                covariance.base_kernel.lengthscale, limit=limit, rng=rng
            )
            covariance.outputscale += self.sample_truncated_gaussian(
                covariance.outputscale, limit=limit, rng=rng
            )

        self.covar_module = covariance

    def sample_truncated_gaussian(self, x: Tensor, limit: float, rng: Generator) -> Tensor:
        noise = truncnorm.rvs(-limit, limit, size=x.shape, random_state=rng)
        return torch.tensor(noise, dtype=x.dtype)

    def forward(self, inputs: Tensor) -> MultivariateNormal:
        mean = self.mean_module(inputs)
        covariance = self.covar_module(inputs)
        return MultivariateNormal(mean, covariance)