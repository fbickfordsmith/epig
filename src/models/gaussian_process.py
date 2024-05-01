"""
F = number of features
N = number of examples
O = number of outputs
"""

from functools import partial

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    DeltaVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    MeanFieldVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from torch import Generator, Size, Tensor

from src.random import sample_truncated_gaussian


class GaussianProcess:
    def set_mean_fn(
        self, name: str, batch_shape: Size, limit: float = 0.1, torch_rng: Generator = None
    ) -> None:
        """
        The random-initialization procedure is based on the following.
        >>> ConstantMean().constant == torch.zeros(n_output_dims, 1)  # -> True
        """
        mean_fns = {"constant": ConstantMean, "zero": ZeroMean}
        mean_fn = mean_fns[name]
        mean_fn = mean_fn(batch_shape=batch_shape)

        if (name == "constant") and (torch_rng is not None):
            mean_fn.constant += sample_truncated_gaussian(mean_fn.constant, limit, torch_rng)

        self.mean_fn = mean_fn

    def set_covariance_fn(
        self,
        covariance_fn: str,
        inputs: Tensor,
        batch_shape: Size,
        ard: bool,
        pdist_init: bool,
        limit: float = 0.1,
        torch_rng: Generator = None,
    ) -> None:
        """
        The pdist_init option is based on code by Joost van Amersfoort and John Bradshaw (see
        references below).

        The random-initialization procedure is based on the following.
        >>> covariance_fn = ScaleKernel(RBFKernel(), batch_shape=batch_shape)
        >>> covariance_fn.base_kernel.lengthscale == softplus(covariance_fn.base_kernel.raw_lengthscale)  # -> True
        >>> covariance_fn.base_kernel.raw_lengthscale == torch.zeros(n_output_dims, 1, n_features)  # -> True
        >>> covariance_fn.outputscale == softplus(covariance_fn.raw_outputscale)  # -> True
        >>> covariance_fn.raw_outputscale == torch.zeros(n_output_dims)  # -> True

        References:
            https://github.com/y0ast/DUE/blob/main/due/dkl.py
            https://gist.github.com/john-bradshaw/e6784db56f8ae2cf13bb51eec51e9057#file-gpdnns-py-L93
        """
        if covariance_fn == "linear":
            covariance_fn = LinearKernel()

        else:
            if ard:
                assert inputs.ndim == 2
                covariance_kwargs = dict(ard_num_dims=inputs.shape[-1])
            else:
                covariance_kwargs = {}

            covariance_fns = {
                "matern12": partial(MaternKernel, nu=0.5),
                "matern32": partial(MaternKernel, nu=1.5),
                "matern52": partial(MaternKernel, nu=2.5),
                "rbf": RBFKernel,
                "rq": RQKernel,
            }
            covariance_fn = covariance_fns[covariance_fn]
            covariance_fn = covariance_fn(batch_shape=batch_shape, **covariance_kwargs)

            if pdist_init:
                assert len(inputs) > 1
                covariance_fn.lengthscale = torch.mean(torch.pdist(inputs))

            covariance_fn = ScaleKernel(covariance_fn, batch_shape=batch_shape)

            if torch_rng is not None:
                covariance_fn.base_kernel.lengthscale += sample_truncated_gaussian(
                    covariance_fn.base_kernel.lengthscale, limit, torch_rng
                )
                covariance_fn.outputscale += sample_truncated_gaussian(
                    covariance_fn.outputscale, limit, torch_rng
                )

        self.covariance_fn = covariance_fn

    def forward(self, inputs: Tensor) -> MultivariateNormal:
        mean = self.mean_fn(inputs)
        covariance = self.covariance_fn(inputs)

        return MultivariateNormal(mean, covariance)


class VariationalGaussianProcess(GaussianProcess, ApproximateGP):
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
        torch_rng: Generator = None,
    ) -> None:
        # Here "batch" corresponds to a "batch of GPs" so batch_shape = output_size.
        batch_shape = torch.Size([output_size]) if output_size > 1 else torch.Size([])

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

        batch_shape = batch_shape if ard else torch.Size([])

        super().__init__(variational_strategy=variational_strategy)

        self.set_mean_fn(mean_fn, batch_shape, torch_rng=torch_rng)

        self.set_covariance_fn(
            covariance_fn, inputs, batch_shape, ard, pdist_init, torch_rng=torch_rng
        )
