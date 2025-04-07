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
    NaturalVariationalDistribution,
    TrilNaturalVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from torch import Generator, Size, Tensor

from src.random import sample_truncated_gaussian_like


class GaussianProcess:
    def set_mean_fn(
        self, name: str, batch_shape: Size, limit: float = 0.1, torch_rng: Generator | None = None
    ) -> None:
        """
        The random-initialization procedure is based on the following.
        >>> ConstantMean().constant == torch.zeros(n_output_dims, 1)  # -> True
        """
        mean_fns = {"constant": ConstantMean, "zero": ZeroMean}
        mean_fn = mean_fns[name]
        mean_fn = mean_fn(batch_shape=batch_shape)

        if (name == "constant") and (torch_rng is not None):
            sample = sample_truncated_gaussian_like(mean_fn.constant, limit, torch_rng=torch_rng)
            mean_fn.constant = mean_fn.constant + sample

        self.mean_fn = mean_fn

    def set_covariance_fn(
        self,
        covariance_fn: str,
        inputs: Tensor,
        batch_shape: Size,
        use_ard: bool,
        pdist_init: bool,
        limit: float = 0.1,
        torch_rng: Generator | None = None,
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
            if use_ard:
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
                length_scale_sample = sample_truncated_gaussian_like(
                    covariance_fn.base_kernel.lengthscale, limit, torch_rng=torch_rng
                )
                output_scale_sample = sample_truncated_gaussian_like(
                    covariance_fn.outputscale, limit, torch_rng=torch_rng
                )
                covariance_fn.base_kernel.lengthscale = (
                    covariance_fn.base_kernel.lengthscale + length_scale_sample
                )
                covariance_fn.outputscale = covariance_fn.outputscale + output_scale_sample

        self.covariance_fn = covariance_fn

    def forward(self, inputs: Tensor) -> MultivariateNormal:
        mean = self.mean_fn(inputs)
        covariance = self.covariance_fn(inputs)

        return MultivariateNormal(mean, covariance)


class VariationalGaussianProcess(GaussianProcess, ApproximateGP):
    """
    References:
    [1] https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/variational/unwhitened_variational_strategy.py
    [2] http://www.gatsby.ucl.ac.uk/~snelson/thesis.pdf - Section 2.2.1
    [3] https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.html
    """

    def __init__(
        self,
        inducing_inputs: Tensor,
        output_size: int,
        mean_fn: str = "constant",
        covariance_fn: str = "rbf",
        variational_form: str = "full",
        inducing_inputs_are_train_inputs: bool = True,
        optimize_inducing_inputs: bool = False,
        pdist_init: bool = False,
        use_ard: bool = False,
        torch_rng: Generator | None = None,
    ) -> None:
        # Here "batch" corresponds to a "batch of GPs" so batch_shape = output_size.
        batch_shape = torch.Size([output_size]) if output_size > 1 else torch.Size([])

        # Set the variational distribution over the latent-function values at the inducing inputs.
        variational_distributions = {
            "full": CholeskyVariationalDistribution,  # Full-covariance Gaussian
            "diag": MeanFieldVariationalDistribution,  # Diagonal-covariance Gaussian
            "delta": DeltaVariationalDistribution,  # Delta (equivalent to MAP estimation)
            "natural_fast": NaturalVariationalDistribution,  # Natural gradient decent (fast)
            "natural_tril": TrilNaturalVariationalDistribution,  # Natural gradient decent (stable)
        }
        variational_distribution = variational_distributions[variational_form](
            num_inducing_points=len(inducing_inputs), batch_shape=batch_shape
        )

        # We can skip whitening the inducing inputs if they are the training inputs, unless we are
        # using ARD. Skipping whitening speeds things up (see [1]).
        if inducing_inputs_are_train_inputs and not use_ard:
            variational_strategy = UnwhitenedVariationalStrategy
        else:
            variational_strategy = VariationalStrategy

        # Don't optimize the inducing inputs if they are the training inputs. We can't do better
        # than the training inputs themselves (see [2]).
        optimize_inducing_inputs &= not inducing_inputs_are_train_inputs

        variational_strategy = variational_strategy(
            model=self,
            inducing_points=inducing_inputs,
            variational_distribution=variational_distribution,
            learn_inducing_locations=optimize_inducing_inputs,
        )

        if output_size > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy, num_tasks=output_size
            )

        batch_shape = batch_shape if use_ard else torch.Size([])

        super().__init__(variational_strategy=variational_strategy)

        self.set_mean_fn(mean_fn, batch_shape, torch_rng=torch_rng)

        self.set_covariance_fn(
            covariance_fn, inducing_inputs, batch_shape, use_ard, pdist_init, torch_rng=torch_rng
        )
