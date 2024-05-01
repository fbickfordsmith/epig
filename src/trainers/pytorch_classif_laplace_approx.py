"""
Cl = number of classes
F = number of features
K = number of model samples
N = number of examples
"""

from typing import Any, Sequence, Tuple, Union

import torch
from laplace import DiagLaplace, DiagSubnetLaplace, ParametricLaplace
from laplace.utils import (
    LargestMagnitudeSubnetMask,
    LastLayerSubnetMask,
    ParamNameSubnetMask,
    RandomSubnetMask,
    SubnetMask,
)
from torch import Tensor
from torch.func import functional_call
from torch.nn.functional import log_softmax, nll_loss, softmax
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader

from src.metrics import accuracy_from_marginals
from src.trainers.base_classif_probs import ProbsClassificationStochasticTrainer
from src.trainers.pytorch_classif import PyTorchClassificationTrainer


class PyTorchClassificationLaplaceTrainer(
    PyTorchClassificationTrainer, ProbsClassificationStochasticTrainer
):
    def __init__(
        self,
        laplace_approx: ParametricLaplace,
        likelihood_temperature: Union[float, int, str] = 1,
        subnet_mask: SubnetMask = None,
        subnet_mask_inds: Sequence[int] = None,
        subnet_mask_names: Sequence[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.laplace_approx = laplace_approx
        self.likelihood_temperature = likelihood_temperature
        self.subnet_mask = subnet_mask
        self.subnet_mask_inds = subnet_mask_inds
        self.subnet_mask_names = subnet_mask_names

    def eval_mode(self) -> None:
        if isinstance(self.model, ParametricLaplace):
            self.model.model.eval()
        else:
            self.model.eval()

    def sample_model_parameters(self, n_model_samples: int, sample_on_cpu: bool = False) -> Tensor:
        """
        Sample from the parameter distribution with control over the random-number generator, which
        is not possible using the sample() method of DiagLaplace or DiagSubnetLaplace. If we want to
        handle self.model being another subclass of ParametricLaplace, we need to adapt sample()
        from that subclass.
        """
        assert isinstance(self.model, DiagLaplace)

        if isinstance(self.model, DiagSubnetLaplace):
            n_params = self.model.n_params_subnet
        else:
            n_params = self.model.n_params

        if sample_on_cpu:
            device = self.torch_rng.device

            seed = torch.randint(high=int(1e6), size=[1], generator=self.torch_rng, device=device)
            seed = seed.item()

            torch_rng_cpu = torch.Generator().manual_seed(seed)

            mean = self.model.mean.new_zeros(n_model_samples, n_params, device="cpu")  # [K, P]
            std = self.model.mean.new_ones(n_model_samples, n_params, device="cpu")  # [K, P]
            samples = torch.normal(mean, std, generator=torch_rng_cpu).to(device)  # [K, P]

        else:
            mean = self.model.mean.new_zeros(n_model_samples, n_params)  # [K, P]
            std = self.model.mean.new_ones(n_model_samples, n_params)  # [K, P]
            samples = torch.normal(mean, std, generator=self.torch_rng)  # [K, P]

        samples *= self.model.posterior_scale[None, :]  # [K, P]

        if isinstance(self.model, DiagSubnetLaplace):
            samples += self.model.mean_subnet[None, :]
            return self.model.assemble_full_samples(samples)
        else:
            samples += self.model.mean[None, :]
            return samples

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        The predictive_samples() method of ParametricLaplace takes a pred_type argument.

        pred_type="glm":
        - Idea: compute a Gaussian over latent-function values, sample from the Gaussian, then pass
          the sampled latent-function values through a softmax.
        - Issue: involves calling _glm_predictive_distribution(), which in turn involves computing
          a Jacobian matrix, which uses a lot of memory if the number of classes is big.

        pred_type="nn":
        - Idea: sample from the Gaussian over the model parameters, then for each sampled parameter
          configuration compute a forward pass through the model.
        - Issue: involves calling _nn_predictive_samples(), which does not allow for passing a
          random-number generator (needed to ensure samples are the same across data batches).

        We use pred_type="nn" and address the issue by reimplementing _nn_predictive_samples()
        within this function.

        References:
            https://github.com/aleximmer/Laplace/blob/main/laplace/baselaplace.py#L684

        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        assert isinstance(self.model, ParametricLaplace)

        probs = []

        for param_sample in self.sample_model_parameters(n_model_samples):
            # Set the model parameters to the sampled configuration.
            vector_to_parameters(param_sample, self.model.model.parameters())

            features = self.model.model(inputs)  # [N, Cl]
            probs += [softmax(features, dim=-1)]

        # Set the model parameters to the mean.
        vector_to_parameters(self.model.mean, self.model.model.parameters())

        return torch.stack(probs, dim=1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        if isinstance(self.model, ParametricLaplace):
            probs = self.conditional_predict(
                inputs, n_model_samples, independent=True
            )  # [N, K, Cl]
            return torch.mean(probs, dim=1)  # [N, Cl]

        else:
            features = self.model(inputs)  # [N, Cl]
            return softmax(features, dim=-1)  # [N, Cl]

    def evaluate_train(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        loss = 1/N ∑_{i=1}^N L(x_i,y_i,θ)

        Here we use
            L_1(x_i,y_i,θ) = nll_loss(x_i,y_i,θ) = -log p(y_i|x_i,θ)
            L_2(x_i,y_i,θ) = binary_loss(x_i,y_i,θ) = {argmax p(y|x_i,θ) != y_i}
        """
        features = self.model(inputs)  # [N, Cl]
        logprobs = log_softmax(features, dim=-1)  # [N, Cl]

        acc = accuracy_from_marginals(logprobs, labels)  # [1,]
        nll = nll_loss(logprobs, labels)  # [1,]

        return acc, nll  # [1,], [1,]

    def postprocess_model(self, train_loader: DataLoader) -> None:
        if self.subnet_mask is not None:
            simple_mask_fns = (LargestMagnitudeSubnetMask, LastLayerSubnetMask, RandomSubnetMask)

            if self.subnet_mask.func in simple_mask_fns:
                subnet_mask = self.subnet_mask(self.model)

            elif self.subnet_mask.func == ParamNameSubnetMask:
                param_names = []
                param_count = 0
                param_inds = {n: 0 for n in self.subnet_mask_names}

                for name, param in self.model.named_parameters():
                    for _name in self.subnet_mask_names:
                        if (_name in name) and (param_inds[_name] in self.subnet_mask_inds):
                            param_names += [name]
                            param_count += param.numel()
                            param_inds[_name] += 1

                subnet_mask = self.subnet_mask(self.model, param_names)

            else:
                raise ValueError

            laplace_approx_kwargs = dict(subnetwork_indices=subnet_mask.select())

        else:
            param_count = sum(param.numel() for param in self.model.parameters())

            laplace_approx_kwargs = {}

        if self.likelihood_temperature == "inverse_param_count":
            laplace_approx_kwargs["temperature"] = 1 / param_count
        elif isinstance(self.likelihood_temperature, (float, int)):
            laplace_approx_kwargs["temperature"] = self.likelihood_temperature
        else:
            raise NotImplementedError

        self.model = self.laplace_approx(self.model, **laplace_approx_kwargs)
        self.model.fit(train_loader)
        self.model.optimize_prior_precision(method="marglik")

    def compute_badge_pseudoloss_v1(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N,]
        """
        assert isinstance(self.model, ParametricLaplace)

        features = self.model.model(inputs)  # [N, Cl]
        logprobs = log_softmax(features, dim=-1)  # [N, Cl]
        pseudolabels = torch.argmax(logprobs, dim=-1)  # [N,]

        return nll_loss(logprobs, pseudolabels, reduction="none")  # [N,]

    def compute_badge_pseudoloss_v2(
        self, _input: Tensor, grad_params: dict, no_grad_params: dict
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [1, *F]

        Returns:
            Tensor[float], [1,]
        """
        features = functional_call(
            self.model, (grad_params, no_grad_params), _input[None, :]
        )  # [1, Cl]

        logprobs = log_softmax(features, dim=-1)  # [1, Cl]
        pseudolabel = torch.argmax(logprobs, dim=-1)  # [1,]

        return nll_loss(logprobs, pseudolabel)  # [1,]
