"""
Cl = number of classes
F = number of features
K = number of model samples
N = number of examples
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn.functional import log_softmax, nll_loss

from src.math import logmeanexp
from src.metrics import accuracy_from_conditionals
from src.trainers.base_classif_logprobs import LogprobsClassificationStochasticTrainer
from src.trainers.pytorch_classif import PyTorchClassificationTrainer


class PyTorchClassificationMCDropoutTrainer(
    PyTorchClassificationTrainer, LogprobsClassificationStochasticTrainer
):
    """
    Important: dropout masks are resampled at each call of self.eval_mode().
    """

    def eval_mode(self) -> None:
        self.model.eval()

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        features = self.model(inputs, n_model_samples)  # [N, K, Cl]
        return log_softmax(features, dim=-1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        logprobs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]

        if n_model_samples == 1:
            return torch.squeeze(logprobs, dim=1)  # [N, Cl]
        else:
            return logmeanexp(logprobs, dim=1)  # [N, Cl]

    def evaluate_train(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        loss = 1/KN ∑_{j=1}^K ∑_{i=1}^N L(x_i,y_i,θ_j) where θ_j ~ p(θ)

        Here we use
            L_1(x_i,y_i,θ_j) = nll_loss(x_i,y_i,θ_j) = -log p(y_i|x_i,θ_j)
            L_2(x_i,y_i,θ_j) = binary_loss(x_i,y_i,θ_j) = {argmax p(y|x_i,θ_j) != y_i}
        """
        logprobs = self.conditional_predict(
            inputs, self.n_samples_train, independent=False
        )  # [N, K, Cl]

        acc = accuracy_from_conditionals(logprobs, labels)  # [K,]
        acc = torch.mean(acc)  # [1,]

        nll_loss_fn = torch.vmap(nll_loss, in_dims=(1, None))

        nll = nll_loss_fn(logprobs, labels)  # [K,]
        nll = torch.mean(nll)  # [1,]

        return acc, nll  # [1,], [1,]

    def compute_badge_pseudoloss_v1(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N,]
        """
        logprobs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]
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
            self.model, (grad_params, no_grad_params), _input[None, :], dict(k=self.n_samples_test)
        )  # [1, K, Cl]

        logprobs = log_softmax(features, dim=-1)  # [1, K, Cl]
        logprobs = logmeanexp(logprobs, dim=1)  # [1, Cl]

        pseudolabel = torch.argmax(logprobs, dim=-1)  # [1,]

        return nll_loss(logprobs, pseudolabel)  # [1,]
