"""
Cl = number of classes
F = number of features
K = number of model samples
N = number of examples
"""

import torch
from torch import Tensor

from src.random import sample_gaussian
from src.trainers.base_classif_probs import ProbsClassificationStochasticTrainer
from src.trainers.gpytorch import GPyTorchTrainer


class GPyTorchClassificationBernoulliTrainer(GPyTorchTrainer, ProbsClassificationStochasticTrainer):
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
        f_dist = self.model(inputs)
        f_dist = f_dist.to_data_independent_dist() if independent else f_dist

        f_samples = sample_gaussian(f_dist, [n_model_samples], self.torch_rng)  # [K, N]
        f_samples = f_samples.permute(1, 0)  # [N, K]

        probs = self.likelihood_fn(f_samples).probs  # [N, K]

        return torch.stack((1 - probs, probs), dim=-1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        probs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]

        return torch.mean(probs, dim=1)  # [N, Cl]
