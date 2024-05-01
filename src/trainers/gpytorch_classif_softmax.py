"""
Cl = number of classes
F = number of features
K = number of model samples
N = number of examples
"""

from torch import Tensor
from torch.nn.functional import log_softmax

from src.math import logmeanexp
from src.random import sample_gaussian
from src.trainers.base_classif_logprobs import LogprobsClassificationStochasticTrainer
from src.trainers.gpytorch import GPyTorchTrainer


class GPyTorchClassificationSoftmaxTrainer(
    GPyTorchTrainer, LogprobsClassificationStochasticTrainer
):
    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        If we were using mixing weights in the softmax likelihood function, we would have to use
        this syntax:
        >>> with settings.num_likelihood_samples(n_model_samples):
        >>>     categoricals = likelihood_fn(f_dist)
        >>> logprobs = categoricals.logits.permute(1, 0, 2)  # [N, K, Cl]

        Note that categoricals.logits are actually logprobs! This can be checked with
        >>> torch.allclose(torch.log(categoricals.probs), categoricals.logits)

        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        f_dist = self.model(inputs)
        f_dist = f_dist.to_data_independent_dist() if independent else f_dist

        f_samples = sample_gaussian(f_dist, [n_model_samples], self.torch_rng)  # [K, N, Cl]
        f_samples = f_samples.permute(1, 0, 2)  # [N, K, Cl]

        return log_softmax(f_samples, dim=-1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        We pass independent=True to conditional_predict() because we always use marginal_predict()
        in cases where we don't want the test predictions to influence each other.
        """
        logprobs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]

        return logmeanexp(logprobs, dim=1)  # [N, Cl]
