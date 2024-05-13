"""
B = batch size
Cl = number of classes
F = number of features
K = number of model samples
N = number of examples
N_t = number of target examples
"""

import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import count_correct_from_marginals, nll_loss_from_probs
from src.trainers.base import DeterministicTrainer, StochasticTrainer
from src.uncertainty import (
    bald_from_probs,
    entropy_from_probs,
    epig_from_probs,
    epig_from_probs_using_matmul,
    epig_from_probs_using_weights,
    marginal_entropy_from_probs,
    mean_standard_deviation_from_probs,
    predictive_margin_from_probs,
    variation_ratio_from_probs,
)


class ProbsClassificationDeterministicTrainer(DeterministicTrainer):
    """
    Base trainer for a deterministic classification model that outputs the probs of a categorical
    predictive distribution.
    """

    uncertainty_estimators = {
        "marg_entropy": entropy_from_probs,  # Not marginal_entropy_from_probs
        "pred_margin": predictive_margin_from_probs,
        "var_ratio": variation_ratio_from_probs,
    }

    def evaluate_test(self, inputs: Tensor, labels: Tensor, n_classes: int = None) -> dict:
        probs = self.predict(inputs)  # [N, Cl]

        if (n_classes is not None) and (n_classes < probs.shape[-1]):
            probs = probs[:, :n_classes]  # [N, n_classes]
            probs /= torch.sum(probs, dim=-1, keepdim=True)  # [N, n_classes]

        n_correct = count_correct_from_marginals(probs, labels)  # [1,]
        nll = nll_loss_from_probs(probs, labels, reduction="sum")  # [1,]

        return {"n_correct": n_correct, "nll": nll}

    def estimate_uncertainty_batch(self, inputs: Tensor, method: str) -> Tensor:
        uncertainty_estimator = self.uncertainty_estimators[method]

        probs = self.predict(inputs)  # [N, Cl]

        return uncertainty_estimator(probs)  # [N,]


class ProbsClassificationStochasticTrainer(StochasticTrainer):
    """
    Base trainer for a stochastic classification model that outputs the probs of a categorical
    predictive distribution.
    """

    uncertainty_estimators = {
        "bald": bald_from_probs,
        "marg_entropy": marginal_entropy_from_probs,
        "mean_std": mean_standard_deviation_from_probs,
        "pred_margin": predictive_margin_from_probs,
        "var_ratio": variation_ratio_from_probs,
    }

    def evaluate_test(self, inputs: Tensor, labels: Tensor, n_classes: int = None) -> dict:
        probs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]

        if (n_classes is not None) and (n_classes < probs.shape[-1]):
            probs = probs[:, :n_classes]  # [N, n_classes]
            probs /= torch.sum(probs, dim=-1, keepdim=True)  # [N, n_classes]

        n_correct = count_correct_from_marginals(probs, labels)  # [1,]
        nll = nll_loss_from_probs(probs, labels, reduction="sum")  # [1,]

        return {"n_correct": n_correct, "nll": nll}

    def estimate_uncertainty_batch(self, inputs: Tensor, method: str) -> Tensor:
        uncertainty_estimator = self.uncertainty_estimators[method]

        probs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]

        return uncertainty_estimator(probs)  # [N,]

    def estimate_epig_batch(self, inputs_pool: Tensor, inputs_targ: Tensor) -> Tensor:
        probs = self.conditional_predict(
            torch.cat((inputs_pool, inputs_targ)), self.n_samples_test, independent=False
        )  # [N_p + N_t, K, Cl]

        probs_pool = probs[: len(inputs_pool)]  # [N_p, K, Cl]
        probs_targ = probs[len(inputs_pool) :]  # [N_t, K, Cl]

        if self.epig_cfg.use_matmul:
            scores = epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]
        else:
            scores = epig_from_probs(probs_pool, probs_targ)  # [N_p,]

        return scores  # [N_p,]

    def estimate_epig_using_pool(self, loader: DataLoader, n_input_samples: int = None) -> Tensor:
        probs_cond = []

        for inputs, _ in loader:
            probs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            probs_cond += [probs_cond_i]

        probs_cond = torch.cat(probs_cond)  # [N, K, Cl]
        probs_marg = torch.mean(probs_cond, dim=1)  # [N, Cl]
        probs_marg_marg = torch.mean(probs_marg, dim=0, keepdim=True)  # [1, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        target_class_dist = self.epig_cfg.target_class_dist
        target_class_dist = torch.tensor([target_class_dist]).to(inputs.device)  # [1, Cl]
        target_class_dist /= torch.sum(target_class_dist)  # [1, Cl]
        weights = torch.sum(target_class_dist * probs_marg / probs_marg_marg, dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = []

        if n_input_samples is not None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            probs_targ = probs_cond[inds]  # [N_s, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                if self.epig_cfg.use_matmul:
                    scores_i = epig_from_probs_using_matmul(probs_cond_i, probs_targ)  # [B,]
                else:
                    scores_i = epig_from_probs(probs_cond_i, probs_targ)  # [B,]

                scores += [scores_i.cpu()]

        else:
            probs_targ = probs_cond  # [N, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                scores_i = epig_from_probs_using_weights(probs_cond_i, probs_targ, weights)  # [B,]
                scores += [scores_i.cpu()]

        return torch.cat(scores)  # [N,]
