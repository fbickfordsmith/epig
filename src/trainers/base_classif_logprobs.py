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
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader

from src.math import logmeanexp
from src.metrics import count_correct_from_marginals
from src.trainers.base import DeterministicTrainer, StochasticTrainer
from src.uncertainty import (
    bald_from_logprobs,
    entropy_from_logprobs,
    epig_from_logprobs,
    epig_from_logprobs_using_matmul,
    epig_from_logprobs_using_weights,
    marginal_entropy_from_logprobs,
    mean_standard_deviation_from_logprobs,
    predictive_margin_from_logprobs,
    variation_ratio_from_logprobs,
)


class LogprobsClassificationDeterministicTrainer(DeterministicTrainer):
    """
    Base trainer for a deterministic classification model that outputs the logprobs of a categorical
    predictive distribution.
    """

    uncertainty_estimators = {
        "marg_entropy": entropy_from_logprobs,  # Not marginal_entropy_from_logprobs
        "pred_margin": predictive_margin_from_logprobs,
        "var_ratio": variation_ratio_from_logprobs,
    }

    def evaluate_test(self, inputs: Tensor, labels: Tensor, n_classes: int | None = None) -> dict:
        logprobs = self.predict(inputs)  # [N, Cl]

        if (n_classes is not None) and (n_classes < logprobs.shape[-1]):
            logprobs = logprobs[:, :n_classes]  # [N, n_classes]
            logprobs -= torch.logsumexp(logprobs, dim=-1, keepdim=True)  # [N, n_classes]

        n_correct = count_correct_from_marginals(logprobs, labels)  # [1,]
        nll = nll_loss(logprobs, labels, reduction="sum")  # [1,]

        return {"n_correct": n_correct, "nll": nll}

    def estimate_uncertainty_batch(self, inputs: Tensor, method: str) -> Tensor:
        uncertainty_estimator = self.uncertainty_estimators[method]

        logprobs = self.predict(inputs)  # [N, Cl]

        return uncertainty_estimator(logprobs)  # [N,]


class LogprobsClassificationStochasticTrainer(StochasticTrainer):
    """
    Base trainer for a stochastic classification model that outputs the logprobs of a categorical
    predictive distribution.
    """

    uncertainty_estimators = {
        "bald": bald_from_logprobs,
        "marg_entropy": marginal_entropy_from_logprobs,
        "mean_std": mean_standard_deviation_from_logprobs,
        "pred_margin": predictive_margin_from_logprobs,
        "var_ratio": variation_ratio_from_logprobs,
    }

    def evaluate_test(self, inputs: Tensor, labels: Tensor, n_classes: int | None = None) -> dict:
        logprobs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]

        if (n_classes is not None) and (n_classes < logprobs.shape[-1]):
            logprobs = logprobs[:, :n_classes]  # [N, n_classes]
            logprobs -= torch.logsumexp(logprobs, dim=-1, keepdim=True)  # [N, n_classes]

        n_correct = count_correct_from_marginals(logprobs, labels)  # [1,]
        nll = nll_loss(logprobs, labels, reduction="sum")  # [1,]

        return {"n_correct": n_correct, "nll": nll}

    def estimate_uncertainty_batch(self, inputs: Tensor, method: str) -> Tensor:
        uncertainty_estimator = self.uncertainty_estimators[method]

        logprobs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]

        return uncertainty_estimator(logprobs)  # [N,]

    def estimate_epig_batch(self, inputs_pool: Tensor, inputs_targ: Tensor) -> Tensor:
        logprobs = self.conditional_predict(
            torch.cat((inputs_pool, inputs_targ)), self.n_samples_test, independent=False
        )  # [N_p + N_t, K, Cl]

        logprobs_pool = logprobs[: len(inputs_pool)]  # [N_p, K, Cl]
        logprobs_targ = logprobs[len(inputs_pool) :]  # [N_t, K, Cl]

        if self.epig_cfg.use_matmul:
            scores = epig_from_logprobs_using_matmul(logprobs_pool, logprobs_targ)  # [N_p,]
        else:
            scores = epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p,]

        return scores  # [N_p,]

    def estimate_epig_using_target_class_dist(
        self, loader: DataLoader, n_input_samples: int | None = None
    ) -> Tensor:
        logprobs_cond = []

        for inputs, _ in loader:
            logprobs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            logprobs_cond += [logprobs_cond_i]

        logprobs_cond = torch.cat(logprobs_cond)  # [N, K, Cl]
        logprobs_marg = logmeanexp(logprobs_cond, dim=1)  # [N, Cl]
        logprobs_marg_marg = logmeanexp(logprobs_marg, dim=0, keepdim=True)  # [1, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        target_class_dist = self.epig_cfg.target_class_dist
        target_class_dist = torch.tensor([target_class_dist], device=inputs.device)  # [1, Cl]
        target_class_dist /= torch.sum(target_class_dist)  # [1, Cl]
        log_ratio = logprobs_marg - logprobs_marg_marg  # [N, Cl]
        weights = torch.sum(target_class_dist * torch.exp(log_ratio), dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = []

        if n_input_samples is not None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            logprobs_targ = logprobs_cond[inds]  # [N_s, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                if self.epig_cfg.use_matmul:
                    scores_i = epig_from_logprobs_using_matmul(
                        logprobs_cond_i, logprobs_targ
                    )  # [B,]
                else:
                    scores_i = epig_from_logprobs(logprobs_cond_i, logprobs_targ)  # [B,]

                scores += [scores_i.cpu()]

        else:
            logprobs_targ = logprobs_cond  # [N, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                scores_i = epig_from_logprobs_using_weights(
                    logprobs_cond_i, logprobs_targ, weights
                )  # [B,]
                scores += [scores_i.cpu()]

        return torch.cat(scores)  # [N,]
