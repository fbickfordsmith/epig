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
from numpy.random import Generator
from src.math import count_correct, logmeanexp, nll_loss_from_probs
from src.uncertainty import (
    bald_from_logprobs,
    bald_from_probs,
    epig_from_logprobs,
    epig_from_logprobs_using_matmul,
    epig_from_logprobs_using_weights,
    epig_from_probs,
    epig_from_probs_using_matmul,
    epig_from_probs_using_weights,
    marginal_entropy_from_logprobs,
    marginal_entropy_from_probs,
)
from src.utils import Dictionary
from torch import Tensor
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple


class Trainer:
    def __init__(self) -> None:
        pass

    def eval_mode(self) -> None:
        pass

    def conditional_predict(self, inputs: Tensor, n_model_samples: int, independent: bool) -> None:
        pass

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> None:
        pass

    def evaluate(self, inputs: Tensor, labels: Tensor, n_model_samples: int) -> None:
        pass

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        pass

    def test(self, loader: DataLoader) -> Tuple[Tensor, Tensor]:
        self.eval_mode()
        total_correct = total_loss = n_examples = 0

        for inputs, labels in loader:
            n_correct, loss = self.evaluate(inputs, labels, self.n_samples_test)  # [1,], [1,]
            total_correct += n_correct  # [1,]
            total_loss += loss * len(inputs)  # [1,]
            n_examples += len(inputs)  # [1,]

        acc = total_correct / n_examples  # [1,]
        loss = total_loss / n_examples  # [1,]
        return acc, loss  # [1,], [1,]

    def estimate_uncertainty(
        self,
        pool_loader: DataLoader,
        target_inputs: Tensor,
        mode: str,
        rng: Generator,
        epig_probs_target: List[float] = None,
        epig_probs_adjustment: List[float] = None,
        epig_using_matmul: bool = False,
    ) -> Dictionary:
        pool_loader = tqdm(pool_loader, desc="Uncertainty") if self.verbose else pool_loader

        if mode == "bald":
            scores = self.estimate_bald(pool_loader)

        elif mode == "epig":
            if epig_probs_target != None:
                scores = self.estimate_epig_using_pool(
                    pool_loader, epig_probs_target, epig_probs_adjustment, len(target_inputs)
                )
            else:
                scores = self.estimate_epig(pool_loader, target_inputs, epig_using_matmul)

        elif mode == "marg_entropy":
            scores = self.estimate_marginal_entropy(pool_loader)

        elif mode == "random":
            scores = self.sample_uniform(pool_loader, rng)

        return scores

    def estimate_marginal_entropy(self, loader: DataLoader) -> Dictionary:
        self.eval_mode()
        scores = Dictionary()

        for inputs, _ in loader:
            marg_entropy_scores = self.estimate_marginal_entropy_minibatch(inputs)  # [B,]
            scores.append({"marg_entropy": marg_entropy_scores.cpu()})

        return scores.concatenate()

    def estimate_marginal_entropy_minibatch(self, inputs: Tensor) -> None:
        pass

    def estimate_bald(self, loader: DataLoader) -> Dictionary:
        self.eval_mode()
        scores = Dictionary()

        for inputs, _ in loader:
            bald_scores = self.estimate_bald_minibatch(inputs)  # [B,]
            scores.append({"bald": bald_scores.cpu()})

        return scores.concatenate()

    def estimate_bald_minibatch(self, inputs: Tensor) -> None:
        pass

    def estimate_epig(
        self, loader: DataLoader, target_inputs: Tensor, use_matmul: bool
    ) -> Dictionary:
        self.eval_mode()
        scores = Dictionary()

        for inputs, _ in loader:
            epig_scores = self.estimate_epig_minibatch(inputs, target_inputs, use_matmul)  # [B,]
            scores.append({"epig": epig_scores.cpu()})

        return scores.concatenate()

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> None:
        pass

    def estimate_epig_using_pool(
        self,
        loader: DataLoader,
        probs_target: List[float],
        probs_adjustment: List[float],
        n_input_samples: int,
    ) -> None:
        pass

    def sample_uniform(self, loader: DataLoader, rng: Generator) -> Dictionary:
        n_inputs = len(loader.dataset.indices)
        samples = rng.uniform(size=n_inputs)
        samples = torch.tensor(samples)
        scores = Dictionary()
        scores.append({"random": samples})
        return scores.concatenate()


class LogProbsTrainer(Trainer):
    """
    Base trainer for a model that outputs log probabilities.
    """

    def evaluate(
        self, inputs: Tensor, labels: Tensor, n_model_samples: int
    ) -> Tuple[Tensor, Tensor]:
        logprobs = self.marginal_predict(inputs, n_model_samples)  # [N, Cl]
        n_correct = count_correct(logprobs, labels)  # [1,]
        loss = nll_loss(logprobs, labels)  # [1,]
        return n_correct, loss

    def estimate_marginal_entropy_minibatch(self, inputs: Tensor) -> Tensor:
        logprobs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]
        return marginal_entropy_from_logprobs(logprobs)  # [N,]

    def estimate_bald_minibatch(self, inputs: Tensor) -> Tensor:
        logprobs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]
        return bald_from_logprobs(logprobs)  # [N,]

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> Tensor:
        combined_inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]
        logprobs = self.conditional_predict(
            combined_inputs, self.n_samples_test, independent=False
        )  # [N + N_t, K, Cl]
        epig_fn = epig_from_logprobs_using_matmul if use_matmul else epig_from_logprobs
        return epig_fn(logprobs[: len(inputs)], logprobs[len(inputs) :])  # [N,]

    @torch.inference_mode()
    def estimate_epig_using_pool(
        self,
        loader: DataLoader,
        probs_target: List[float],
        probs_adjustment: List[float] = None,
        n_input_samples: int = None,
    ) -> Dictionary:
        self.eval_mode()

        logprobs_cond = []
        for inputs, _ in loader:
            logprobs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            logprobs_cond.append(logprobs_cond_i)
        logprobs_cond = torch.cat(logprobs_cond)  # [N, K, Cl]

        logprobs_marg = logmeanexp(logprobs_cond, dim=1)  # [N, Cl]
        logprobs_marg_marg = logmeanexp(logprobs_marg, dim=0, keepdim=True)  # [1, Cl]

        if probs_adjustment != None:
            probs_adjustment = torch.tensor([probs_adjustment])  # [1, Cl]
            probs_adjustment = probs_adjustment.to(inputs.device)  # [1, Cl]
            probs_marg = torch.exp(logprobs_marg)  # [N, Cl]
            probs_marg += probs_adjustment * torch.exp(logprobs_marg_marg)  # [N, Cl]
            probs_marg /= torch.sum(probs_marg, dim=-1, keepdim=True)  # [N, Cl]
            logprobs_marg = torch.log(probs_marg)  # [N, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        probs_target = torch.tensor([probs_target])  # [1, Cl]
        probs_target = probs_target.to(inputs.device)  # [1, Cl]
        log_ratio = logprobs_marg - logprobs_marg_marg  # [N, Cl]
        weights = torch.sum(probs_target * torch.exp(log_ratio), dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = Dictionary()

        if n_input_samples != None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            logprobs_target = logprobs_cond[inds]  # [N_s, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                epig_scores = epig_from_logprobs(logprobs_cond_i, logprobs_target)  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        else:
            logprobs_target = logprobs_cond  # [N, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                epig_scores = epig_from_logprobs_using_weights(
                    logprobs_cond_i, logprobs_target, weights
                )  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        return scores.concatenate()


class ProbsTrainer(Trainer):
    """
    Base trainer for a model that outputs probabilities.
    """

    def evaluate(
        self, inputs: Tensor, labels: Tensor, n_model_samples: int
    ) -> Tuple[Tensor, Tensor]:
        probs = self.marginal_predict(inputs, n_model_samples)  # [N, Cl]
        n_correct = count_correct(probs, labels)  # [1,]
        loss = nll_loss_from_probs(probs, labels)  # [1,]
        return n_correct, loss

    def estimate_marginal_entropy_minibatch(self, inputs: Tensor) -> Tensor:
        probs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]
        return marginal_entropy_from_probs(probs)  # [N,]

    def estimate_bald_minibatch(self, inputs: Tensor) -> Tensor:
        probs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]
        return bald_from_probs(probs)  # [N,]

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> Tensor:
        _inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]
        probs = self.conditional_predict(
            _inputs, self.n_samples_test, independent=False
        )  # [N + N_t, K, Cl]
        epig_fn = epig_from_probs_using_matmul if use_matmul else epig_from_probs
        return epig_fn(probs[: len(inputs)], probs[len(inputs) :])  # [N,]

    @torch.inference_mode()
    def estimate_epig_using_pool(
        self,
        loader: DataLoader,
        probs_target: List[float],
        probs_adjustment: List[float] = None,
        n_input_samples: int = None,
    ) -> Dictionary:
        self.eval_mode()

        probs_cond = []
        for inputs, _ in loader:
            probs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            probs_cond.append(probs_cond_i)
        probs_cond = torch.cat(probs_cond)  # [N, K, Cl]

        probs_marg = torch.mean(probs_cond, dim=1)  # [N, Cl]
        probs_marg_marg = torch.mean(probs_marg, dim=0, keepdim=True)  # [1, Cl]

        if probs_adjustment != None:
            probs_adjustment = torch.tensor([probs_adjustment])  # [1, Cl]
            probs_adjustment = probs_adjustment.to(inputs.device)  # [1, Cl]
            probs_marg += probs_adjustment * probs_marg_marg  # [N, Cl]
            probs_marg /= torch.sum(probs_marg, dim=-1, keepdim=True)  # [N, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        probs_target = torch.tensor([probs_target])  # [1, Cl]
        probs_target = probs_target.to(inputs.device)  # [1, Cl]
        weights = torch.sum(probs_target * probs_marg / probs_marg_marg, dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = Dictionary()

        if n_input_samples != None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            probs_target = probs_cond[inds]  # [N_s, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                epig_scores = epig_from_probs(probs_cond_i, probs_target)  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        else:
            probs_target = probs_cond  # [N, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                epig_scores = epig_from_probs_using_weights(
                    probs_cond_i, probs_target, weights
                )  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        return scores.concatenate()
