"""
B = batch size
Cl = number of classes
E = embedding size
K = number of model samples
N = number of examples
"""

import logging
import math
import torch
from operator import lt, gt
from src.math import logmeanexp
from src.trainers.base import LogProbsTrainer
from src.utils import Dictionary
from time import time
from torch.nn import Module
from torch.nn.functional import log_softmax, nll_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm
from typing import Sequence, Tuple


class NeuralNetworkTrainer(LogProbsTrainer):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        n_optim_steps_min: int,
        n_optim_steps_max: int,
        n_samples_train: int,
        n_samples_test: int,
        n_validations: int,
        early_stopping_metric: str,
        early_stopping_patience: int,
        restore_best_model: bool,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer(params=self.model.parameters())
        self.n_optim_steps_min = n_optim_steps_min
        self.n_optim_steps_max = n_optim_steps_max
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.validation_gap = max(1, int(n_optim_steps_max / n_validations))
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_model = restore_best_model
        self.verbose = verbose

    def eval_mode(self) -> None:
        self.model.eval()

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]
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
            inputs: Tensor[float], [N, ...]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        logprobs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]

        if n_model_samples == 1:
            return torch.squeeze(logprobs, dim=1)  # [N, Cl]
        else:
            return logmeanexp(logprobs, dim=1)  # [N, Cl]

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dictionary:
        log = Dictionary()
        start_time = time()

        step_range = range(self.n_optim_steps_max)
        step_range = tqdm(step_range, desc="Training") if self.verbose else step_range

        best_score = 0 if "acc" in self.early_stopping_metric else math.inf
        early_stopping_operator = gt if "acc" in self.early_stopping_metric else lt

        for step in step_range:
            train_acc, train_loss = self.train_step(train_loader)

            if step % self.validation_gap == 0:
                with torch.inference_mode():
                    val_acc, val_loss = self.test(val_loader)

                log_update = {
                    "time": time() - start_time,
                    "step": step,
                    "train_acc": train_acc.item(),
                    "train_loss": train_loss.item(),
                    "val_acc": val_acc.item(),
                    "val_loss": val_loss.item(),
                }
                log.append(log_update)

                latest_score = log_update[self.early_stopping_metric]
                score_has_improved = early_stopping_operator(latest_score, best_score)

                if (step < self.n_optim_steps_min) or score_has_improved:
                    best_model_state = self.model.state_dict()
                    best_score = latest_score
                    patience_left = self.early_stopping_patience
                else:
                    patience_left -= self.validation_gap

                if (self.early_stopping_patience != -1) and (patience_left <= 0):
                    logging.info(f"Stopping training at step {step}")
                    break

        if self.restore_best_model:
            self.model.load_state_dict(best_model_state)

        return log

    def train_step(self, loader: DataLoader) -> Tuple[Tensor, Tensor]:
        try:
            inputs, labels = next(loader)
        except:
            loader = iter(loader)
            inputs, labels = next(loader)

        self.model.train()
        self.optimizer.zero_grad()

        n_correct, loss = self.evaluate(inputs, labels, self.n_samples_train)  # [1,], [1,]
        acc = n_correct / len(inputs)  # [1,]

        loss.backward()
        self.optimizer.step()

        return acc, loss  # [1,], [1,]

    def compute_badge_embeddings(
        self, loader: DataLoader, embedding_params: Sequence[str]
    ) -> Tensor:
        self.eval_mode()

        embeddings = []

        for inputs, _ in loader:
            pseudolosses = self.compute_pseudoloss(inputs)  # [B,]

            for pseudoloss in pseudolosses:
                # Prevent the grad attribute of each tensor accumulating a sum of gradients.
                self.model.zero_grad()

                pseudoloss.backward(retain_graph=True)

                gradients = []

                for name, param in self.model.named_parameters():
                    if name in embedding_params:
                        gradient = param.grad.flatten().cpu()  # [E_i,]
                        gradients.append(gradient)

                embedding = torch.cat(gradients)  # [E,]
                embeddings.append(embedding)

        return torch.stack(embeddings)  # [N, E]

    def compute_pseudoloss(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]

        Returns:
            Tensor[float], [N,]
        """
        logprobs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]
        pseudolabels = torch.argmax(logprobs, dim=-1)  # [N,]
        return nll_loss(logprobs, pseudolabels, reduction="none")  # [N,]