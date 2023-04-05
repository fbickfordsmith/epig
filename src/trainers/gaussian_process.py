"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import logging
import math
import torch
import warnings
from gpytorch import settings
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood, VariationalELBO
from operator import lt, gt
from src.math import logmeanexp
from src.models import VariationalGaussianProcess
from src.trainers.base import LogProbsTrainer, ProbsTrainer
from src.utils import Dictionary
from time import time
from torch.nn.functional import log_softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm


class GaussianProcessTrainer:
    """
    Mean-function and covariance-function hyperparameters:
    - Typically we're fine with zero mean. This is the default with gpytorch.means.ConstantMean.
    - The covariance-function output scale is important in determining predictive uncertainty when
      we are using a probit or softmax likelihood function.

    Suppress user warnings in __init__() to avoid getting this on every run (see GPyTorch issue):
    ```
    .../triangular_lazy_tensor.py:130: UserWarning: torch.triangular_solve is deprecated in
    favor of torch.linalg.solve_triangular and will be removed in a future PyTorch release...
    ```

    References:
        https://github.com/cornellius-gp/gpytorch/issues/689
    """

    def __init__(
        self,
        model: VariationalGaussianProcess,
        likelihood_fn: Likelihood,
        optimizer: Optimizer,
        n_optim_steps_min: int,
        n_optim_steps_max: int,
        n_samples_train: int,
        n_samples_test: int,
        n_validations: int,
        early_stopping_metric: str,
        early_stopping_patience: int,
        restore_best_model: bool,
        learning_rates: dict = None,
        init_mean: float = None,
        init_output_scale: float = None,
        init_length_scale: float = None,
        verbose: bool = False,
    ) -> None:
        warnings.simplefilter("ignore", UserWarning)
        self.model = model
        self.likelihood_fn = likelihood_fn
        self.initialize_model(init_mean, init_output_scale, init_length_scale)
        self.set_optimizer(optimizer, learning_rates)
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

    def initialize_model(
        self, init_mean: float, init_output_scale: float, init_length_scale: float
    ) -> None:
        if init_mean != None:
            self.model.mean_module.constant = init_mean

        if init_output_scale != None:
            self.model.covar_module.outputscale = init_output_scale

        if init_length_scale != None:
            self.model.covar_module.base_kernel.lengthscale = init_length_scale

    def set_optimizer(self, optimizer: Optimizer, learning_rates: dict) -> None:
        if learning_rates == None:
            self.optimizer = optimizer(params=self.model.parameters())
        else:
            parameters = []
            for name, parameter in self.model.named_parameters():
                if name in learning_rates.keys():
                    parameters.append({"params": parameter, "lr": learning_rates[name]})
                else:
                    parameters.append({"params": parameter})
            self.optimizer = optimizer(params=parameters)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dictionary:
        n_inputs = len(train_loader.dataset.indices)
        mll_fn = VariationalELBO(likelihood=self.likelihood_fn, model=self.model, num_data=n_inputs)

        log = Dictionary()
        start_time = time()

        step_range = range(self.n_optim_steps_max)
        step_range = tqdm(step_range, desc="Training") if self.verbose else step_range

        best_score = 0 if "acc" in self.early_stopping_metric else math.inf
        early_stopping_operator = gt if "acc" in self.early_stopping_metric else lt

        for step in step_range:
            train_loss = self.train_step(train_loader, mll_fn)

            if step % self.validation_gap == 0:
                with torch.inference_mode():
                    # Avoid overwriting train_loss. We want to see the true training loss when we
                    # inspect the logs. The loss returned by test() will be less noisy than the
                    # true loss because it is computed with more model samples.
                    train_acc, _ = self.test(train_loader)
                    val_acc, val_loss = self.test(val_loader)

                log_update = {
                    "time": time() - start_time,
                    "step": step,
                    "train_acc": train_acc.item(),
                    "train_loss": train_loss.item(),
                    "val_acc": val_acc.item(),
                    "val_loss": val_loss.item(),
                    "length_scale": self.model.covar_module.base_kernel.lengthscale.item(),
                    "output_scale": self.model.covar_module.outputscale.item(),
                    "mean": self.model.mean_module.constant.item(),
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

    def train_step(self, loader: DataLoader, mll_fn: MarginalLogLikelihood) -> Tensor:
        try:
            inputs, labels = next(loader)  # [N, ...], [N,]
        except:
            loader = iter(loader)
            inputs, labels = next(loader)  # [N, ...], [N,]

        self.model.train()
        self.optimizer.zero_grad()

        f_dist = self.model(inputs)

        with settings.num_likelihood_samples(self.n_samples_train):
            loss = -mll_fn(f_dist, labels)  # [1,]

        loss.backward()
        self.optimizer.step()

        return loss  # [1,]


class BernoulliGaussianProcessTrainer(GaussianProcessTrainer, ProbsTrainer):
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
        f_dist = self.model(inputs)
        f_dist = f_dist.to_data_independent_dist() if independent else f_dist

        logits = f_dist.sample(torch.Size([n_model_samples]))  # [K, N]
        logits = logits.permute(1, 0)  # [N, K]

        probs = self.likelihood_fn(logits).probs  # [N, K]
        return torch.stack((1 - probs, probs), dim=-1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        probs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]
        return torch.mean(probs, dim=1)  # [N, Cl]


class SoftmaxGaussianProcessTrainer(GaussianProcessTrainer, LogProbsTrainer):
    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        Since we don't use mixing weights here, we use syntax that looks like the underlying
        mathematics. If we were using mixing weights in the softmax likelihood function, we would
        have to instead use this syntax:
        >>> with settings.num_likelihood_samples(n_model_samples):
        >>>     categoricals = likelihood_fn(f_dist)
        >>> logprobs = categoricals.logits.permute(1, 0, 2)  # [N, K, Cl]

        Note that categoricals.logits are actually logprobs! This can be checked with
        >>> torch.allclose(torch.log(categoricals.probs), categoricals.logits)

        Arguments:
            inputs: Tensor[float], [N, ...]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        f_dist = self.model(inputs)
        f_dist = f_dist.to_data_independent_dist() if independent else f_dist

        logits = f_dist.sample(torch.Size([n_model_samples]))  # [K, N, Cl]
        logits = logits.permute(1, 0, 2)  # [N, K, Cl]

        return log_softmax(logits, dim=-1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        We pass independent=True to conditional_predict() because we always use marginal_predict()
        in cases where we don't want the test predictions to influence each other.
        """
        logprobs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]
        return logmeanexp(logprobs, dim=1)  # [N, Cl]
