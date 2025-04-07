import math
import warnings
from dataclasses import dataclass
from operator import gt, lt
from time import time
from typing import Dict, Tuple

import torch
from gpytorch import settings
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood, VariationalELBO
from omegaconf import DictConfig
from torch import Generator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.utils import get_next_batch
from src.logging import Dictionary, prepend_to_keys
from src.models import VariationalGaussianProcess


@dataclass
class GPyTorchTrainer:
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

    model: VariationalGaussianProcess
    likelihood_fn: Likelihood
    optimizer: Optimizer
    torch_rng: Generator
    n_optim_steps_min: int
    n_optim_steps_max: int
    n_samples_train: int
    n_samples_test: int
    n_validations: int
    early_stopping_metric: str
    early_stopping_patience: int
    restore_best_model: bool
    learning_rates: Dict[str, float] | None = None
    init_mean: float | None = None
    init_output_scale: float | None = None
    init_length_scale: float | None = None
    epig_cfg: DictConfig | None = None

    def __post_init__(self) -> None:
        warnings.simplefilter("ignore", UserWarning)
        self.initialize_model(self.init_mean, self.init_output_scale, self.init_length_scale)
        self.set_optimizer(self.optimizer, self.learning_rates)
        self.validation_gap = max(1, int(self.n_optim_steps_max / self.n_validations))

    def eval_mode(self) -> None:
        self.model = self.model.eval()

    def set_rng_seed(self, seed: int) -> None:
        self.torch_rng.manual_seed(seed)

    def initialize_model(
        self, init_mean: float, init_output_scale: float, init_length_scale: float
    ) -> None:
        if init_mean is not None:
            self.model.mean_fn.constant = init_mean

        if init_output_scale is not None:
            self.model.covariance_fn.outputscale = init_output_scale

        if init_length_scale is not None:
            self.model.covariance_fn.base_kernel.lengthscale = init_length_scale

    def set_optimizer(self, optimizer: Optimizer, learning_rates: Dict[str, float] | None) -> None:
        if learning_rates is None:
            self.optimizer = optimizer(params=self.model.parameters())

        else:
            parameters = []

            for name, parameter in self.model.named_parameters():
                if name in learning_rates.keys():
                    parameters += [{"params": parameter, "lr": learning_rates[name]}]
                else:
                    parameters += [{"params": parameter}]

            self.optimizer = optimizer(params=parameters)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False
    ) -> Tuple[int, Dictionary]:
        mll_fn = VariationalELBO(
            likelihood=self.likelihood_fn, model=self.model, num_data=len(train_loader.dataset)
        )

        device = next(self.model.parameters()).device
        mll_fn = mll_fn.to(device)

        log = Dictionary()
        start_time = time()

        step_range = range(self.n_optim_steps_max)
        step_range = tqdm(step_range, desc="Training") if verbose else step_range

        best_score = 0 if "acc" in self.early_stopping_metric else math.inf
        early_stopping_operator = gt if "acc" in self.early_stopping_metric else lt

        for step in step_range:
            train_nll = self.train_step(train_loader, mll_fn)

            if step % self.validation_gap == 0:
                with torch.inference_mode():
                    # Avoid overwriting train_nll. We want to see the true training NLL when we
                    # inspect the logs. The NLL returned by test() will be less noisy than the
                    # true NLL because it is computed with more model samples (and potentially
                    # more training examples if we are using minibatching).
                    train_metrics = self.test(train_loader)
                    val_metrics = self.test(val_loader)

                train_metrics["nll"] = train_nll

                train_metrics = prepend_to_keys(train_metrics, "train")
                val_metrics = prepend_to_keys(val_metrics, "val")

                try:
                    model_hparams = {
                        "length_scale": self.model.covariance_fn.base_kernel.lengthscale.item(),
                        "output_scale": self.model.covariance_fn.outputscale.item(),
                        "mean": self.model.mean_fn.constant.item(),
                    }
                except:
                    model_hparams = {}

                log_update = {
                    "time": time() - start_time,
                    "step": step,
                    **train_metrics,
                    **val_metrics,
                    **model_hparams,
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
                    break

        if self.restore_best_model:
            self.model.load_state_dict(best_model_state)

        return step, log

    def train_step(self, loader: DataLoader, mll_fn: MarginalLogLikelihood) -> float:
        inputs, labels = get_next_batch(loader)  # [N, ...], [N,]

        self.model.train()

        f_dist = self.model(inputs)  # [N, ...]

        with settings.num_likelihood_samples(self.n_samples_train):
            nll = -mll_fn(f_dist, labels)  # [1,]

        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()

        return nll.item()  # [1,]
