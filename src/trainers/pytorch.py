import math
from dataclasses import dataclass
from operator import gt, lt
from time import time
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import Generator
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logging import Dictionary, prepend_to_keys


@dataclass
class PyTorchTrainer:
    model: Module
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
    epig_cfg: DictConfig | None = None

    def __post_init__(self) -> None:
        self.optimizer = self.optimizer(params=self.model.parameters())
        self.validation_gap = max(1, int(self.n_optim_steps_max / self.n_validations))

    def set_rng_seed(self, seed: int) -> None:
        self.torch_rng.manual_seed(seed)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False
    ) -> Tuple[int, Dictionary]:
        log = Dictionary()
        start_time = time()

        step_range = range(self.n_optim_steps_max)
        step_range = tqdm(step_range, desc="Training") if verbose else step_range

        best_score = 0 if "acc" in self.early_stopping_metric else math.inf
        early_stopping_operator = gt if "acc" in self.early_stopping_metric else lt

        for step in step_range:
            train_metrics = self.train_step(train_loader)

            if step % self.validation_gap == 0:
                with torch.inference_mode():
                    val_metrics = self.test(val_loader)

                log_update = {
                    "time": time() - start_time,
                    "step": step,
                    **prepend_to_keys(train_metrics, "train"),
                    **prepend_to_keys(val_metrics, "val"),
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

        self.postprocess_model(train_loader)

        return step, log

    def postprocess_model(self, train_loader: DataLoader) -> None:
        pass
