"""
Cl = number of classes
F = number of features
N = number of examples
"""

from typing import Tuple

import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import Tensor
from torch.utils.data import DataLoader

from src.trainers.base import Trainer
from src.trainers.base_classif_probs import ProbsClassificationDeterministicTrainer


class SKLearnNeuralNetTrainer(Trainer):
    def __init__(self, model: BaseEstimator) -> None:
        self.model = model

    @staticmethod
    def transform(x: Tensor) -> Tensor:
        return x.flatten(start_dim=1).cpu()

    def eval_mode(self) -> None:
        pass

    def set_rng_seed(self, seed: int) -> None:
        pass

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]

        Returns:
            Tensor[float]
        """
        if isinstance(self.model, ClassifierMixin):
            preds = self.model.predict_proba(self.transform(inputs))  # [N, Cl]
        else:
            preds = self.model.predict(self.transform(inputs))  # [N,]

        return torch.tensor(preds, device=inputs.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[None, None]:
        inputs, labels = next(iter(train_loader))  # [N, *F], [N,]

        assert len(inputs) == len(train_loader.dataset)

        self.model.fit(self.transform(inputs), labels.cpu())

        return None, None


class SKLearnNeuralNetClassificationTrainer(
    SKLearnNeuralNetTrainer, ProbsClassificationDeterministicTrainer
):
    pass
