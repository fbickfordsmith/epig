"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from src.trainers.base import ProbsTrainer
from torch.utils.data import DataLoader
from torch import Tensor


class RandomForestTrainer(ProbsTrainer):
    def __init__(self, model: RandomForestClassifier, verbose: bool = False) -> None:
        self.model = model
        self.n_samples_test = model.n_estimators  # Placeholder: this isn't used
        self.verbose = verbose

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        References:
            https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py#L862

        Arguments:
            inputs: Tensor[float], [N, ...]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        probs = [tree.predict_proba(inputs.cpu()) for tree in self.model.estimators_]
        probs = np.stack(probs, axis=1)  # [N, K, Cl]
        probs = torch.tensor(probs)  # [N, K, Cl]
        probs = probs.to(inputs.device)  # [N, K, Cl]
        return probs  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        probs = self.model.predict_proba(inputs.cpu())  # [N, Cl]
        probs = torch.tensor(probs)  # [N, Cl]
        probs = probs.to(inputs.device)  # [N, Cl]
        return probs  # [N, Cl]

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        inputs, labels = next(iter(train_loader))

        assert len(inputs) == len(train_loader.dataset.indices)

        self.model.fit(inputs.cpu(), labels.cpu())
        return None
