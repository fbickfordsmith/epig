"""
Cl = number of classes
F = number of features
K = number of model samples
N = number of examples
"""

from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.cross_decomposition import PLSCanonical
from sklearn.ensemble import RandomForestClassifier
from torch import Tensor
from torch.utils.data import DataLoader

from src.trainers.base_classif_probs import ProbsClassificationStochasticTrainer


class SKLearnRandomForestClassificationTrainer(ProbsClassificationStochasticTrainer):
    def __init__(
        self,
        model: RandomForestClassifier,
        n_classes_pls: int | None = None,
        epig_cfg: DictConfig | None = None,
    ) -> None:
        self.model = model
        self.n_classes_pls = n_classes_pls
        self.n_samples_test = model.n_estimators  # Placeholder: this isn't used
        self.epig_cfg = epig_cfg
        self.use_val_data = False

    def eval_mode(self) -> None:
        pass

    def set_rng_seed(self, seed: int) -> None:
        pass

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        References:
            https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py

        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        transformed_inputs = self.transform(inputs)  # [N, F]

        probs = [tree.predict_proba(transformed_inputs) for tree in self.model.estimators_]
        probs = np.stack(probs, axis=1)  # [N, K, Cl]
        probs = torch.tensor(probs, device=inputs.device)  # [N, K, Cl]

        return probs  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        probs = self.model.predict_proba(self.transform(inputs))  # [N, Cl]
        probs = torch.tensor(probs, device=inputs.device)  # [N, Cl]

        return probs  # [N, Cl]

    def train(self, train_loader: DataLoader) -> Tuple[None, None]:
        inputs, labels = next(iter(train_loader))  # [N, *F], [N,]

        assert len(inputs) == len(train_loader.dataset)

        if self.n_classes_pls is not None:
            inputs = inputs.flatten(start_dim=1).cpu()  # [N, F]
            onehot_labels = np.eye(self.n_classes_pls)[labels.cpu().numpy()]  # [N, Cl]
            n_components = min(1, inputs.shape[0], inputs.shape[1], self.n_classes_pls)

            self.pls = PLSCanonical(n_components)
            self.pls.fit(inputs, onehot_labels)
            self.transform = lambda x: self.pls.transform(x.flatten(start_dim=1).cpu())

        else:
            self.transform = lambda x: x.flatten(start_dim=1).cpu()

        self.model.fit(self.transform(inputs), labels.cpu())

        return None, None
