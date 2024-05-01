"""
F = number of features
N = number of examples
N_tr = number of training examples
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.utils import preprocess_inputs_for_unit_norm, preprocess_inputs_for_unit_variance
from src.typing import Array, Shape


class BaseDataset(Dataset):
    """
    Note: we can change self.__class__.__name__ in __init__() to ensure self.raw_folder points to
    the right place (see PyTorch reference below).

    References:
        https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
        https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
        https://stackoverflow.com/a/21220030
    """

    def __getitem__(self, index: int) -> Tuple[Array, Array]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def input_shape(self) -> Shape:
        return self.data.shape[1:]

    @property
    def task(self) -> str:
        if isinstance(self.targets, np.ndarray):
            labels_are_floats = self.targets.dtype.kind == "f"
        else:
            labels_are_floats = torch.is_floating_point(self.targets)

        return "regression" if labels_are_floats else "classification"

    @property
    def n_classes(self) -> int:
        if self.task == "classification":
            if isinstance(self.targets, np.ndarray):
                return len(np.unique(self.targets))
            else:
                return len(torch.unique(self.targets))
        else:
            raise Exception

    @property
    def n_regression_targets(self) -> int:
        if self.task == "regression":
            if len(self.targets.shape) == 1:
                return 1
            else:
                return self.targets.shape[-1]
        else:
            raise Exception

    def numpy(self) -> Dataset:
        if isinstance(self.data, Tensor):
            self.data = self.data.numpy()

        if isinstance(self.targets, Tensor):
            self.targets = self.targets.numpy()

        return self

    def torch(self) -> Dataset:
        if isinstance(self.data, np.ndarray):
            self.data = torch.tensor(self.data)

        if isinstance(self.targets, np.ndarray):
            self.targets = torch.tensor(self.targets)

        return self

    def to(self, device: str) -> Dataset:
        if isinstance(self.data, Tensor):
            self.data = self.data.to(device)

        if isinstance(self.targets, Tensor):
            self.targets = self.targets.to(device)

        return self


class BaseEmbeddingDataset(BaseDataset):
    def __init__(
        self,
        data_dir: Path,
        embedding_type: str,
        train: bool = True,
        input_preprocessing: str = "unit_variance",
    ) -> None:
        subset = "train" if train else "test"

        self.data = np.load(data_dir / f"embeddings_{embedding_type}_{subset}.npy")  # [N, F]
        self.targets = np.load(data_dir / f"labels_{subset}.npy")  # [N,]

        if input_preprocessing == "unit_norm":
            self = preprocess_inputs_for_unit_norm(self)

        elif input_preprocessing == "unit_variance":
            if train:
                train_inputs = self.data  # [N_tr, F]
            else:
                embeddings_filename = f"embeddings_{embedding_type}_train.npy"
                train_inputs = np.load(data_dir / embeddings_filename)  # [N_tr, F]

            self = preprocess_inputs_for_unit_variance(self, train_inputs)
