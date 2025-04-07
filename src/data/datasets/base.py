from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.input_preprocessing import preprocess_2d_inputs
from src.data.label_preprocessing import preprocess_1d_labels
from src.data.utils import is_float
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
        return "regression" if is_float(self.targets) else "classification"

    @property
    def n_classes(self) -> int:
        if self.task == "classification":
            if isinstance(self.targets, np.ndarray):
                return len(np.unique(self.targets))
            else:
                return len(torch.unique(self.targets))
        else:
            raise Exception("n_classes is only defined for classification tasks")

    @property
    def n_regression_targets(self) -> int:
        if self.task == "regression":
            if len(self.targets.shape) == 1:
                return 1
            else:
                return self.targets.shape[-1]
        else:
            raise Exception("n_regression_targets is only defined for regression tasks")

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
        input_preprocessing: str | None = "zero_mean_and_unit_variance",
        label_preprocessing: str | None = None,
    ) -> None:
        if "imagenet" in str(data_dir):
            subset = "train" if train else "val"
        else:
            subset = "train" if train else "test"

        self.data = np.load(data_dir / f"embeddings_{embedding_type}_{subset}.npy")
        self.targets = np.load(data_dir / f"labels_{subset}.npy")

        if input_preprocessing is not None:
            if train:
                train_inputs = self.data
            else:
                embeddings_filepath = data_dir / f"embeddings_{embedding_type}_train.npy"
                train_inputs = np.load(embeddings_filepath)

            self = preprocess_2d_inputs(self, train_inputs, input_preprocessing)

        if label_preprocessing is not None:
            if train:
                train_labels = self.targets
            else:
                labels_filepath = data_dir / "labels_train.npy"
                train_labels = np.load(labels_filepath)

            self = preprocess_1d_labels(self, train_labels, label_preprocessing)
