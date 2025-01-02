"""
N = number of examples
H = height
W = width
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from torch import Tensor
from torchvision.datasets import MNIST as TorchVisionMNIST

from src.data.datasets.base import BaseDataset, BaseEmbeddingDataset
from src.data.input_preprocessing import preprocess_inputs_for_zero_mean_and_unit_variance
from src.typing import Array


class BaseMNIST(BaseDataset):
    """
    If dataset = TorchVisionMNIST() then
    - dataset.data is a torch.Tensor with dtype torch.uint8, shape [N, H, W] and values in [0, 255]
    - dataset.targets is a torch.Tensor with dtype torch.int64
    - the number of examples in each class are [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949].
    """

    @staticmethod
    def preprocess_inputs_dtype_shape(inputs: Tensor) -> Tensor:
        inputs = inputs.numpy().astype(np.float32) / 255  # [N, 28, 28]
        return inputs[:, None, :, :]  # [N, 1, 28, 28]


class MNIST(TorchVisionMNIST, BaseMNIST):
    def __init__(
        self,
        data_dir: Path | str,
        train: bool = True,
        input_preprocessing: str | None = "zero_mean_and_unit_variance",
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir) / "mnist"

        super().__init__(root=data_dir, download=True, train=train, **kwargs)

        self.data = self.preprocess_inputs_dtype_shape(self.data)
        self.targets = self.targets.numpy()

        if input_preprocessing == "zero_mean_and_unit_variance":
            train_dataset = TorchVisionMNIST(data_dir)
            train_inputs = self.preprocess_inputs_dtype_shape(train_dataset.data)

            self = preprocess_inputs_for_zero_mean_and_unit_variance(self, train_inputs)

    def __getitem__(self, index: int) -> Tuple[Array, Array]:
        return self.data[index], self.targets[index]


class EmbeddingMNIST(BaseEmbeddingDataset):
    def __init__(self, data_dir: Path | str, **kwargs: Any) -> None:
        data_dir = Path(data_dir) / "mnist"
        super().__init__(data_dir=data_dir, **kwargs)
