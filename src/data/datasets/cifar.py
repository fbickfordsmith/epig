"""
N = number of examples
Ch = number of channels
H = height
W = width
"""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
from torchvision.datasets import CIFAR10 as TorchVisionCIFAR10

from src.data.datasets.base import BaseDataset, BaseEmbeddingDataset
from src.data.utils import preprocess_inputs_for_unit_variance
from src.typing import Array


class BaseCIFAR(BaseDataset):
    """
    If dataset = TorchVisionCIFAR10() then
    - dataset.data is an np.array with dtype np.uint8, shape [N, H, W, Ch] and values in [0, 255]
    - dataset.targets is a list
    """

    @staticmethod
    def preprocess_inputs_dtype_shape(inputs: np.ndarray) -> np.ndarray:
        inputs = inputs.astype(np.float32) / 255  # [N, H, W, Ch]
        return inputs.transpose(0, 3, 1, 2)  # [N, Ch, H, W]


class CIFAR10(TorchVisionCIFAR10, BaseCIFAR):
    def __init__(
        self,
        data_dir: Union[Path, str],
        train: bool = True,
        input_preprocessing: str = "unit_variance",
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir) / "cifar10"

        super().__init__(root=data_dir, download=True, train=train, **kwargs)

        self.data = self.preprocess_inputs_dtype_shape(self.data)
        self.targets = np.array(self.targets)

        if input_preprocessing == "unit_variance":
            train_dataset = TorchVisionCIFAR10(data_dir)
            train_inputs = self.preprocess_inputs_dtype_shape(train_dataset.data)
            self = preprocess_inputs_for_unit_variance(self, train_inputs, axis=(0, 2, 3))

    def __getitem__(self, index: int) -> Tuple[Array, Array]:
        return self.data[index], self.targets[index]


class EmbeddingCIFAR10(BaseEmbeddingDataset):
    def __init__(self, data_dir: Union[Path, str], **kwargs: Any) -> None:
        data_dir = Path(data_dir) / "cifar10"
        super().__init__(data_dir=data_dir, **kwargs)
