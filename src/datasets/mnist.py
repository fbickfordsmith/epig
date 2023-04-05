"""
N = number of examples
H = height
W = width
"""

import numpy as np
from pathlib import Path
from src.datasets.base import BaseDataset
from src.utils import Array
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from typing import Tuple, Union


class ImageMNIST(MNIST, BaseDataset):
    """
    If dataset = MNIST() then
    - dataset.data is a torch.Tensor with dtype torch.uint8, shape [N, H, W] and values in [0, 255]
    - dataset.targets is a torch.Tensor with dtype torch.int64
    - The number of examples per class is [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    """

    def __init__(
        self, data_dir: Union[str, Path], train: bool = True, normalize: bool = True, **kwargs
    ) -> None:
        data_dir = Path(data_dir) / "mnist"
        self.__class__.__name__ = "MNIST"

        super().__init__(root=data_dir, download=True, train=train, **kwargs)
        
        self.preprocess()
        
        if normalize:
            self.normalize(dataset=MNIST(data_dir))

    def __getitem__(self, index: int) -> Tuple[Array, Array]:
        return self.data[index], self.targets[index]
    
    def preprocess(self) -> None:
        self.data = self.data.numpy().astype(np.float32) / 255  # [N, 28, 28]
        self.data = self.data[:, None, :, :]  # [N, 1, 28, 28]
        self.targets = self.targets.numpy()  # [N,]

    def normalize(self, dataset: Dataset) -> None:
        data = dataset.data.numpy().astype(np.float32) / 255  # [N', 28, 28]
        self.mean = np.mean(data)  # [1,]
        self.std = np.std(data)  # [1,]
        self.data = (self.data - self.mean) / self.std  # [N, 1, 28, 28]