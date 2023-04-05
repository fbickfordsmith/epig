import numpy as np
import torch
from src.utils import Array
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple


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

    def numpy(self) -> None:
        if isinstance(self.data, Tensor):
            self.data = self.data.numpy()

        if isinstance(self.targets, Tensor):
            self.targets = self.targets.numpy()

    def torch(self) -> None:
        if isinstance(self.data, np.ndarray):
            self.data = torch.tensor(self.data)

        if isinstance(self.targets, np.ndarray):
            self.targets = torch.tensor(self.targets)

    def to(self, device: str) -> None:
        if isinstance(self.data, Tensor):
            self.data = self.data.to(device)

        if isinstance(self.targets, Tensor):
            self.targets = self.targets.to(device)
