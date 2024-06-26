"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

from typing import Sequence

from torch import Tensor
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU

from src.models.utils import compute_conv_output_size


class ConvBlock(Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int, dropout_rate: float) -> None:
        super().__init__()

        self.conv = Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size)
        self.dropout = Dropout(p=dropout_rate)
        self.maxpool = MaxPool2d(kernel_size=2)
        self.activation_fn = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch_in, H_in, W_in]

        Returns:
            Tensor[float], [N, Ch_out, H_out, W_out]
        """
        x = self.conv(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.activation_fn(x)

        return x


class FullyConnectedBlock(Module):
    def __init__(self, n_in: int, n_out: int, dropout_rate: float) -> None:
        super().__init__()

        self.fc = Linear(in_features=n_in, out_features=n_out)
        self.dropout = Dropout(p=dropout_rate)
        self.activation_fn = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, F_in]

        Returns:
            Tensor[float], [N, F_out]
        """
        x = self.fc(x)
        x = self.dropout(x)
        x = self.activation_fn(x)

        return x


class BatchBALD2BlockConvNet(Module):
    """
    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/03_consistent_mc_dropout.ipynb
    """

    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__()

        n_input_channels, _, image_width = input_shape

        block3_size = compute_conv_output_size(
            image_width, kernel_sizes=(2 * [5, 2]), strides=(2 * [1, 2]), n_output_channels=64
        )

        l_kwargs = dict(dropout_rate=dropout_rate)

        self.block1 = ConvBlock(n_in=n_input_channels, n_out=32, kernel_size=5, **l_kwargs)
        self.block2 = ConvBlock(n_in=32, n_out=64, kernel_size=5, **l_kwargs)
        self.block3 = FullyConnectedBlock(n_in=block3_size, n_out=128, **l_kwargs)
        self.fc = Linear(in_features=128, out_features=output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.block1(x)
        x = self.block2(x)

        x = x.flatten(start_dim=1)

        x = self.block3(x)
        x = self.fc(x)

        return x


class BatchBALD3BlockConvNet(Module):
    """
    References:
        https://arxiv.org/abs/1906.08158
    """

    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__()

        n_input_channels, _, image_width = input_shape

        block4_size = compute_conv_output_size(
            image_width, kernel_sizes=(3 * [3, 2]), strides=(3 * [1, 2]), n_output_channels=128
        )

        l_kwargs = dict(dropout_rate=dropout_rate)

        self.block1 = ConvBlock(n_in=n_input_channels, n_out=32, kernel_size=3, **l_kwargs)
        self.block2 = ConvBlock(n_in=32, n_out=64, kernel_size=3, **l_kwargs)
        self.block3 = ConvBlock(n_in=64, n_out=128, kernel_size=3, **l_kwargs)
        self.block4 = FullyConnectedBlock(n_in=block4_size, n_out=512, **l_kwargs)
        self.fc = Linear(in_features=512, out_features=output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.flatten(start_dim=1)

        x = self.block4(x)
        x = self.fc(x)

        return x
