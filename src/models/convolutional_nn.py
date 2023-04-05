"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

from batchbald_redux.consistent_mc_dropout import (
    BayesianModule,
    ConsistentMCDropout,
    ConsistentMCDropout2d,
)
from src.models.utils import compute_conv_output_size
from torch import Tensor
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU
from typing import Sequence


class ConvBlockMC(BayesianModule):
    def __init__(self, dropout_rate: float, n_in: int, n_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size)
        self.dropout = ConsistentMCDropout2d(p=dropout_rate)
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


class FullyConnectedBlockMC(BayesianModule):
    def __init__(self, dropout_rate: float, n_in: int, n_out: int) -> None:
        super().__init__()
        self.fc = Linear(in_features=n_in, out_features=n_out)
        self.dropout = ConsistentMCDropout(p=dropout_rate)
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


class ConvolutionalNeuralNetwork(BayesianModule):
    """
    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/03_consistent_mc_dropout.ipynb
    """

    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        n_input_channels, _, image_width = input_shape
        fc1_size = compute_conv_output_size(
            image_width, kernel_sizes=(2 * (5, 2)), strides=(2 * (1, 2)), n_output_channels=64
        )
        super().__init__()
        self.block1 = ConvBlockMC(dropout_rate, n_in=n_input_channels, n_out=32, kernel_size=5)
        self.block2 = ConvBlockMC(dropout_rate, n_in=32, n_out=64, kernel_size=5)
        self.block3 = FullyConnectedBlockMC(dropout_rate, n_in=fc1_size, n_out=128)
        self.fc = Linear(in_features=128, out_features=output_size)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
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