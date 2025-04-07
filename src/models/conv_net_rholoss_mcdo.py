"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

from typing import Sequence

from batchbald_redux.consistent_mc_dropout import BayesianModule
from torch import Tensor
from torch.nn import Linear

from src.models.conv_net_batchbald_mcdo import MCDropoutConvBlock, MCDropoutFullyConnectedBlock
from src.models.utils import compute_conv_output_size


class MCDropoutRHOLossConvNet(BayesianModule):
    """
    References:
        https://github.com/OATML/RHO-Loss/blob/main/src/models/modules/models.py#L81
    """

    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__()

        n_input_channels, _, image_width = input_shape

        block4_size = compute_conv_output_size(
            image_width, kernel_sizes=(3 * [3, 2]), strides=(3 * [1, 2]), n_output_channels=256
        )

        l_kwargs = dict(dropout_rate=dropout_rate)

        self.block1 = MCDropoutConvBlock(n_in=n_input_channels, n_out=64, kernel_size=3, **l_kwargs)
        self.block2 = MCDropoutConvBlock(n_in=64, n_out=128, kernel_size=3, **l_kwargs)
        self.block3 = MCDropoutConvBlock(n_in=128, n_out=256, kernel_size=3, **l_kwargs)
        self.block4 = MCDropoutFullyConnectedBlock(n_in=block4_size, n_out=128, **l_kwargs)
        self.block5 = MCDropoutFullyConnectedBlock(n_in=128, n_out=256, **l_kwargs)
        self.fc = Linear(in_features=256, out_features=output_size)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
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
        x = self.block5(x)
        x = self.fc(x)

        return x
