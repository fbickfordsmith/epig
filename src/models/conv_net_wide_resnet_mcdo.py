"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width

References:
    https://github.com/y0ast/DUE/blob/main/due/wide_resnet.py
"""

from typing import Callable, Sequence

from batchbald_redux.consistent_mc_dropout import BayesianModule, ConsistentMCDropout2d
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Identity, Linear, ReLU, Sequential
from torch.nn.functional import avg_pool2d
from torch.nn.init import constant_, kaiming_normal_


class MCDropoutWideBasicBlock(BayesianModule):
    def __init__(
        self,
        get_conv: Callable,
        n_input_channels: int,
        n_output_channels: int,
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.activation_fn = ReLU()

        self.bn1 = BatchNorm2d(n_input_channels)
        self.bn2 = BatchNorm2d(n_output_channels)

        self.conv1 = get_conv(n_input_channels, n_output_channels, kernel_size=3, stride=stride)
        self.conv2 = get_conv(n_output_channels, n_output_channels, kernel_size=3, stride=1)

        self.dropout = ConsistentMCDropout2d(p=dropout_rate)

        if (stride != 1) or (n_input_channels != n_output_channels):
            kernel_size = 1
            self.shortcut = get_conv(n_input_channels, n_output_channels, kernel_size, stride)
        else:
            self.shortcut = Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        shortcut_x = self.shortcut(x)

        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x += shortcut_x
        return x


class MCDropoutWideResNet(BayesianModule):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_size: int,
        dropout_rate: float = 0.3,
        depth: int = 28,
        width_multiplier: int = 10,
    ) -> None:
        super().__init__()

        assert (depth - 4) % 6 == 0, "Depth should be 6x + 4 where x is a positive integer"

        n_blocks = (depth - 4) // 6
        widths = (16, 16 * width_multiplier, 32 * width_multiplier, 64 * width_multiplier)
        strides = (1, 1, 2, 2)
        n_input_channels, _, _ = input_shape

        self.activation_fn = ReLU()

        self.bn = BatchNorm2d(widths[3])

        self.conv = self.get_conv(n_input_channels, widths[0], 3, strides[0])

        self.block1 = self.get_block(*widths[0:2], n_blocks, strides[1], dropout_rate)
        self.block2 = self.get_block(*widths[1:3], n_blocks, strides[2], dropout_rate)
        self.block3 = self.get_block(*widths[2:4], n_blocks, strides[3], dropout_rate)

        self.linear = Linear(widths[3], output_size)

        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, Linear):
                kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                constant_(module.bias, 0)

    @staticmethod
    def get_conv(
        n_input_channels: int, n_output_channels: int, kernel_size: int, stride: int
    ) -> Conv2d:
        padding = 1 if kernel_size == 3 else 0
        return Conv2d(n_input_channels, n_output_channels, kernel_size, stride, padding, bias=False)

    @staticmethod
    def get_block(
        n_input_channels: int,
        n_output_channels: int,
        n_blocks: int,
        stride: int,
        dropout_rate: float,
    ) -> Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []

        for stride in strides:
            layer = MCDropoutWideBasicBlock(
                MCDropoutWideResNet.get_conv,
                n_input_channels,
                n_output_channels,
                stride,
                dropout_rate,
            )
            layers += [layer]
            n_input_channels = n_output_channels

        return Sequential(*layers)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.conv(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.bn(x)
        x = self.activation_fn(x)
        x = avg_pool2d(x, kernel_size=x.shape[-1])

        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x

    def forward(self, input_B: Tensor, k: int) -> Tensor:
        """
        Copied from batchbald_redux.consistent_mc_dropout.BayesianModule.

        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, K, O]
        """
        BayesianModule.k = k
        mc_input_BK = BayesianModule.mc_tensor(input_B, k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K
