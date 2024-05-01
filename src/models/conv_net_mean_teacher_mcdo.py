"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

from typing import Sequence

from batchbald_redux.consistent_mc_dropout import BayesianModule, ConsistentMCDropout2d
from torch import Tensor
from torch.nn import AvgPool2d, BatchNorm2d, Conv2d, LeakyReLU, Linear, MaxPool2d, Module
from torch.nn.utils import weight_norm

from src.models.utils import compute_conv_output_size, compute_conv_output_width


class MeanTeacherBlock(Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int, padding: int) -> None:
        super().__init__()

        self.activation_fn = LeakyReLU(negative_slope=0.1)
        self.bn = BatchNorm2d(num_features=n_out)
        self.conv = weight_norm(
            Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch_in, H_in, W_in]

        Returns:
            Tensor[float], [N, Ch_out, H_out, W_out]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        return x


class MCDropoutMeanTeacherConvNet(BayesianModule):
    """
    References:
        https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py#L336
    """

    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__()

        n_input_channels, _, image_width = input_shape
        fc_size = self.get_fc_size(image_width)

        self.block1a = MeanTeacherBlock(n_in=n_input_channels, n_out=128, kernel_size=3, padding=1)
        self.block1b = MeanTeacherBlock(n_in=128, n_out=128, kernel_size=3, padding=1)
        self.block1c = MeanTeacherBlock(n_in=128, n_out=128, kernel_size=3, padding=1)

        self.block2a = MeanTeacherBlock(n_in=128, n_out=256, kernel_size=3, padding=1)
        self.block2b = MeanTeacherBlock(n_in=256, n_out=256, kernel_size=3, padding=1)
        self.block2c = MeanTeacherBlock(n_in=256, n_out=256, kernel_size=3, padding=1)

        self.block3a = MeanTeacherBlock(n_in=256, n_out=512, kernel_size=3, padding=0)
        self.block3b = MeanTeacherBlock(n_in=512, n_out=256, kernel_size=1, padding=0)
        self.block3c = MeanTeacherBlock(n_in=256, n_out=128, kernel_size=1, padding=0)

        self.dropout1 = ConsistentMCDropout2d(p=dropout_rate)
        self.dropout2 = ConsistentMCDropout2d(p=dropout_rate)

        self.fc = weight_norm(Linear(in_features=fc_size, out_features=output_size))

        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)

        self.meanpool = AvgPool2d(kernel_size=6, stride=2)

    @staticmethod
    def get_fc_size(image_width: int) -> int:
        """
        LAYER_width is the width of the representation at the input to LAYER.
        """
        maxpool1_width = compute_conv_output_width(
            image_width,
            kernel_sizes=(3, 3, 3),
            strides=(1, 1, 1),
            padding=1,
        )
        conv2a_width = compute_conv_output_width(
            maxpool1_width,
            kernel_sizes=(2,),
            strides=(2,),
        )
        maxpool2_width = compute_conv_output_width(
            conv2a_width,
            kernel_sizes=(3, 3, 3),
            strides=(1, 1, 1),
            padding=1,
        )
        conv3a_width = compute_conv_output_width(
            maxpool2_width,
            kernel_sizes=(2,),
            strides=(2,),
        )
        meanpool_width = compute_conv_output_width(
            conv3a_width,
            kernel_sizes=(3, 1, 1),
            strides=(1, 1, 1),
        )
        fc_size = compute_conv_output_size(
            meanpool_width,
            kernel_sizes=(6,),
            strides=(2,),
            n_output_channels=128,
        )
        return fc_size

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.block1a(x)
        x = self.block1b(x)
        x = self.block1c(x)

        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.block2a(x)
        x = self.block2b(x)
        x = self.block2c(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block3c(x)

        x = self.meanpool(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
