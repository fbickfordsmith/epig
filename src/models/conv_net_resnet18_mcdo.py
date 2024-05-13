"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

from typing import Sequence

from batchbald_redux.consistent_mc_dropout import (
    BayesianModule,
    ConsistentMCDropout,
    ConsistentMCDropout2d,
)
from torch import Tensor
from torch.nn import Conv2d, Identity, Linear, ReLU, Sequential
from torchvision.models.resnet import BasicBlock, ResNet


class BaseMCDropoutResNet18(ResNet, BayesianModule):
    """
    References:
        https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18
        https://github.com/BlackHC/batchbald_redux/blob/master/03_consistent_mc_dropout.ipynb
        https://github.com/y0ast/pytorch-snippets/blob/main/minimal_cifar/train_cifar.py
    """

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


class MCDropoutResNet18V1(BaseMCDropoutResNet18):
    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=output_size)

        n_input_channels, _, _ = input_shape

        self.activation_fn = ReLU()
        self.conv1 = Conv2d(
            in_channels=n_input_channels, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.dropout1 = ConsistentMCDropout2d(p=dropout_rate)
        self.dropout2 = ConsistentMCDropout2d(p=dropout_rate)
        self.dropout3 = ConsistentMCDropout2d(p=dropout_rate)
        self.maxpool = Identity()

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.dropout3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)

        x = self.fc(x)

        return x


class MCDropoutResNet18V2(BaseMCDropoutResNet18):
    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=output_size)

        n_input_channels, _, _ = input_shape

        self.activation_fn = ReLU()
        self.conv1 = Conv2d(
            in_channels=n_input_channels, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.dropout_block = Sequential(
            Linear(self.fc.in_features, self.fc.in_features),
            ReLU(),
            ConsistentMCDropout(p=dropout_rate),
            Linear(self.fc.in_features, self.fc.in_features),
            ReLU(),
            ConsistentMCDropout(p=dropout_rate),
        )
        self.maxpool = Identity()

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)

        x = self.dropout_block(x)
        x = self.fc(x)

        return x
