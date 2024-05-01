"""
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

import sys
from pathlib import Path
from typing import Sequence, Union

import torch
from batchbald_redux.consistent_mc_dropout import BayesianModule, ConsistentMCDropout2d
from torch import Tensor
from torch.nn import Conv2d, Linear, ReLU

from src.models.conv_net_batchbald_mcdo import MCDropoutFullyConnectedBlock


class MCDropoutBurgessConvBlock(BayesianModule):
    def __init__(self, dropout_rate: float, n_in: int, n_out: int) -> None:
        super().__init__()

        self.conv = Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=4, stride=2, padding=1)
        self.dropout = ConsistentMCDropout2d(p=dropout_rate)
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
        x = self.activation_fn(x)
        return x


class MCDropoutBurgessEncoder(BayesianModule):
    def __init__(self, input_shape: Sequence[int], output_size: int, dropout_rate: float) -> None:
        super().__init__()

        n_input_channels, image_height, image_width = input_shape

        self.conv1 = MCDropoutBurgessConvBlock(dropout_rate, n_in=n_input_channels, n_out=32)
        self.conv2 = MCDropoutBurgessConvBlock(dropout_rate, n_in=32, n_out=32)
        self.conv3 = MCDropoutBurgessConvBlock(dropout_rate, n_in=32, n_out=32)

        if image_height == image_width == 64:
            self.conv4 = MCDropoutBurgessConvBlock(dropout_rate, n_in=32, n_out=32)

        self.fc1 = MCDropoutFullyConnectedBlock(dropout_rate, n_in=(32 * 4 * 4), n_out=256)
        self.fc2 = MCDropoutFullyConnectedBlock(dropout_rate, n_in=256, n_out=256)
        self.fc3 = Linear(in_features=256, out_features=(2 * output_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) if hasattr(self, "conv4") else x
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.split(x, (x.shape[-1] // 2, x.shape[-1] // 2), dim=-1)


class MCDropoutBurgessConvNet(BayesianModule):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_size: int,
        dropout_rate: float,
        vae_dir: Union[Path, str],
        use_deterministic_encoder: bool = True,
    ) -> None:
        super().__init__()

        if use_deterministic_encoder:
            sys.path += [str(vae_dir)]

            from disvae.models.encoders import EncoderBurgess

            self.encoder = EncoderBurgess(input_shape, latent_dim=128)

        else:
            self.encoder = MCDropoutBurgessEncoder(
                input_shape, output_size=128, dropout_rate=dropout_rate
            )

        self.block1 = MCDropoutFullyConnectedBlock(dropout_rate, n_in=128, n_out=128)
        self.block2 = MCDropoutFullyConnectedBlock(dropout_rate, n_in=128, n_out=128)
        self.block3 = MCDropoutFullyConnectedBlock(dropout_rate, n_in=128, n_out=128)
        self.fc = Linear(in_features=128, out_features=output_size)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]

        Returns:
            Tensor[float], [N, O]
        """
        x, _ = self.encoder(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x
