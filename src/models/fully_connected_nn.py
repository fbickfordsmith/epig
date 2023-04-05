"""
F = number of features
N = number of examples
O = number of outputs
"""

import math
from batchbald_redux.consistent_mc_dropout import BayesianModule, ConsistentMCDropout
from torch import Tensor
from torch.nn import Linear, ModuleList, ReLU
from typing import Sequence


class FullyConnectedNeuralNetwork(BayesianModule):
    """
    References:
        https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py#L74
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        output_size: int,
        dropout_rate: float,
        hidden_sizes: Sequence[int],
    ) -> None:
        activation_fn = ReLU()
        sizes = (math.prod(input_shape), *hidden_sizes)

        super().__init__()

        dropout = ConsistentMCDropout(p=dropout_rate)
        self.layers = ModuleList()
        self.layers.append(dropout)

        for i in range(len(sizes) - 1):
            linear = Linear(in_features=sizes[i], out_features=sizes[i + 1])
            self.layers.append(linear)
            self.layers.append(activation_fn)

            if i < len(sizes) - 2:
                dropout = ConsistentMCDropout(p=dropout_rate)
                self.layers.append(dropout)

        linear = Linear(in_features=sizes[-1], out_features=output_size)
        self.layers.append(linear)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, F]

        Returns:
            Tensor[float], [N, O]
        """
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
