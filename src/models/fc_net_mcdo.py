"""
F = number of features
N = number of examples
O = number of outputs
"""

import math
from typing import Callable, Sequence

from batchbald_redux.consistent_mc_dropout import BayesianModule, ConsistentMCDropout
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential


class MCDropoutFullyConnectedNet(BayesianModule):
    """
    References:
        https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py#L74
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        output_size: int,
        activation_fn: Callable = ReLU,
        dropout_rate: float = 0.1,
        use_input_dropout: bool = False,
    ) -> None:
        super().__init__()

        sizes = (math.prod(input_shape), *hidden_sizes)
        layers = []

        if use_input_dropout:
            layers += [ConsistentMCDropout(p=dropout_rate)]

        for i in range(len(sizes) - 1):
            layers += [Linear(in_features=sizes[i], out_features=sizes[i + 1])]
            layers += [activation_fn()]

            if i < len(sizes) - 2:
                layers += [ConsistentMCDropout(p=dropout_rate)]

        layers += [Linear(in_features=sizes[-1], out_features=output_size)]

        self.layers = Sequential(*layers)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, *F]

        Returns:
            Tensor[float], [N, O]
        """
        x = x.flatten(start_dim=1)  # [N, F]
        return self.layers(x)  # [N, O]
