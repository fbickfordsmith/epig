import math
from typing import Sequence


def compute_conv_output_size(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    n_output_channels: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    width = compute_conv_output_width(input_width, kernel_sizes, strides, padding, dilation)
    return n_output_channels * (width**2)


def compute_conv_output_width(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    References:
        https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    width = input_width
    for kernel_size, stride in zip(kernel_sizes, strides):
        width = width + (2 * padding) - (dilation * (kernel_size - 1)) - 1
        width = math.floor((width / stride) + 1)
    return width
