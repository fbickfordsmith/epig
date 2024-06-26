import logging
import math

import torch
from torch import Tensor


def check(
    scores: Tensor,
    min_value: float = 0.0,
    max_value: float = math.inf,
    epsilon: float = 1e-6,
    score_type: str = "",
) -> Tensor:
    """
    Warn if any element of scores is negative, a NaN or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores >= min_value - epsilon) & (scores <= max_value + epsilon)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")

    return scores
