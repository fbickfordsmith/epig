import logging
import math

import torch
from torch import Tensor


def check(
    scores: Tensor, min_value: float = 0.0, max_value: float = math.inf, score_type: str = ""
) -> Tensor:
    """
    Warn if any element of scores is a NaN or lies outside the range [min_value, max_value].
    """
    epsilon = 10 * torch.finfo(scores.dtype).eps

    if not torch.all((scores >= min_value - epsilon) & (scores <= max_value + epsilon)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(
            f"Invalid score (type = {score_type}, min = {min_score}, max = {max_score})"
        )

    return scores
