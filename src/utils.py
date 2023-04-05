import numpy as np
import pandas as pd
import random
import torch
from copy import deepcopy
from datetime import timedelta
from numpy.random import Generator
from pathlib import Path
from subprocess import check_output
from torch import Tensor
from typing import Callable, Sequence, Union


Array = Union[np.ndarray, Tensor]


class Dictionary(dict):
    def append(self, update: dict) -> None:
        for key in update:
            try:
                self[key].append(update[key])
            except:
                self[key] = [update[key]]

    def extend(self, update: dict) -> None:
        for key in update:
            try:
                self[key].extend(update[key])
            except:
                self[key] = update[key]

    def concatenate(self) -> dict:
        scores = deepcopy(self)
        for key in scores.keys():
            scores[key] = torch.cat(scores[key])
        return scores

    def numpy(self) -> dict:
        scores = deepcopy(self)
        for key in scores.keys():
            scores[key] = scores[key].numpy()
        return scores

    def subset(self, inds: Sequence) -> dict:
        scores = deepcopy(self)
        for key in scores.keys():
            scores[key] = scores[key][inds]
        return scores

    def save_to_csv(self, filepath: Path, formatting: Union[Callable, dict] = None) -> None:
        table = pd.DataFrame(self)

        if callable(formatting):
            table = table.applymap(formatting)

        elif isinstance(formatting, dict):
            for key in formatting.keys():
                table[key] = table[key].apply(formatting[key])

        table.to_csv(filepath, index=False)

    def save_to_npz(self, filepath: Path) -> None:
        np.savez(filepath, **self)


def format_time(seconds: float) -> str:
    hours, minutes, seconds = str(timedelta(seconds=seconds)).split(":")
    return f"{int(hours):02}:{minutes}:{float(seconds):02.0f}"


def get_repo_status() -> dict:
    """
    References:
        https://stackoverflow.com/a/21901260
    """
    status = {
        "branch.txt": check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit.txt": check_output(["git", "rev-parse", "HEAD"]),
        "uncommitted.diff": check_output(["git", "diff"]),
    }
    return status


def set_rngs(seed: int = -1, constrain_cudnn: bool = False) -> Generator:
    """
    References:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed == -1:
        seed = random.randint(0, 1000)

    rng = np.random.default_rng(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if constrain_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return rng
