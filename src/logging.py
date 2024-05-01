from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from subprocess import check_output
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


class Dictionary(dict):
    def append(self, dictionary: dict) -> None:
        for key in dictionary:
            if key in self:
                self[key] += [dictionary[key]]
            else:
                self[key] = [dictionary[key]]

    def extend(self, dictionary: dict) -> None:
        for key in dictionary:
            if key in self:
                self[key] += dictionary[key]
            else:
                self[key] = dictionary[key]

    def concatenate(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            if isinstance(dictionary[key][0], np.ndarray):
                dictionary[key] = np.concatenate(dictionary[key])
            elif isinstance(dictionary[key][0], Tensor):
                if dictionary[key][0].ndim == 0:
                    dictionary[key] = torch.tensor(dictionary[key])
                else:
                    dictionary[key] = torch.cat(dictionary[key])
            else:
                raise TypeError

        return dictionary

    def numpy(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = dictionary[key].numpy()

        return dictionary

    def torch(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = torch.tensor(dictionary[key])

        return dictionary

    def subset(self, inds: Sequence) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = dictionary[key][inds]

        return dictionary

    def save_to_csv(self, filepath: Path, formatting: Union[Callable, dict] = None) -> None:
        table = pd.DataFrame(self)

        if callable(formatting):
            table = table.applymap(formatting)

        elif isinstance(formatting, dict):
            for key in formatting:
                if key in self:
                    table[key] = table[key].apply(formatting[key])

        table.to_csv(filepath, index=False)

    def save_to_npz(self, filepath: Path) -> None:
        np.savez(filepath, **self)


def prepend_to_keys(dictionary: dict, string: str) -> dict:
    return {f"{string}_{key}": value for key, value in dictionary.items()}


def format_time(seconds: float) -> str:
    time = timedelta(seconds=seconds)
    assert time.days == 0
    hours, minutes, seconds = str(time).split(":")
    return f"{int(hours):02}:{minutes}:{float(seconds):02.0f}"


def get_formatters() -> dict:
    formatters = {
        "step": "{:05}".format,
        "time": format_time,
        "n_labels": "{:04}".format,
        "train_acc": "{:.4f}".format,
        "train_kl": "{:03.4f}".format,
        "train_mae": "{:.4f}".format,
        "train_mse": "{:.4f}".format,
        "train_nll": "{:.4f}".format,
        "val_acc": "{:.4f}".format,
        "val_mae": "{:.4f}".format,
        "val_mse": "{:.4f}".format,
        "val_nll": "{:.4f}".format,
        "test_acc": "{:.4f}".format,
        "test_mae": "{:.4f}".format,
        "test_mse": "{:.4f}".format,
        "test_nll": "{:.4f}".format,
    }
    return formatters


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


def save_repo_status(save_dir: Path) -> None:
    for filename, content in get_repo_status().items():
        with open(save_dir / filename, mode="wb") as _file:
            _file.write(content)


def set_up_wandb(cfg: DictConfig, slurm_job_id: int) -> None:
    # W&B supports nested config dictionaries (https://docs.wandb.ai/guides/track/config) but
    # the Hydra config needs converting first, using resolve=True to deal with interpolations
    # (https://docs.wandb.ai/guides/integrations/hydra#track-hyperparameters).
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_cfg["slurm_job_id"] = slurm_job_id
    wandb.init(config=wandb_cfg, **dict(cfg.wandb.init))

    # https://docs.wandb.ai/guides/track/log/customize-logging-axes
    wandb.define_metric("n_labels", hidden=True)  # Don't plot n_labels as a metric
    wandb.define_metric("test_*", step_metric="n_labels")  # Use n_labels as the x axis


def save_dir_contents_to_wandb(glob_str: str, base_dir: Path, policy: str = "live") -> None:
    wandb.save(glob_str=str(base_dir / glob_str), base_path=base_dir, policy=policy)


def save_run_to_wandb(
    results_dir: Path, results_subdirs: Sequence[Union[Path, str]], policy: str = "live"
) -> None:
    for subdir in results_subdirs:
        save_dir_contents_to_wandb(glob_str=f"{subdir}/*", base_dir=results_dir, policy=policy)

    save_dir_contents_to_wandb(glob_str=".hydra/*", base_dir=results_dir, policy=policy)
    save_dir_contents_to_wandb(glob_str="run_time.txt", base_dir=results_dir, policy=policy)
    save_dir_contents_to_wandb(glob_str="testing.csv", base_dir=results_dir, policy=policy)
