"""
F = number of features
N = number of examples
N_tr = number of training examples
"""

from pathlib import Path
from subprocess import check_call
from typing import Any, Tuple, Union

import numpy as np
from numpy.lib.npyio import NpzFile

from src.data.datasets.base import BaseDataset, BaseEmbeddingDataset
from src.data.utils import split_indices_using_class_labels


class DSprites(BaseDataset):
    """
    DSprites dataset with one of the six underlying latents as the class label. The dataset is split
    into training and test sets. The split depends on three parameters: the latent used as the class
    label, the fraction of the dataset used for testing, and the random seed.

    References:
        https://github.com/google-deepmind/dsprites-dataset
    """

    filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    target_names = {"color": 0, "shape": 1, "scale": 2, "orientation": 3, "pos_x": 4, "pos_y": 5}

    def __init__(
        self,
        data_dir: Union[Path, str],
        train: bool = True,
        input_preprocessing: str = "unit_variance",
        test_fraction: float = 0.15,
        target_latent: str = "shape",
        seed: int = 0,
    ) -> None:
        data_dir = Path(data_dir) / "dsprites"

        if not (data_dir / self.filename).exists():
            self.download(data_dir)

        npz_file = np.load(data_dir / self.filename)

        inputs, labels = self.process_npz(npz_file, target_latent)

        rng = np.random.default_rng(seed=seed)

        train_inds, test_inds = split_indices_using_class_labels(labels, test_fraction, rng)

        if train:
            self.data = inputs[train_inds]
            self.targets = labels[train_inds]
        else:
            self.data = inputs[test_inds]
            self.targets = labels[test_inds]

        if input_preprocessing == "unit_variance":
            self.preprocess_inputs_for_unit_variance(inputs[train_inds])

    def download(self, data_dir: Path) -> None:
        url = f"https://github.com/google-deepmind/dsprites-dataset/raw/master/{self.filename}"
        data_dir.mkdir(parents=True, exist_ok=True)
        check_call(["curl", "--location", "--output", data_dir / self.filename, url])

    def process_npz(self, npz_file: NpzFile, target_latent: str) -> Tuple[np.ndarray, np.ndarray]:
        inputs = npz_file["imgs"]  # [N, 64, 64], np.uint8, values in {0, 1}
        inputs = inputs[:, None, :, :]  # [N, 1, 64, 64]
        inputs = inputs.astype(np.float32)  # [N, 1, 64, 64]

        labels = npz_file["latents_classes"]  # [N, 6], np.int64
        labels = labels[:, self.target_names[target_latent]]  # [N,]

        return inputs, labels  # [N, 1, 64, 64], [N,]

    def preprocess_inputs_for_unit_variance(self, train_inputs: np.ndarray) -> None:
        """
        If X is binary then X^2 = X and Var(X) = E[X^2] - E[X]^2 = E[X] - E[X]^2.
        """
        self.data_mean = np.sum(train_inputs) / train_inputs.size  # [1,]
        self.data_std = np.sqrt(self.data_mean - np.square(self.data_mean))  # [1,]
        self.data = (self.data - self.data_mean) / self.data_std  # [N, 1, 64, 64]


class EmbeddingDSprites(BaseEmbeddingDataset):
    def __init__(self, data_dir: Union[Path, str], **kwargs: Any) -> None:
        data_dir = Path(data_dir) / "dsprites"
        super().__init__(data_dir=data_dir, **kwargs)
