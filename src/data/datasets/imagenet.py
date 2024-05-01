"""
F = number of features
N = number of examples
N_tr = number of training examples
"""

from pathlib import Path
from typing import Union

import numpy as np

from src.data.datasets.base import BaseEmbeddingDataset
from src.data.utils import preprocess_inputs_for_unit_norm, preprocess_inputs_for_unit_variance


class EmbeddingImageNet(BaseEmbeddingDataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        embedding_type: str,
        train: bool = True,
        input_preprocessing: str = "unit_variance",
    ) -> None:
        data_dir = Path(data_dir) / "imagenet"

        subset = "train" if train else "val"

        self.data = np.load(data_dir / f"embeddings_{embedding_type}_{subset}.npy")  # [N, F]
        self.targets = np.load(data_dir / f"labels_{subset}.npy")  # [N,]

        if input_preprocessing == "unit_norm":
            self = preprocess_inputs_for_unit_norm(self)

        elif input_preprocessing == "unit_variance":
            if train:
                train_inputs = self.data  # [N_tr, F]
            else:
                train_inputs = np.load(
                    data_dir / f"embeddings_{embedding_type}_train.npy"
                )  # [N_tr, F]

            self = preprocess_inputs_for_unit_variance(self, train_inputs)
