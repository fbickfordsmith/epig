"""
F = number of features
N = number of examples
"""

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.datasets.base import BaseDataset
from src.data.input_preprocessing import preprocess_2d_inputs


class BaseUCI(BaseDataset):
    def __init__(
        self,
        data_dir: Path | str,
        train: bool = True,
        label_counts_test: Dict[int, int] | None = None,
        seed: int | None = None,
        verbose: bool = False,
        input_preprocessing: str | None = "zero_mean_and_unit_variance",
    ) -> None:
        data, n_test = self.download()

        # Separate the inputs (all columns but the last) from the labels (last column).
        self.data = data[:, :-1]  # [N, F]
        self.targets = data[:, -1]  # [N,]

        # Map the classes to start at zero.
        class_map = {_class: i for i, _class in enumerate(np.unique(self.targets))}
        class_map = np.vectorize(class_map.get)
        self.targets = class_map(self.targets)

        self.data = self.data.astype(np.float32)  # [N, F]

        if input_preprocessing is not None:
            train_inputs = self.data[:-n_test]
            self = preprocess_2d_inputs(self, train_inputs, input_preprocessing)

        if verbose:
            self.log_class_frequencies(self.targets, n_test)

        if train:
            self.data = self.data[:-n_test]
            self.targets = self.targets[:-n_test]

        else:
            is_test = np.full(len(self.data), False)
            is_test[-n_test:] = True

            rng = np.random.default_rng(seed=seed)
            test_inds = []

            for label, count in label_counts_test.items():
                _test_inds = np.flatnonzero(is_test & (self.targets == label))
                _test_inds = rng.choice(_test_inds, size=count, replace=False)
                test_inds += [_test_inds]

            test_inds = np.concatenate(test_inds)
            test_inds = rng.permutation(test_inds)

            self.data = self.data[test_inds]
            self.targets = self.targets[test_inds]

    def download(self) -> None:
        raise NotImplementedError

    def log_class_frequencies(self, labels: Sequence[int], n_test: int) -> None:
        """
        Report the class frequencies before and after making the train-test split.
        """
        free = np.full(len(labels), True)

        free_train = np.copy(free)
        free_train[-n_test:] = False

        free_test = np.copy(free)
        free_test[:-n_test] = False

        freqs_all = np.bincount(labels)
        freqs_train = np.bincount(labels[free_train])
        freqs_test = np.bincount(labels[free_test])

        rel_freqs_all = np.round(freqs_all / np.sum(freqs_all), 2)
        rel_freqs_train = np.round(freqs_train / np.sum(freqs_train), 2)
        rel_freqs_test = np.round(freqs_test / np.sum(freqs_test), 2)

        logging.info("Before split: " + str(freqs_all) + " " + str(rel_freqs_all))
        logging.info("Train after split: " + str(freqs_train) + " " + str(rel_freqs_train))
        logging.info("Test after split: " + str(freqs_test) + " " + str(rel_freqs_test))


class Magic(BaseUCI):
    def download(self) -> Tuple[np.ndarray, int]:
        """
        Use a fixed 70-30 train-test split.

        References:
            https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"

        data = pd.read_csv(url, header=None)
        data = data.sample(frac=1, random_state=0)
        data = data.to_numpy()

        n_test = int(0.3 * len(data))

        return data, n_test


class Satellite(BaseUCI):
    def download(self) -> Tuple[np.ndarray, int]:
        """
        References:
            https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/"

        train_data = pd.read_csv(url + "sat.trn", header=None, delim_whitespace=True)
        test_data = pd.read_csv(url + "sat.tst", header=None, delim_whitespace=True)

        data = pd.concat((train_data, test_data))
        data = data.to_numpy()

        n_test = len(test_data)

        return data, n_test


class Vowels(BaseUCI):
    def download(self) -> Tuple[np.ndarray, int]:
        """
        Columns: 0 = test, 1 = speaker, 2 = sex, 3-12 = features, 13 = class.

        References:
            https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Vowel+Recognition+-+Deterding+Data%29
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data"

        data = pd.read_csv(url, header=None, delim_whitespace=True)

        train_data = data[data[0] == 0]
        test_data = data[data[0] == 1]

        data = pd.concat((train_data, test_data))
        data = data.drop([0, 1, 2], axis="columns")
        data = data.to_numpy()

        n_test = len(test_data)

        return data, n_test
