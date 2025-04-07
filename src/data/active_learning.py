from typing import Callable, Sequence

import numpy as np
from numpy.random import Generator
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data.datasets.base import BaseDataset
from src.typing import ConfigDict, ConfigList


class ActiveLearningData:
    def __init__(
        self,
        dataset: BaseDataset,
        device: str,
        rng: Generator,
        batch_sizes: ConfigDict,
        label_counts_main: ConfigDict,
        label_counts_test: ConfigDict | ConfigList | int,
        class_map: ConfigDict | ConfigList | None = None,
        loader_kwargs: ConfigDict | None = None,
    ) -> None:
        self.main_dataset = dataset(train=True)
        self.test_dataset = dataset(train=False)

        self.main_inds = {}

        free_inds = np.arange(len(self.main_dataset), dtype=int)

        for subset in label_counts_main.keys():
            selected_inds = initialize_indices(
                self.main_dataset.targets[free_inds], label_counts_main[subset], rng
            )  # Index into free_inds
            selected_inds = free_inds[selected_inds]  # Index into self.main_dataset

            self.main_inds[subset] = selected_inds.tolist()

            free_inds = np.setdiff1d(free_inds, selected_inds)

        self.test_inds = initialize_indices(self.test_dataset.targets, label_counts_test, rng)
        self.test_inds = self.test_inds.tolist()

        if class_map is not None:
            self.main_dataset = map_classes(self.main_dataset, class_map)
            self.test_dataset = map_classes(self.test_dataset, class_map)

        self.batch_sizes = batch_sizes
        self.device = device
        self.loader_kwargs = loader_kwargs if loader_kwargs is not None else {}

    @property
    def n_train_labels(self) -> int:
        return len(self.main_inds["train"])

    def convert_datasets_to_numpy(self) -> None:
        self.main_dataset = self.main_dataset.numpy()
        self.test_dataset = self.test_dataset.numpy()

    def convert_datasets_to_torch(self) -> None:
        self.main_dataset = self.main_dataset.torch()
        self.test_dataset = self.test_dataset.torch()

    def get_loader(self, subset: str, shuffle: bool | None = None) -> DataLoader:
        if subset == "test":
            inputs = self.test_dataset.data[self.test_inds]
            labels = self.test_dataset.targets[self.test_inds]
        else:
            subset_inds = self.main_inds[subset]
            inputs = self.main_dataset.data[subset_inds]
            labels = self.main_dataset.targets[subset_inds]

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if self.batch_sizes[subset] == -1:
            batch_size = len(inputs)
        else:
            batch_size = self.batch_sizes[subset]

        if shuffle is None:
            shuffle = subset in {"train", "target"}

        # Use drop_last=True during training to avoid small batches that produce higher gradient
        # variance. Elsewhere use drop_last=False to ensure all the examples are used.
        loader = DataLoader(
            dataset=TensorDataset(inputs, labels),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=(subset == "train"),
            **self.loader_kwargs,
        )

        return loader

    def move_from_pool_to_train(self, pool_inds_to_move: int | Sequence[int]) -> None:
        """
        Important:
        - pool_inds_to_move and pool_inds_to_keep index into self.main_inds["pool"]
        - self.main_inds["pool"] and train_inds_to_add index into self.main_dataset
        """
        if isinstance(pool_inds_to_move, int):
            pool_inds_to_move = [pool_inds_to_move]

        pool_inds_to_keep = range(len(self.main_inds["pool"]))
        pool_inds_to_keep = np.setdiff1d(pool_inds_to_keep, pool_inds_to_move)

        train_inds_to_add = [self.main_inds["pool"][ind] for ind in pool_inds_to_move]

        self.main_inds["pool"] = [self.main_inds["pool"][ind] for ind in pool_inds_to_keep]
        self.main_inds["train"] += train_inds_to_add


def initialize_indices(
    labels: np.ndarray, label_counts: ConfigDict | ConfigList | int, rng: Generator
) -> np.ndarray:
    if isinstance(label_counts, int):
        if label_counts == -1:
            label_counts = len(labels)

        selected_inds = rng.choice(len(labels), size=label_counts, replace=False)

    else:
        if isinstance(label_counts, (list, ListConfig)):
            label_counts = preprocess_label_counts(label_counts, rng)

        selected_inds = []

        for _class, count in label_counts.items():
            class_inds = np.flatnonzero(labels == _class)

            if count == -1:
                count = len(class_inds)

            selected_inds += [rng.choice(class_inds, size=count, replace=False)]

        selected_inds = np.concatenate(selected_inds)
        selected_inds = rng.permutation(selected_inds)

    return selected_inds


def preprocess_label_counts(label_counts: ConfigList, rng: Generator) -> dict:
    processed_label_counts = {}

    for cfg in label_counts:
        classes = eval(cfg["classes"])
        n_classes = cfg["n_classes"]

        if (n_classes != -1) and (n_classes < len(classes)):
            classes = rng.choice(list(classes), size=n_classes, replace=False)

        for _class in classes:
            assert _class not in processed_label_counts
            processed_label_counts[_class] = cfg["n_per_class"]

    return processed_label_counts


def map_classes(dataset: Dataset, class_map: ConfigDict | ConfigList) -> Dataset:
    class_map = preprocess_class_map(class_map)

    dataset.original_targets = dataset.targets
    dataset.targets = class_map(dataset.targets)

    return dataset


def preprocess_class_map(class_map: ConfigDict | ConfigList) -> Callable:
    if isinstance(class_map, (list, ListConfig)):
        processed_class_map = {}

        for cfg in class_map:
            old_class = eval(cfg["old_class"])

            if isinstance(old_class, int):
                old_class = [old_class]

            for _class in old_class:
                assert _class not in processed_class_map
                processed_class_map[_class] = cfg["new_class"]

    else:
        processed_class_map = dict(class_map)

    return np.vectorize(processed_class_map.get)
