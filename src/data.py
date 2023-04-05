import numpy as np
import torch
from numpy.random import Generator
from omegaconf import DictConfig
from src.datasets.base import BaseDataset
from torch.utils.data import DataLoader, Dataset, Subset
from typing import List, Sequence, Tuple, Union


class Data:
    """
    Specifying label_counts:
    - label_counts is a dictionary of dictionaries.
    - label_counts[subset] specifies how many examples we want from each class for the given subset.
    - The simplest way to specify label_counts[subset] is with integer keys.
        - Example: we want 5 examples from class 0 and 10 example from class 1.
        - Solution: d = {0: 5, 1: 10}.
    - If there are lots of classes, it becomes verbose to express the dictionary this way.
    - Instead we can use a special key to specify want we want for multiple classes at once.
        - Example: we want 5 examples from class 0, 10 examples from 2 classes in (1, 3, 4)
          and 15 examples from 3 classes in range(5, 10).
        - Solution: d = {0: 5, "2_classes_in_(1,3,4)": 10, "3_classes_in_range(5,10)": 15}.

    Specifying label_map:
    - label_map is a dictionary specifying how we want to redefine the classes in our dataset.
    - As for label_counts, we can use integer keys but also optionally one special key.
        - Example: we want to map 0 to 1, 1 to 2, and all others in range(10) to 0.
        - Solution: d = {0: 1, 1: 2, "rest_in_range(10)": 0}.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_sizes: Union[dict, DictConfig],
        label_counts: Union[dict, DictConfig],
        rng: Generator,
        label_map: Union[dict, DictConfig] = None,
        test_classes_to_remove: Sequence[int] = None,
        loader_kwargs: Union[dict, DictConfig] = None,
    ) -> None:
        self.main_dataset = dataset(train=True)
        self.test_dataset = dataset(train=False)

        self.inds = Data.initialize_subset_indices(self.main_dataset.targets, label_counts, rng)

        if label_map != None:
            Data.map_labels(self.main_dataset, label_map)
            Data.map_labels(self.test_dataset, label_map)

        if test_classes_to_remove != None:
            Data.remove_classes(self.test_dataset, test_classes_to_remove)

        self.batch_sizes = batch_sizes
        self.loader_kwargs = loader_kwargs if loader_kwargs != None else {}

    @property
    def input_shape(self) -> Union[Tuple[int], torch.Size]:
        return self.main_dataset.data.shape[1:]

    @property
    def n_classes(self) -> int:
        if isinstance(self.main_dataset.targets, np.ndarray):
            return len(np.unique(self.main_dataset.targets))
        else:
            return len(torch.unique(self.main_dataset.targets))

    @property
    def n_train_labels(self) -> int:
        return len(self.inds["train"])

    @staticmethod
    def initialize_subset_indices(
        labels: np.ndarray, label_counts: Union[dict, DictConfig], rng: Generator
    ) -> dict:
        """
        Create a dictionary, inds, where inds[subset] is a list of indices chosen such that the
        number of labels from each class is as specified in label_counts[subset].
        """
        inds = {}
        free_inds = np.arange(len(labels))

        for subset, subset_label_counts in label_counts.items():
            subset_label_counts = Data.preprocess_label_counts(subset_label_counts, rng)
            inds[subset] = Data.select_inds_to_match_label_counts(
                free_inds, labels[free_inds], subset_label_counts, rng
            )
            free_inds = np.setdiff1d(free_inds, inds[subset])

        return inds

    @staticmethod
    def preprocess_label_counts(label_counts: Union[dict, DictConfig], rng: Generator) -> dict:
        """
        As noted in the class docstring, label_counts can have special, non-integer keys. This
        function converts label_counts into an equivalent dictionary with only integer keys.
        """
        processed_label_counts = {}

        for class_or_class_set, count in label_counts.items():
            if isinstance(class_or_class_set, int):
                _class = class_or_class_set
                counts_to_add = {_class: count}

            elif isinstance(class_or_class_set, str):
                class_set = class_or_class_set
                n_classes, _, _, class_set = class_set.split("_")

                n_classes = int(n_classes)
                class_set = eval(class_set)

                if n_classes < len(class_set):
                    class_set = rng.choice(list(class_set), n_classes, replace=False)

                counts_to_add = {_class: count for _class in sorted(class_set)}

            else:
                raise ValueError

            assert np.all([k not in processed_label_counts.keys() for k in counts_to_add.keys()])

            processed_label_counts = {**processed_label_counts, **counts_to_add}

        return processed_label_counts

    @staticmethod
    def select_inds_to_match_label_counts(
        inds: np.ndarray,
        labels: np.ndarray,
        label_counts: Union[dict, DictConfig],
        rng: Generator,
    ) -> List[int]:
        """
        There's no need to return rng. Its internal state gets updated each time it is used.
        >>> f = lambda rng: rng.integers(0, 10)
        >>> rng = np.random.default_rng(0)
        >>> print(f(rng), f(rng))  # -> (8, 6)
        """
        shuffle_inds = rng.permutation(len(inds))
        inds = inds[shuffle_inds]
        labels = labels[shuffle_inds]

        selected_inds = []

        for label, count in label_counts.items():
            label_inds = inds[np.flatnonzero(labels == label)]
            assert len(label_inds) >= count
            selected_inds.append(label_inds[:count])

        selected_inds = np.concatenate(selected_inds)
        selected_inds = rng.permutation(selected_inds)

        return list(selected_inds)

    @staticmethod
    def map_labels(dataset: Dataset, label_map: Union[dict, DictConfig]) -> None:
        """
        Apply label_map to update dataset.targets. Also keep a copy of the old labels.
        """
        label_map = Data.preprocess_label_map(label_map)
        labels = dataset.targets

        # Copying here ensures the correct behavior if label_map is an empty dictionary.
        mapped_labels = np.copy(labels)

        for old_label, new_label in label_map.items():
            old_label_inds = np.flatnonzero(labels == old_label)
            mapped_labels[old_label_inds] = new_label

        dataset.original_targets = labels
        dataset.targets = mapped_labels

    @staticmethod
    def preprocess_label_map(label_map: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        """
        As noted in the class docstring, label_map can have a special, non-integer key. This
        function converts label_map into an equivalent dictionary with only integer keys.
        """
        if not isinstance(label_map, (dict, DictConfig)):
            return {}

        special_keys = [key for key in label_map.keys() if isinstance(key, str)]

        assert len(special_keys) <= 1, "Up to one special key allowed in a label map"

        if len(special_keys) == 1:
            special_key = special_keys[0]

            _, _, old_labels = special_key.split("_")
            old_labels = eval(old_labels)

            new_label_map = {old_label: label_map[special_key] for old_label in old_labels}

            for key, new_label in label_map.items():
                if key != special_key:
                    new_label_map[key] = new_label

            label_map = new_label_map

        return label_map

    @staticmethod
    def remove_classes(dataset: Dataset, classes_to_remove: Sequence[int]) -> None:
        inds_to_keep = np.flatnonzero([label not in classes_to_remove for label in dataset.targets])
        dataset.data = dataset.data[inds_to_keep]
        dataset.targets = dataset.targets[inds_to_keep]
        dataset.original_targets = dataset.original_targets[inds_to_keep]

    def numpy(self) -> None:
        self.main_dataset.numpy()
        self.test_dataset.numpy()

    def torch(self) -> None:
        self.main_dataset.torch()
        self.test_dataset.torch()

    def to(self, device: str) -> None:
        self.main_dataset.to(device)
        self.test_dataset.to(device)

    def get_loader(self, subset: str) -> DataLoader:
        if subset == "test":
            dataset = self.test_dataset
        else:
            subset_inds = self.inds[subset]
            dataset = Subset(self.main_dataset, subset_inds)

        batch_size = self.batch_sizes[subset]
        batch_size = len(dataset) if batch_size == -1 else batch_size

        shuffle = True if subset in {"train", "target"} else False

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **self.loader_kwargs)

    def move_from_pool_to_train(self, pool_inds_to_move: Union[int, list, np.ndarray]) -> None:
        """
        Important:
        - pool_inds_to_move and pool_inds_to_keep index into self.inds["pool"]
        - self.inds["pool"] and train_inds_to_add index into self.main_dataset
        """
        if not isinstance(pool_inds_to_move, (list, np.ndarray)):
            pool_inds_to_move = [pool_inds_to_move]

        pool_inds_to_keep = np.setdiff1d(range(len(self.inds["pool"])), pool_inds_to_move)
        train_inds_to_add = [self.inds["pool"][ind] for ind in pool_inds_to_move]

        self.inds["pool"] = [self.inds["pool"][ind] for ind in pool_inds_to_keep]
        self.inds["train"].extend(train_inds_to_add)
