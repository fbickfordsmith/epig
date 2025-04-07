from pathlib import Path
from typing import Any

from src.data.datasets.base import BaseEmbeddingDataset


class EmbeddingImageNet(BaseEmbeddingDataset):
    def __init__(self, data_dir: Path | str, **kwargs: Any) -> None:
        data_dir = Path(data_dir) / "imagenet"
        super().__init__(data_dir=data_dir, **kwargs)
