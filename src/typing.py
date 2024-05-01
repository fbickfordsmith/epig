from typing import List, Sequence, Union

import numpy as np
from omegaconf import DictConfig, ListConfig
from torch import Size, Tensor


Array = Union[np.ndarray, Tensor]
ConfigDict = Union[dict, DictConfig]
ConfigList = Union[List[ConfigDict], ListConfig]
IndexSequence = Union[range, Sequence[int]]
Shape = Union[Sequence[int], Size]
