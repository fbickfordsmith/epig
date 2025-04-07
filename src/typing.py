from typing import Dict, List, Sequence

import numpy as np
from omegaconf import DictConfig, ListConfig
from torch import Size, Tensor
from torch.nn import Parameter


Array = np.ndarray | Tensor
ConfigDict = dict | DictConfig
ConfigList = List[ConfigDict] | ListConfig
IndexSequence = range | Sequence[int]
ParamDict = Dict[str, Parameter]
Shape = Sequence[int] | Size
