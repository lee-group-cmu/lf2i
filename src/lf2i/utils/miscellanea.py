from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch


def np_to_pd(array: np.ndarray, names: Sequence[str]):
    return pd.DataFrame({names[i]: array[:, i] for i in range(len(names))})


def to_np_if_torch(inp: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    return inp if isinstance(inp, np.ndarray) else inp.cpu().detach().numpy()
