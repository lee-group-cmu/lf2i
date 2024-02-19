from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch


def np_to_pd(array: np.ndarray, names: Sequence[str]):
    return pd.DataFrame({names[i]: array[:, i] for i in range(len(names))})


def to_np_if_torch(inp: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    return inp if isinstance(inp, np.ndarray) else inp.cpu().detach().numpy()

def to_torch_if_np(inp: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    return inp if isinstance(inp, torch.Tensor) else torch.from_numpy(inp)

def check_for_nans(inp: Union[np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]:
    if isinstance(inp, np.ndarray):
        if np.isnan(inp).sum() > 0:
            raise ValueError("Input contains NaN values")
    elif isinstance(inp, (pd.Series, pd.DataFrame)):
        if inp.isnull().values.any():
            raise ValueError("Input contains NaN values")
    else:
        if torch.isnan(inp).sum() > 0:
            raise ValueError("Input contains NaN values")
