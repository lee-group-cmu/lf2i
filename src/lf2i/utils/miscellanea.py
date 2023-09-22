from typing import Sequence

import numpy as np
import pandas as pd


def np_to_pd(array: np.ndarray, names: Sequence[str]):
    return pd.DataFrame({names[i]: array[:, i] for i in range(len(names))})
