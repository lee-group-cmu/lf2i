import numpy as np
import pandas as pd


def np_to_pd(array, names):
    return pd.DataFrame({names[i]: array[:, i] for i in range(len(names))})
