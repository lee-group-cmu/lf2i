from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def hist_multiD_data(
    data: np.ndarray,
    save_fig_path: Optional[str] = None,
    figsize: Tuple = (15, 15),
) -> None:
    rows = cols = data.shape[1]  # data dim
    _, ax = plt.subplots(rows, cols, figsize=figsize)
    
    for row in range(rows):
        for col in range(cols):
            # plots
            if col > row:
                ax[row, col].axis('off')
            elif col == row:
                ax[row, col].hist(x=data[:, row], bins=100, density=True)
            else:
                ax[row, col].hist2d(x=data[:, col], y=data[:, row], density=False, cmin=1, bins=100)

            # labels
            if row == rows-1:
                ax[row, col].set_xlabel(r'$x_{}$'.format(col), fontsize=25)
                ax[row, col].tick_params(axis='x', labelsize=12)
            else:
                ax[row, col].tick_params(axis='x', labelbottom=False)
            if col == 0:
                ax[row, col].set_ylabel(r'$x_{}$'.format(row), fontsize=25, rotation=0, labelpad=20)
                ax[row, col].tick_params(axis='y', labelsize=12)
            else:
                if row != col:  # axis is different for 1D hist
                    ax[row, col].tick_params(axis='y', labelleft=False)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
