import numpy as np
from scipy.stats import binomtest

def tomographic_binning(
    parameter_grid: np.ndarray,
    response_variable: np.ndarray,
    poi_idx: int=0,
    n_bins: int=5,
    min_cell_size: int=30,
    nominal_confidence: float=0.9,
    return_if_exact: bool=False
):
    """Bin `response_variable` according to bins of `parameter_grid` (1D). This
    util is for displaying conditional size and coverage of confidence sets as
    a function of 1 parameter at a time without estimating size or coverage as
    a function of the parameter (which introduces estimation error).

    Args:
    - parameter_grid : shape (B_DOUBLE_PRIME, param_dim) or (B_DOUBLE_PRIME, ) if param_dim == 1
    - response_variable : shape (B_DOUBLE_PRIME, )
    - poi_idx : integer between 0 and param_dim-1, indicating which column of
      `parameter_grid` to bin on. Ignored if `parameter_grid` is 1D.
    - n_bins : number of bins in poi over which to aggregate
    - min_cell_size : only relevant if parameter_grid has more than 1 column.
      Within each POI bin, the remaining (nuisance) columns are cut into a
      regular quantile grid, as fine as possible while keeping the average
      number of points per cell >= min_cell_size. Since the POI bins need
      not have equal counts, the resulting grid can differ across POI bins.

    Returns (groups, labels), where `groups` is a list of arrays holding the
    `response_variable` values falling in each bin (or, if `parameter_grid`
    has more than 1 column, the average `response_variable` value within
    each cell of a nuisance-parameter grid nested inside that bin), and
    `labels` are the corresponding parameter-range strings, suitable for
    `coverage_boxplot`/`set_size_boxplot` from the lf2i.plot module.
    """
    parameter_grid = np.asarray(parameter_grid)
    if parameter_grid.ndim == 1:
        parameter_grid = parameter_grid[:, None]
    response_variable = np.asarray(response_variable).reshape(-1)
    assert parameter_grid.shape[0] == response_variable.shape[0]

    param_dim = parameter_grid.shape[1]
    np_cols = [j for j in range(param_dim) if j != poi_idx]

    # Bin the POI column
    poi_values = parameter_grid[:, poi_idx]
    edges = np.linspace(poi_values.min(), poi_values.max(), n_bins + 1)
    edges[0] -= 1e-8
    bin_idx = np.digitize(poi_values, edges[1:-1], right=True)

    # For each POI bin, either list the raw response values in that bin
    # (param_dim == 1) or the average response value within each cell of a
    # nuisance-parameter grid nested inside that bin (param_dim > 1).
    groups, labels, test_results = [], [], []
    for b in range(n_bins):
        poi_mask = bin_idx == b
        n_b = poi_mask.sum()
        if n_b == 0:
            continue
        label = f"[{edges[b]:.2f}, {edges[b + 1]:.2f}]"

        if not np_cols:
            groups.append(response_variable[poi_mask])
            labels.append(label)
            if return_if_exact:
                test_results.append(np.asarray([
                    exact_coverage_test(response_variable[poi_mask], nominal_confidence)
                ]))
            continue

        # Regular (equal-count-per-dimension) NP grid, computed from the
        # points in this POI bin only, as fine as possible while keeping
        # >= min_cell_size points per cell on average.
        n_np_bins = max(1, int((n_b / min_cell_size) ** (1 / len(np_cols))))
        np_bin_idx = np.zeros(n_b, dtype=int)
        for j in np_cols:
            np_values = parameter_grid[poi_mask, j]
            np_edges = np.quantile(np_values, np.linspace(0, 1, n_np_bins + 1))
            np_edges[0] -= 1e-8
            np_bin_idx = np_bin_idx * n_np_bins + np.digitize(np_values, np_edges[1:-1], right=True)

        resp_in_bin = response_variable[poi_mask]
        cell_means = [resp_in_bin[np_bin_idx == cell].mean() for cell in np.unique(np_bin_idx)]

        if return_if_exact:
            test_result = [
                exact_coverage_test(resp_in_bin[np_bin_idx == cell], nominal_confidence) for cell in np.unique(np_bin_idx)
            ]
        else:
            test_result = []

        groups.append(np.asarray(cell_means))
        labels.append(label)
        test_results.append(np.asarray(test_result))

    if return_if_exact:
        return groups, labels, test_results
    else:
        return groups, labels


def exact_coverage_test(
    coverage_indicators: np.ndarray,
    nominal_confidence: float,
):
    r"""
    Return an over-, under-, or exact coverage flag as defined by

    0: H_0: p = 1-\alpha accepted in favor of H_1: p \ne 1-\alpha at level 0.1
    1: H_0: p = 1-\alpha rejected in favor of H_1: p \le 1-\alpha at level 0.05
    2: H_0: p = 1-\alpha rejected in favor of H_1: p \ge 1-\alpha at level 0.05

    where the test statistic is

    T = sum of c_i ~ Bin(N*(1-alpha), N*alpha*(1-alpha))

    under the null, where the c_i are the coverage indicators, N is the length
    of that array, and `1-alpha = nominal_confidence`.
    """
    coverage_indicators = np.asarray(coverage_indicators).reshape(-1)
    n = coverage_indicators.shape[0]
    successes = int(coverage_indicators.sum())

    two_sided = binomtest(successes, n, nominal_confidence, alternative='two-sided')
    if two_sided.pvalue > 0.1:
        return 0

    under = binomtest(successes, n, nominal_confidence, alternative='less')
    if under.pvalue <= 0.05:
        return 1

    over = binomtest(successes, n, nominal_confidence, alternative='greater')
    if over.pvalue <= 0.05:
        return 2

    return 0