import numpy as np


class FlexCodeCDFEstimator:
    """Wrapper around a FlexCodeModel that exposes a scikit-learn-style CDF interface.

    Both fit and predict accept a combined matrix X = np.hstack([z, x]) where
    column 0 holds the 1D response z and columns 1: hold the covariates x.
    """

    def __init__(self, model):
        """
        :param model: a FlexCodeModel instance (may be pre-fitted or unfitted).
        """
        self.model = model

    def fit(self, X, weight=None):
        """Fit the underlying FlexCodeModel.

        :param X: numpy array of shape (N, 1 + d); column 0 is z_train, columns 1: are x_train.
        :param weight: optional numpy array of sample weights passed to the model.
        :returns: self
        """
        z_train = X[:, 0:1]
        x_train = X[:, 1:]
        self.model.fit(x_train, z_train, weight)
        return self

    def predict(self, X, n_grid=1000):
        """Evaluate the conditional CDF at query points.

        :param X: numpy array of shape (N, 1 + d); column 0 is the z query values,
            columns 1: are covariates x. Construct via np.hstack([z, x_new]).
        :param n_grid: int, number of grid points used internally for the PDF.
        :returns: 1D numpy array of shape (N,) where result[i] = P(Z < z[i] | X = x[i]).
        :rtype: numpy array
        """
        z_query = X[:, 0]
        x_new = X[:, 1:]

        cdes, z_grid = self.model.predict(x_new, n_grid)
        z_flat = z_grid.flatten()
        dz = z_flat[1] - z_flat[0]

        cdfs = np.zeros_like(cdes)
        cdfs[:, 1:] = np.cumsum((cdes[:, :-1] + cdes[:, 1:]) / 2.0 * dz, axis=1)
        cdfs = np.clip(cdfs, 0.0, 1.0)

        return np.array([np.interp(z_query[i], z_flat, cdfs[i]) for i in range(len(z_query))])

    def predict_proba(self, X, n_grid=1000):
        """Return CDF values as a two-column array following the sklearn predict_proba convention.

        :param X: numpy array of shape (N, 1 + d); column 0 is the z query values,
            columns 1: are covariates x.
        :param n_grid: int, number of grid points used internally for the PDF.
        :returns: numpy array of shape (N, 2): col 0 = 1 - CDF, col 1 = CDF.
        """
        cdf_vals = self.predict(X, n_grid=n_grid)
        return np.column_stack([1.0 - cdf_vals, cdf_vals])
