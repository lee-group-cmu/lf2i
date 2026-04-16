from typing import Union, Any, Dict, Optional, List, Tuple

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from lf2i.test_statistics.acore import ACORE
from lf2i.utils.odds_inputs import preprocess_odds_maximization


class FTS(ACORE):
    """Implements the Focus Test Statistic (FTS), which extends ACORE with a focus function
    that uses Laplace approximation to marginalize over nuisance parameters.

    Parameters
    ----------
    focus_function : Any
        Callable that takes a POI value and returns a scalar weight. Used in the Laplace
        approximation when marginalizing over nuisance parameters.
    All other parameters are inherited from ACORE.
    """

    def __init__(
        self,
        estimator: Union[str, Any],
        poi_dim: int,
        nuisance_dim: int,
        batch_size: int,
        data_dim: int,
        focus_function: Any,
        estimator_kwargs: Dict = {},
        verbose: bool = True,
        n_jobs: int = -2,
        optimizer: str = 'coordinate_descent',
        param_space_bounds: List[Tuple[float]] = None,
        max_iter: Optional[int] = 1,
        estimator_train_kwargs: Optional[Dict] = None
    ) -> None:
        super().__init__(
            estimator=estimator,
            poi_dim=poi_dim,
            nuisance_dim=nuisance_dim,
            batch_size=batch_size,
            data_dim=data_dim,
            estimator_kwargs=estimator_kwargs,
            verbose=verbose,
            n_jobs=n_jobs,
            optimizer=optimizer,
            param_space_bounds=param_space_bounds,
            max_iter=max_iter,
            estimator_train_kwargs=estimator_train_kwargs,
        )
        self.focus_function = focus_function

    def _denominator_method_selector(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        fixed_poi: Union[np.ndarray, torch.Tensor],
        optimization_bounds: List[Tuple[float]],
        condition_on_poi: bool = False
    ) -> float:
        if self.nuisance_dim > 0:
            return self._marginalize_log_odds(sample, fixed_poi, optimization_bounds, condition_on_poi=condition_on_poi)
        else:
            return self._maximize_log_odds(sample, fixed_poi, optimization_bounds, condition_on_poi=condition_on_poi)

    def _marg_log_lik(
        self,
        parameter: Union[np.ndarray, torch.Tensor],
        sample: Union[np.ndarray, torch.Tensor],
        epsilon: float = 1e-4
    ) -> float:
        """
        Evaluate the marginal log-likelihood (up to a normalization constant) for a given parameter and sample,
        using the trained estimator for odds and Laplace approximation to marginalize over nuisances.
        """
        parameter_minus = parameter.clone()
        parameter_minus[0] = parameter[0] - epsilon
        parameter_plus = parameter.clone()
        parameter_plus[0] = parameter[0] + epsilon

        log_lik = self._log_odds(self.estimator.predict_proba(
            X=preprocess_odds_maximization(self.estimator, parameter, parameter[0], 0, sample, self.param_space_bounds)
        ))[0, 1].item()
        log_lik_minus = self._log_odds(self.estimator.predict_proba(
            X=preprocess_odds_maximization(self.estimator, parameter_minus, parameter_minus[0], 0, sample, self.param_space_bounds)
        ))[0, 1].item()
        log_lik_plus = self._log_odds(self.estimator.predict_proba(
            X=preprocess_odds_maximization(self.estimator, parameter_plus, parameter_plus[0], 0, sample, self.param_space_bounds)
        ))[0, 1].item()
        second_derivative = np.abs(
            np.clip(
                (log_lik_plus - 2 * log_lik + log_lik_minus) / (epsilon ** 2),
                a_min=1e-10,
                a_max=None
            )
        )

        return -1 * (
            log_lik +  # log-likelihood
            np.log(self.focus_function(parameter[0])) +  # log focus
            0.5 * np.log(2 * np.pi) - 0.5 * np.log(second_derivative)  # Laplace approximation correction term
        )

    def _marginalize_log_odds(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        fixed_poi: Union[np.ndarray, torch.Tensor],
        optimization_bounds: List[Tuple[float]],
        argmax: bool = False,
        epsilon: float = 1e-4,
        condition_on_poi: bool = False
    ) -> float:
        """
        Algorithm
        - Compute the max a posteriori estimator via log p(x; mu, nu) + log f(mu)
        - Approximate the determinant of the Hessian of the negative log-posterior at the MAP estimator on diagonal terms via finite differences
        - Use Laplace approximation to compute the marginal likelihood,
            log p(x; mu) = log p(x; mu, nu_hat) + log f(mu) + (1/2) * log(2 * pi) - (1/2) * log(det(H_(mu, mu)(nu_hat))))
         where nu_hat is the MAP estimator of the nuisance parameters, d is the number of nuisance parameters, and H is the Hessian of the negative log-posterior at the MAP estimator.
         See https://en.wikipedia.org/wiki/Laplace%27s_method_(statistics) for more details on Laplace approximation.
        """
        assert fixed_poi.shape[0] in [0, self.poi_dim], f"fixed_poi should be either empty or have the same number of dimensions as the number of POIs, got {fixed_poi.shape[0]} and {self.poi_dim} respectively"

        if self.optimizer == 'nelder_mead':
            if fixed_poi.shape[0] > 0:
                opt_bounds = optimization_bounds[self.poi_dim:]
                x0 = np.array([np.mean(b) for b in opt_bounds])
                def objective(nu):
                    return self._marg_log_lik(torch.cat([fixed_poi, torch.tensor(nu)]), sample, epsilon=epsilon)
            else:
                opt_bounds = optimization_bounds
                x0 = np.array([np.mean(b) for b in opt_bounds])
                def objective(theta):
                    return self._marg_log_lik(torch.tensor(theta), sample, epsilon=epsilon)
            result = minimize(objective, x0=x0, method='Nelder-Mead', bounds=opt_bounds)
            if not result.success:
                result = minimize(objective, x0=x0, method='Nelder-Mead', bounds=opt_bounds,
                                  options={'maxiter': len(opt_bounds) * 400})
            if argmax:
                return torch.cat([fixed_poi, torch.tensor(result.x)]) if fixed_poi.shape[0] > 0 else torch.tensor(result.x)
            return result.fun  # _marg_log_lik already returns the negative value

        # coordinate descent
        if fixed_poi.shape[0] > 0:
            opt_dims = range(self.poi_dim, self.param_dim)
            nominal_parameter = torch.cat((fixed_poi, torch.tensor(
                np.array([np.mean(bounds) for bounds in optimization_bounds[self.poi_dim:]])
            )))  # use mid-point as initial guess for nuisances
        else:
            opt_dims = range(self.param_dim)
            nominal_parameter = torch.tensor(
                np.array([np.mean(bounds) for bounds in optimization_bounds])
            )  # use mid-point as initial guess

        for iteration in range(self.max_iter):
            current_nominal_parameter = nominal_parameter.clone()

            for pdx in opt_dims:
                if pdx <= self.poi_dim:
                    def objective(theta_j: float) -> float:
                        theta_j_minus = theta_j - epsilon
                        theta_j_plus = theta_j + epsilon

                        log_lik = self._log_odds(self.estimator.predict_proba(
                            X=preprocess_odds_maximization(self.estimator, current_nominal_parameter, theta_j, pdx, sample, self.param_space_bounds)
                        ))[0, 1].item()
                        log_lik_minus = self._log_odds(self.estimator.predict_proba(
                            X=preprocess_odds_maximization(self.estimator, current_nominal_parameter, theta_j_minus, pdx, sample, self.param_space_bounds)
                        ))[0, 1].item()
                        log_lik_plus = self._log_odds(self.estimator.predict_proba(
                            X=preprocess_odds_maximization(self.estimator, current_nominal_parameter, theta_j_plus, pdx, sample, self.param_space_bounds)
                        ))[0, 1].item()
                        second_derivative = np.abs(
                            np.clip(
                                (log_lik_plus - 2 * log_lik + log_lik_minus) / (epsilon ** 2),
                                a_min=1e-10,
                                a_max=None
                            )
                        )

                        return -1 * (
                            log_lik +  # log-likelihood
                            np.log(self.focus_function(theta_j)) +  # log focus
                            0.5 * np.log(2 * np.pi) - 0.5 * np.log(second_derivative)  # Laplace approximation correction term
                        )

                else:
                    def objective(theta_j: float) -> float:
                        current_nominal_parameter_minus = current_nominal_parameter.clone()
                        current_nominal_parameter_minus[0] = current_nominal_parameter[0] - epsilon
                        current_nominal_parameter_plus = current_nominal_parameter.clone()
                        current_nominal_parameter_plus[0] = current_nominal_parameter[0] + epsilon

                        log_lik = self._log_odds(self.estimator.predict_proba(
                            X=preprocess_odds_maximization(self.estimator, current_nominal_parameter, theta_j, pdx, sample, self.param_space_bounds)
                        ))[0, 1].item()
                        log_lik_minus = self._log_odds(self.estimator.predict_proba(
                            X=preprocess_odds_maximization(self.estimator, current_nominal_parameter_minus, theta_j, pdx, sample, self.param_space_bounds)
                        ))[0, 1].item()
                        log_lik_plus = self._log_odds(self.estimator.predict_proba(
                            X=preprocess_odds_maximization(self.estimator, current_nominal_parameter_plus, theta_j, pdx, sample, self.param_space_bounds)
                        ))[0, 1].item()
                        second_derivative = np.abs(
                            np.clip(
                                (log_lik_plus - 2 * log_lik + log_lik_minus) / (epsilon ** 2),
                                a_min=1e-10,
                                a_max=None
                            )
                        )

                        return -1 * (
                            log_lik +  # log-likelihood
                            np.log(self.focus_function(current_nominal_parameter[0])) +  # log focus
                            0.5 * np.log(2 * np.pi) - 0.5 * np.log(second_derivative)  # Laplace approximation correction term
                        )

                result = minimize_scalar(
                    objective,
                    bounds=optimization_bounds[pdx],
                    method='bounded'
                )
                current_nominal_parameter[pdx] = result.x

            nominal_parameter = current_nominal_parameter.clone()

        if argmax:
            return nominal_parameter
        else:
            return self._marg_log_lik(nominal_parameter, sample, epsilon=epsilon)
