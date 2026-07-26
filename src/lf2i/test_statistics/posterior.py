from typing import Union, Any, Dict
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import torch
from torch.distributions import Distribution

from lf2i.utils.parallel import tqdm_joblib
from lf2i.utils.posterior_ts_inputs import preprocess_estimation_evaluation
from lf2i.test_statistics import TestStatistic


class Posterior(TestStatistic):
    """Implements the `Posterior` test statistic, i.e. the (log) posterior density :math:`\\log p(\\theta \\mid x)`
    evaluated at a neural posterior estimator, as described in https://doi.org/10.1088/2632-2153/ae67cd.

    Unlike `Waldo`'s `posterior` mode (which reduces the posterior to a conditional mean and variance via
    Monte Carlo sampling), this test statistic uses the posterior density directly: large values indicate that
    :math:`\\theta` is well-supported by the estimated posterior given `x`, so the acceptance region is `right`.

    Parameters
    ----------
    poi_dim : int
        Dimensionality (number) of the parameters of interest.
    estimator : Union[str, Any]
        Neural posterior estimator. Currently compatible with posterior objects implementing the interface of
        `lf2i.estimators.base_posteriors.AbstractNeuralPosteriorTrainer` (e.g., estimators from the `sbi` library).
        If `str`, must be one of the predefined estimators listed in `test_statistics/_estimators.py`.
    estimator_kwargs : Dict, optional
        Hyperparameters and settings for the posterior estimator, by default {}.
    n_jobs : int, optional
        Number of workers to use when evaluating the test statistic over multiple inputs, by default -2, which
        uses all cores minus one. `n_jobs == -1` uses all cores. If `n_jobs < -1`, then `n_jobs = os.cpu_count()+1+n_jobs`.
    **posterior_kwargs : Any
        Additional keyword arguments forwarded to `estimator.log_prob(theta=..., x=..., **posterior_kwargs)` at
        evaluation time.
    """

    def __init__(
        self,
        poi_dim: int,
        estimator: Union[str, Any],
        estimator_kwargs: Dict = {},
        n_jobs: int = -2,
        **posterior_kwargs
    ) -> None:
        # Accept for high values, i.e. if posterior is very high
        super().__init__(acceptance_region='right', estimation_method='posterior')
        self.poi_dim = self.param_dim = poi_dim
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
        self.posterior_kwargs = posterior_kwargs
        self.n_jobs = n_jobs

    def estimate(
        self,
        parameters: torch.Tensor,
        samples: torch.Tensor,
    ) -> None:
        """Train the neural posterior estimator.

        Parameters
        ----------
        parameters : torch.Tensor
            Simulated parameters to be used for training.
        samples : torch.Tensor
            Simulated samples to be used for training.
        """
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)
        _ = self.estimator.append_simulations(parameters, samples).train()
        self.estimator = self.estimator.build_posterior()
        self._estimator_trained['posterior'] = True

    def evaluate(
        self,
        parameters: torch.Tensor,
        samples: torch.Tensor,
        mode: str
    ) -> np.ndarray:
        """Evaluate the `Posterior` test statistic, i.e. :math:`\\log p(\\theta \\mid x)`, over the given
        parameters and samples.

        Behaviour differs depending on mode:
            - 'critical_values' and 'diagnostics' evaluate the log-posterior once for each pair :math:`(\\theta, x)`.
            - 'confidence_sets' evaluates the log-posterior over all pairs given by the cartesian product of
              `parameters` (the parameter grid to construct confidence sets) and `samples`.

        Parameters
        ----------
        parameters : torch.Tensor
            Parameters over which to evaluate the test statistic.
        samples : torch.Tensor
            Samples over which to evaluate the test statistic.
        mode : str
            Either 'critical_values', 'confidence_sets', 'diagnostics'.

        Returns
        -------
        np.ndarray
            Log-posterior density evaluated over parameters and samples.

        Raises
        ------
        ValueError
            If `mode` is not among the pre-specified values.
        """
        assert self._check_is_trained(), "Estimator is not trained"
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)
                
        if mode in ['critical_values', 'diagnostics']:
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    log_posterior = self.estimator.log_prob(
                        theta=parameters[idx, :], x=samples[idx, ...], **self.posterior_kwargs
                    ).double()
                return log_posterior.numpy()
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating posterior for {samples.shape[0]} points ...", total=len(it))) as _:
                posterior_ts = np.array(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return posterior_ts.reshape(parameters.shape[0], )
        elif mode == 'confidence_sets':
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    log_posterior = self.estimator.log_prob(
                        theta=parameters, x=samples[idx, ...], **self.posterior_kwargs
                    ).double().reshape(1, parameters.shape[0])
                return log_posterior.numpy()
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating posterior for {samples.shape[0]} points ...", total=len(it))) as _:
                posterior_ts = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return posterior_ts.reshape(samples.shape[0], parameters.shape[0])
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")
        

class PosteriorPriorRatio(TestStatistic):
    """Implements the `PosteriorPriorRatio` test statistic, i.e. the log-ratio of the estimated posterior
    density to the prior density, :math:`\\log \\frac{p(\\theta \\mid x)}{p(\\theta)}`, evaluated at a neural
    posterior estimator.

    Large values indicate that the posterior places much more mass on :math:`\\theta` than the prior did, i.e.
    that the data :math:`x` is highly informative about :math:`\\theta` relative to the prior; equivalently,
    that the prior (denominator) is very low relative to the posterior (numerator). The acceptance region is
    `right`.

    Parameters
    ----------
    poi_dim : int
        Dimensionality (number) of the parameters of interest.
    prior : Union[torch.distributions.Distribution, Any]
        Prior distribution over the parameters of interest. Must implement `log_prob(theta)`.
    estimator : Union[str, Any]
        Neural posterior estimator. Currently compatible with posterior objects implementing the interface of
        `lf2i.estimators.base_posteriors.AbstractNeuralPosteriorTrainer` (e.g., estimators from the `sbi` library).
        If `str`, must be one of the predefined estimators listed in `test_statistics/_estimators.py`.
    estimator_kwargs : Dict, optional
        Hyperparameters and settings for the posterior estimator, by default {}.
    n_jobs : int, optional
        Number of workers to use when evaluating the test statistic over multiple inputs, by default -2, which
        uses all cores minus one. `n_jobs == -1` uses all cores. If `n_jobs < -1`, then `n_jobs = os.cpu_count()+1+n_jobs`.
    **posterior_kwargs : Any
        Additional keyword arguments forwarded to `estimator.log_prob(theta=..., x=..., **posterior_kwargs)` at
        evaluation time.
    """

    def __init__(
        self,
        poi_dim: int,
        prior: Union[Distribution, Any],
        estimator: Union[str, Any],
        estimator_kwargs: Dict = {},
        n_jobs: int = -2,
        **posterior_kwargs
    ) -> None:
        # Accept for high values, i.e. if posterior (numerator) is very high relative to the prior (denominator).
        # Equivalently, if prior (denominator) is very low relative to the posterior (numerator).
        super().__init__(acceptance_region='right', estimation_method='posterior')
        self.poi_dim = poi_dim
        self.prior = prior
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
        self.posterior_kwargs = posterior_kwargs
        self.n_jobs = n_jobs

    def estimate(
        self,
        parameters: torch.Tensor,
        samples: torch.Tensor,
    ) -> None:
        """Train the neural posterior estimator.

        Parameters
        ----------
        parameters : torch.Tensor
            Simulated parameters to be used for training.
        samples : torch.Tensor
            Simulated samples to be used for training.
        """
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)
        _ = self.estimator.append_simulations(parameters, samples).train()
        self.estimator = self.estimator.build_posterior()
        self._estimator_trained['posterior'] = True

    def evaluate(
        self,
        parameters: torch.Tensor,
        samples: torch.Tensor,
        mode: str
    ) -> np.ndarray:
        """Evaluate the `PosteriorPriorRatio` test statistic, i.e. :math:`\\log \\frac{p(\\theta \\mid x)}{p(\\theta)}`,
        over the given parameters and samples.

        Behaviour differs depending on mode:
            - 'critical_values' and 'diagnostics' evaluate the log-ratio once for each pair :math:`(\\theta, x)`.
            - 'confidence_sets' evaluates the log-ratio over all pairs given by the cartesian product of
              `parameters` (the parameter grid to construct confidence sets) and `samples`.

        Parameters
        ----------
        parameters : torch.Tensor
            Parameters over which to evaluate the test statistic.
        samples : torch.Tensor
            Samples over which to evaluate the test statistic.
        mode : str
            Either 'critical_values', 'confidence_sets', 'diagnostics'.

        Returns
        -------
        np.ndarray
            Log posterior-to-prior ratio evaluated over parameters and samples.

        Raises
        ------
        ValueError
            If `mode` is not among the pre-specified values.
        """
        assert self._check_is_trained(), "Estimator is not trained"
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)

        if mode in ['critical_values', 'diagnostics']:
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    ppr = torch.log(
                        torch.exp(self.estimator.log_prob(
                                theta=parameters[idx, :], x=samples[idx, :], **self.posterior_kwargs
                            ).double()).double() /
                            torch.exp(self.prior.log_prob(parameters[idx, :]).double()).double()
                    )
                return ppr.numpy()
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating PPR for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.array(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(parameters.shape[0], )
        elif mode == 'confidence_sets':
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    ppr = torch.log(
                        torch.exp(self.estimator.log_prob(
                                theta=parameters, x=samples[idx, :], **self.posterior_kwargs
                            ).double()).double().reshape(parameters.shape[0], ) / 
                            torch.exp(self.prior.log_prob(parameters).double()).double().reshape(parameters.shape[0], )
                    )
                return ppr.numpy().reshape(1, parameters.shape[0])
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating PPR for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(samples.shape[0], parameters.shape[0])
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")


class PriorPosteriorRatio(TestStatistic):
    """Implements the `PriorPosteriorRatio` test statistic, i.e. the log-ratio of the prior density to the
    estimated posterior density, :math:`\\log \\frac{p(\\theta)}{p(\\theta \\mid x)}`, evaluated at a neural
    posterior estimator.

    This is the negative of `PosteriorPriorRatio`'s statistic (:math:`\\log \\frac{p(\\theta)}{p(\\theta \\mid x)}
    = -\\log \\frac{p(\\theta \\mid x)}{p(\\theta)}`), provided for workflows that require an acceptance region on
    the `left` (small values accepted, i.e. when the posterior is high relative to the prior, or equivalently
    when the prior is low relative to the posterior) rather than on the `right`, e.g. for consistency with
    `Waldo`, which also uses `acceptance_region='left'`.

    Parameters
    ----------
    poi_dim : int
        Dimensionality (number) of the parameters of interest.
    prior : Union[torch.distributions.Distribution, Any]
        Prior distribution over the parameters of interest. Must implement `log_prob(theta)`.
    estimator : Union[str, Any]
        Neural posterior estimator. Currently compatible with posterior objects implementing the interface of
        `lf2i.estimators.base_posteriors.AbstractNeuralPosteriorTrainer` (e.g., estimators from the `sbi` library).
        If `str`, must be one of the predefined estimators listed in `test_statistics/_estimators.py`.
    estimator_kwargs : Dict, optional
        Hyperparameters and settings for the posterior estimator, by default {}.
    n_jobs : int, optional
        Number of workers to use when evaluating the test statistic over multiple inputs, by default -2, which
        uses all cores minus one. `n_jobs == -1` uses all cores. If `n_jobs < -1`, then `n_jobs = os.cpu_count()+1+n_jobs`.
    **posterior_kwargs : Any
        Additional keyword arguments forwarded to `estimator.log_prob(theta=..., x=..., **posterior_kwargs)` at
        evaluation time.
    """

    def __init__(
        self,
        poi_dim: int,
        prior: Union[Distribution, Any],
        estimator: Union[str, Any],
        estimator_kwargs: Dict = {},
        n_jobs: int = -2,
        **posterior_kwargs
    ) -> None:
        # Accept for low values, i.e. if posterior (denominator) is high relative to the prior (numerator).
        # Equivalently, if prior (numerator) is low relative to the posterior (denominator).
        super().__init__(acceptance_region='left', estimation_method='posterior')
        self.poi_dim = poi_dim
        self.prior = prior
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
        self.posterior_kwargs = posterior_kwargs
        self.n_jobs = n_jobs

    def estimate(
        self,
        parameters: torch.Tensor,
        samples: torch.Tensor,
    ) -> None:
        """Train the neural posterior estimator.

        Parameters
        ----------
        parameters : torch.Tensor
            Simulated parameters to be used for training.
        samples : torch.Tensor
            Simulated samples to be used for training.
        """
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)
        _ = self.estimator.append_simulations(parameters, samples).train()
        self.estimator = self.estimator.build_posterior()
        self._estimator_trained['posterior'] = True

    def evaluate(
        self,
        parameters: torch.Tensor,
        samples: torch.Tensor,
        mode: str
    ) -> np.ndarray:
        """Evaluate the `PriorPosteriorRatio` test statistic, i.e. :math:`\\log \\frac{p(\\theta)}{p(\\theta \\mid x)}`,
        over the given parameters and samples.

        Behaviour differs depending on mode:
            - 'critical_values' and 'diagnostics' evaluate the log-ratio once for each pair :math:`(\\theta, x)`.
            - 'confidence_sets' evaluates the log-ratio over all pairs given by the cartesian product of
              `parameters` (the parameter grid to construct confidence sets) and `samples`.

        Parameters
        ----------
        parameters : torch.Tensor
            Parameters over which to evaluate the test statistic.
        samples : torch.Tensor
            Samples over which to evaluate the test statistic.
        mode : str
            Either 'critical_values', 'confidence_sets', 'diagnostics'.

        Returns
        -------
        np.ndarray
            Log prior-to-posterior ratio evaluated over parameters and samples.

        Raises
        ------
        ValueError
            If `mode` is not among the pre-specified values.
        """
        assert self._check_is_trained(), "Estimator is not trained"
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)

        if mode in ['critical_values', 'diagnostics']:
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    ppr = torch.log(
                        torch.exp(self.prior.log_prob(parameters[idx, :]).double()).double() / 
                            torch.exp(self.estimator.log_prob(
                                theta=parameters[idx, :], x=samples[idx, :], **self.posterior_kwargs
                            ).double()).double()
                    )
                return ppr.numpy()
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating PPR for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.array(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(parameters.shape[0], )
        elif mode == 'confidence_sets':
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    ppr = torch.log(
                        torch.exp(self.prior.log_prob(parameters).double()).double().reshape(parameters.shape[0], ) / 
                            torch.exp(self.estimator.log_prob(
                                theta=parameters, x=samples[idx, :], **self.posterior_kwargs
                            ).double()).double().reshape(parameters.shape[0], )
                    )
                return ppr.numpy().reshape(1, parameters.shape[0])
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating PPR for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(samples.shape[0], parameters.shape[0])
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")
