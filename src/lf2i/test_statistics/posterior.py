from typing import Union, Any, Dict
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import torch
from torch.distributions import Distribution
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.simulators.simutils import tqdm_joblib
from lf2i.utils.posterior_ts_inputs import preprocess_estimation_evaluation
from lf2i.test_statistics import TestStatistic


class Posterior(TestStatistic):

    def __init__(
        self,
        poi_dim: int,
        estimator: Union[str, NeuralPosterior, Any],
        estimator_kwargs: Dict = {},
        n_jobs: int = -2
    ) -> None:
        # Accept for high values, i.e. if posterior is very high
        super().__init__(acceptance_region='right', estimation_method='posterior')
        self.poi_dim = poi_dim
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
        self.n_jobs = n_jobs

    def estimate(
        self,
        parameters: torch.Tensor, 
        samples: torch.Tensor, 
    ) -> None:
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
        assert self._check_is_trained(), "Estimator is not trained"
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)
                
        if mode in ['critical_values', 'diagnostics']:
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    log_posterior = self.estimator.log_prob(theta=parameters[idx, :], x=samples[idx, :]).double()
                return log_posterior.numpy()
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating posterior for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.array(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(parameters.shape[0], )
        elif mode == 'confidence_sets':
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    log_posterior = self.estimator.log_prob(theta=parameters, x=samples[idx, :]).double().reshape(1, parameters.shape[0])
                return log_posterior.numpy()
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating posterior for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(samples.shape[0], parameters.shape[0])
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")
        

class PosteriorPriorRatio(TestStatistic):

    def __init__(
        self,
        poi_dim: int,
        prior: Union[Distribution, Any],
        estimator: Union[str, NeuralPosterior, Any],
        estimator_kwargs: Dict = {},
        n_jobs: int = -2
    ) -> None:
        # Accept for high values, i.e. if posterior (numerator) is very high relative to the prior (denominator).
        # Equivalently, if prior (denominator) is very low relative to the posterior (numerator).
        super().__init__(acceptance_region='right', estimation_method='posterior')
        self.poi_dim = poi_dim
        self.prior = prior
        self.estimator = self._choose_estimator(estimator, estimator_kwargs, 'posterior')
        self.n_jobs = n_jobs

    def estimate(
        self,
        parameters: torch.Tensor, 
        samples: torch.Tensor, 
    ) -> None:
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
        assert self._check_is_trained(), "Estimator is not trained"
        parameters, samples = preprocess_estimation_evaluation(parameters, samples, self.poi_dim)
                
        if mode in ['critical_values', 'diagnostics']:
            def eval_one(idx):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)  # from nflows: torch.triangular_solve is deprecated in favor of ...
                    ppr = torch.log(
                        torch.exp(self.estimator.log_prob(theta=parameters[idx, :], x=samples[idx, :]).double()).double() / 
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
                        torch.exp(self.estimator.log_prob(theta=parameters, x=samples[idx, :]).double()).double().reshape(parameters.shape[0], ) / 
                            torch.exp(self.prior.log_prob(parameters).double()).double().reshape(parameters.shape[0], )
                    )
                return ppr.numpy().reshape(1, parameters.shape[0])
            with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Evaluating PPR for {samples.shape[0]} points ...", total=len(it))) as _:
                ppr = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(eval_one)(idx) for idx in it))
            return ppr.reshape(samples.shape[0], parameters.shape[0])
        else:
            raise ValueError(f"Only `critical_values`, `confidence_sets`, and `diagnostics` are supported, got {mode}")
