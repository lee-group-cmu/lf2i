from typing import Optional, Union, Dict, Tuple

import numpy as np
from scipy import integrate
import torch
from torch.distributions import (
    Distribution, 
    MultivariateNormal, 
    Exponential, 
    Uniform, 
    Categorical, 
    Poisson
)
from sbi.utils import MultipleIndependent

from lf2i.simulator._base import Simulator


class OnOff(Simulator):
    
    def __init__(
        self,
        poi_space_bounds: Dict[str, float],
        poi_grid_size: int,
        nuisance_space_bounds: Dict[str, Dict[str, float]],
        data_sample_size: int, 
        gamma: float,
        poi_dim: int = 1,
        nuisance_dim: int = 2,
        all_poi: bool = False
    ) -> None:
        super().__init__(poi_dim=poi_dim, data_dim=2, data_sample_size=data_sample_size, nuisance_dim=nuisance_dim)

        self.gamma = gamma
        self.poi_grid = torch.linspace(start=poi_space_bounds['low'], end=poi_space_bounds['high'], steps=poi_grid_size)
        self.poi_space_bounds = poi_space_bounds
        self.nuisance_space_bounds = nuisance_space_bounds
        self.all_poi = all_poi

        self.poi_prior = Uniform(low=self.poi_space_bounds['low'], high=self.poi_space_bounds['high'])
        self.eps_prior = Uniform(low=self.nuisance_space_bounds['eps']['low'], high=self.nuisance_space_bounds['eps']['high'])
        self.b_prior = Uniform(low=self.nuisance_space_bounds['b']['low'], high=self.nuisance_space_bounds['b']['high'])

        self.likelihood = lambda s, eps, b: MultipleIndependent(
            dists=[Poisson(rate=self.gamma*b), Poisson(rate=b + eps*s)]
        )
    
    def likelihood_sample(
        self,
        n_samples: int,
        s: torch.Tensor, 
        eps: torch.Tensor, 
        b: torch.Tensor
    ) -> torch.Tensor:
        # MultipleIndipendent does not work if batch_size > 1, hence need to write this
        assert s.shape == eps.shape == b.shape
        return torch.dstack((
            torch.transpose(Poisson(rate=self.gamma*b).sample(sample_shape=(n_samples, )).reshape(n_samples, b.shape[0], 1), 0, 1),  # background
            torch.transpose(Poisson(rate=b + eps*s).sample(sample_shape=(n_samples, )).reshape(n_samples, s.shape[0], 1), 0, 1)  # signal
        )) 

    def compute_exact_BF(
        self,
        poi_null: torch.Tensor,
        data: torch.Tensor,
        eps_null: Optional[torch.Tensor] = None,
        b_null: Optional[torch.Tensor] = None
    ) -> float:
        assert data.shape == (self.data_sample_size, self.data_dim)
        if self.all_poi:
            numerator = torch.prod(self.likelihood(s=poi_null, eps=eps_null, b=b_null).log_prob(value=data).double().exp()).item()
        else:
            numerator, _ = integrate.nquad(
                func=lambda eps, b: torch.prod(self.likelihood(s=poi_null, eps=torch.Tensor([eps]), b=torch.Tensor([b])).log_prob(value=data).double().exp()).item(),
                ranges=[
                    list(self.nuisance_space_bounds['eps'].values()),
                    list(self.nuisance_space_bounds['b'].values()),
                ]
            )
        denominator, _ = integrate.nquad(
            func=lambda s, eps, b: torch.prod(self.likelihood(s=s, eps=torch.Tensor([eps]), b=torch.Tensor([b])).log_prob(value=data).double().exp()).item(),
            ranges=[
                list(self.poi_space_bounds.values()),
                list(self.nuisance_space_bounds['eps'].values()),
                list(self.nuisance_space_bounds['b'].values()),
            ]
        )
        return numerator / denominator

    def simulate_for_test_statistic(
        self, 
        B: int, 
        estimation_method: str
    ) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        if estimation_method == 'likelihood':
            # class
            Y = Categorical(probs=torch.Tensor([.5, .5])).sample(sample_shape=(B, )).long()
            # parameters
            poi = self.poi_prior.sample(sample_shape=(B, )).reshape(-1, 1)
            eps = self.eps_prior.sample(sample_shape=(B, )).reshape(-1, 1)
            b = self.b_prior.sample(sample_shape=(B, )).reshape(-1, 1)
            poi_marginal = self.poi_prior.sample(sample_shape=(Y[Y == 0].shape[0], )).reshape(-1, 1)
            eps_marginal = self.eps_prior.sample(sample_shape=(Y[Y == 0].shape[0], )).reshape(-1, 1)
            b_marginal = self.b_prior.sample(sample_shape=(Y[Y == 0].shape[0], )).reshape(-1, 1)
            # samples
            samples = self.likelihood_sample(n_samples=1, s=poi, eps=eps, b=b).reshape(B, self.data_dim)
            samples_marginal = self.likelihood_sample(n_samples=1, s=poi_marginal, eps=eps_marginal, b=b_marginal).reshape(poi_marginal.shape[0], self.data_dim)
            samples[Y == 0, :] = samples_marginal
            return Y, torch.hstack((poi, eps, b)), samples
        elif estimation_method in ['prediction', 'posterior']:
            # parameters
            poi = self.poi_prior.sample(sample_shape=(B, )).reshape(-1, 1)
            eps = self.eps_prior.sample(sample_shape=(B, )).reshape(-1, 1)
            b = self.b_prior.sample(sample_shape=(B, )).reshape(-1, 1)
            # samples
            samples = self.likelihood_sample(n_samples=1, s=poi, eps=eps, b=b).reshape(B, self.data_dim)
            return torch.hstack((poi, eps, b)), samples
        else:
            raise ValueError(f"Only one of ['likelihood', 'prediction', 'posterior'] is supported, got {estimation_method}")
        
    def simulate_for_critical_values(
        self, 
        B_prime: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        # parameters
        poi = self.poi_prior.sample(sample_shape=(B_prime, )).reshape(-1, 1)
        eps = self.eps_prior.sample(sample_shape=(B_prime, )).reshape(-1, 1)
        b = self.b_prior.sample(sample_shape=(B_prime, )).reshape(-1, 1)
        # samples
        samples = self.likelihood_sample(n_samples=self.data_sample_size, s=poi, eps=eps, b=b)
        assert samples.shape == (B_prime, self.data_sample_size, self.data_dim)
        return torch.hstack((poi, eps, b)), samples

    def simulate_for_diagnostics(self, B_doubleprime: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        return self.simulate_for_critical_values(B_doubleprime)


class Inferno3DMixture(Simulator):

    _poi_names = {'s'}
    _nuisance_names = {'b', 'r', 'bg_lambda'}
    _fixed_params_names = {'bg_mean', 'bg_cov', 's_mean', 's_cov', 's_lambda'}
    _default_fixed_params = {
        'bg_mean': torch.Tensor([[2, 0]]), 
        'bg_cov': torch.Tensor([[5, 0], [0, 9]]), 
        's_mean': torch.Tensor([[1, 1]]), 
        's_cov': torch.eye(2), 
        's_lambda': torch.Tensor([[2]])
    }

    def __init__(
        self,
        priors: Dict[str, Distribution],
        poi_bounds: Dict[str, float],
        poi_grid_size: int,
        nuisance_dim: int,
        data_sample_size: int,
        nuisance_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        fixed_params: Dict[str, Union[float, torch.Tensor]] = {}
    ) -> None:
        super().__init__(poi_dim=1, data_dim=3, data_sample_size=data_sample_size, nuisance_dim=nuisance_dim)

        # check every poi or (actual) nuisance has a prior and is not fixed; set fixed params
        assert len(priors.keys()) == (self.poi_dim+self.nuisance_dim), \
            "Need to specify a prior distribution for each parameter of interest and nuisance parameter"
        assert len(priors.keys() & fixed_params.keys()) == 0, \
            "Parameters of interest and nuisance parameters are sampled through priors and cannot have fixed values. Remove them from `fixed_params`"
        self.fixed_params = fixed_params
        self.fixed_params.update({name: self._default_fixed_params[name] for name in self._default_fixed_params.keys() - fixed_params.keys()})

        # check that all nuisances are either given a prior or are fixed
        assert self._nuisance_names & (priors.keys() | fixed_params.keys()) == self._nuisance_names, \
            "Nuisance parameters must either be given a prior or be fixed (in `fixed_params`)."

        self.poi_bounds = poi_bounds
        self.nuisance_bounds = nuisance_bounds
        self.poi_grid = torch.linspace(start=poi_bounds['low'], end=poi_bounds['high'], steps=poi_grid_size, dtype=torch.int)
        self.poi_grid_size = poi_grid_size

        self.poi_prior = priors.pop('s')
        self.nuisance_priors = priors if self.nuisance_dim > 0 else {}
        self.qr_poi_prior = Uniform(low=poi_bounds['low']*torch.ones(1), high=poi_bounds['high']*torch.ones(1))
        self.qr_nuisance_priors = {
            nuisance: Uniform(low=nuisance_bounds[nuisance]['low']*torch.ones(1), high=nuisance_bounds[nuisance]['high']*torch.ones(1)) 
            for nuisance in self.nuisance_priors
        }
    
    @classmethod
    def _base_distribution_log_prob(
        cls, 
        sample: torch.Tensor,
        loc: torch.Tensor,
        cov: torch.Tensor,
        lambda_: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        return torch.log(torch.prod(torch.hstack((
            torch.exp(MultivariateNormal(loc=loc, covariance_matrix=cov).log_prob(sample[:, :2]).reshape(-1, 1)).double(),
            torch.exp(Exponential(rate=lambda_).log_prob(sample[:, 2:]).reshape(-1, 1)).double()
        )), dim=1)).reshape(-1, 1)

    @classmethod
    def _base_distribution_sample(
        cls,
        sample_shape: Tuple,
        loc: torch.Tensor,
        cov: torch.Tensor,
        lambda_: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        return torch.hstack((
            MultivariateNormal(loc=loc, covariance_matrix=cov).sample(sample_shape=sample_shape).reshape(-1, 2),
            Exponential(rate=lambda_).sample(sample_shape=sample_shape).reshape(-1, 1)
        ))
    
    def _likelihood_log_prob(
        self,
        sample: torch.Tensor,
        s: torch.Tensor,
        b: torch.Tensor,
        r: torch.Tensor,
        bg_lambda: torch.Tensor
    ) -> torch.Tensor:
        sample = sample.reshape(-1, self.data_dim)
        s = s.reshape(-1, 1)
        b = b.reshape(-1, 1)
        assert s.shape == b.shape

        signal_prob = torch.exp(self._base_distribution_log_prob(
            sample=sample,
            loc=self._default_fixed_params['s_mean'],
            cov=self._default_fixed_params['s_cov'],
            lambda_=self._default_fixed_params['s_lambda']
        )).reshape(1, sample.shape[0])
        bg_prob = torch.exp(self._base_distribution_log_prob(
            sample=sample,
            loc=self._default_fixed_params['bg_mean'] + torch.Tensor([r, 0]),
            cov=self._default_fixed_params['bg_cov'],
            lambda_ = bg_lambda
        )).reshape(1, sample.shape[0])
        
        single_samples_probs = (b/(s+b))*bg_prob + (s/(s+b))*signal_prob
        return torch.log(torch.prod(single_samples_probs.double(), dim=1))

    def _likelihood_sample(
        self,
        n_samples: int,
        s: torch.Tensor,
        b: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        bg_lambda: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        s = s.reshape(-1, 1)  # dim 0 is n_batches
        n_batches = s.shape[0]
        # when nuisances are fixed
        if b is None:
            b = torch.tile(torch.Tensor([self.fixed_params['b']]), dims=(n_batches, 1))
        if r is None:
            r = torch.tile(torch.Tensor([self.fixed_params['r']]), dims=(n_batches, 1))
        if bg_lambda is None:
            bg_lambda = torch.tile(torch.Tensor([self.fixed_params['bg_lambda']]), dims=(n_batches, 1))
        assert b.shape == (n_batches, 1)
        assert r.shape == (n_batches, 1)
        assert bg_lambda.shape == (n_batches, 1)
        
        # dim 0 indexes batch, dim 1 indexes component for each of the n_samples
        which_component = Categorical(probs=torch.hstack((b/(s+b), s/(s+b)))).sample(sample_shape=(n_samples, )).T.long()
        which_r = torch.gather(
            input=torch.hstack((
                r,  # background
                torch.zeros(n_batches).reshape(n_batches, 1)  # signal (no r)
            )),
            dim=1,  # choose along the first dimension,
            index=which_component
        ).reshape(n_batches*n_samples, 1)
        which_lambda = torch.gather(
            input=torch.hstack((
                bg_lambda, 
                torch.tile(self._default_fixed_params['s_lambda'], dims=(n_batches, 1))
            )),
            dim=1,
            index=which_component
        ).reshape(n_batches*n_samples, 1)
        which_loc = torch.stack((  # same as gather, but need to choose whole 2D subvector
            torch.tile(self._default_fixed_params['bg_mean'], dims=(n_batches, 1)),
            torch.tile(self._default_fixed_params['s_mean'], dims=(n_batches, 1)),
        ), dim=1)[torch.repeat_interleave(torch.arange(n_batches), repeats=n_samples), which_component.reshape(n_batches*n_samples, ), :]
        which_loc = which_loc + which_r
        which_cov = torch.stack((
            torch.tile(self._default_fixed_params['bg_cov'], dims=(n_batches, 1, 1)),
            torch.tile(self._default_fixed_params['s_cov'], dims=(n_batches, 1, 1)),
        ), dim=1)[torch.repeat_interleave(torch.arange(n_batches), repeats=n_samples), which_component.reshape(n_batches*n_samples, ), :, :]

        return self._base_distribution_sample(sample_shape=(1, ), loc=which_loc, cov=which_cov, lambda_=which_lambda).reshape(n_batches, n_samples, self.data_dim)
    
    def simulate_for_test_statistic(
        self, 
        b: int, 
        estimation_method: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if estimation_method == 'likelihood':
            raise NotImplementedError
        elif estimation_method == 'likelihood_bap':  # best average power (Heinrich, 2022)
            # each batch has b//2 samples from a unique null and b//2 samples from equidistant alternatives
            poi_h0 = self.poi_prior.sample(sample_shape=(1, )).reshape(1, 1)
            nuisance_h0 = {
                nuisance: prior.sample(sample_shape=(1, )).reshape(1, 1)
                for nuisance, prior in self.nuisance_priors.items()
            }
            samples_h0 = self._likelihood_sample(n_samples=b//2, s=poi_h0, **nuisance_h0)
            labels_h0 = torch.ones(b//2)
            
            scale = self.poi_prior.sample(sample_shape=(1, )).item()  # use value in poi range to determine distance from null
            scale = min(scale, min(poi_h0.item()-self.poi_bounds['low'], self.poi_bounds['high']-poi_h0.item()))  # make sure alternatives remain in poi range
            poi_h1_left, poi_h1_right = poi_h0-scale, poi_h0+scale
            # same nuisance as in null for simplicity (as in Heinrich, 2022)
            samples_h1_left = self._likelihood_sample(n_samples=b//4, s=poi_h1_left, **nuisance_h0)
            samples_h1_right = self._likelihood_sample(n_samples=b//4, s=poi_h1_right, **nuisance_h0)
            labels_h1 = torch.zeros((b//4)*2)
            params_samples = torch.hstack((
                # use only poi null as input to estimate likelihood; nuisances are marginalized out by construction
                torch.tile(poi_h0, dims=((b//2)+(b//4)*2, 1)),
                torch.vstack((
                    samples_h0.reshape(b//2, self.data_dim), 
                    samples_h1_left.reshape(b//4, self.data_dim), 
                    samples_h1_right.reshape(b//4, self.data_dim)
                ))
            ))
            # permute to avoid processing all nulls and then all alternatives in the batch in order
            permuted_index = torch.from_numpy(np.random.choice(a=range(params_samples.shape[0]), size=params_samples.shape[0], replace=False))
            return params_samples[permuted_index, :], torch.concat((labels_h0, labels_h1))[permuted_index]
        elif estimation_method in ['prediction', 'posterior']:
            raise NotImplementedError
        else:
            raise ValueError(f"Only one of ['likelihood', 'likelihood_bap', 'prediction', 'posterior'] is supported, got {estimation_method}")

    def simulate_for_critical_values(
        self, 
        b_prime: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = self.qr_poi_prior.sample(sample_shape=(b_prime, )).reshape(b_prime, self.poi_dim)
        nuisance = {
            nuisance: prior.sample(sample_shape=(b_prime, 1)).reshape(b_prime, 1)
            for nuisance, prior in self.qr_nuisance_priors.items()
        }
        samples = self._likelihood_sample(n_samples=self.data_sample_size, s=poi, **nuisance)
        return poi, nuisance, samples

    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)
