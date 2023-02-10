from typing import Optional, Union, Dict, Tuple

import math
import numpy as np
import torch
from torch.distributions import (
    Distribution, 
    MultivariateNormal, 
    Exponential, 
    Uniform, 
    Categorical
)
from sklearn.mixture import GaussianMixture
from sbi.utils import MultipleIndependent
from pyro import distributions as pdist

from lf2i.simulator._base import Simulator


class GaussianMixture2D(Simulator):

    def __init__(
        self,
        likelihood_cov: torch.Tensor,
        mixture_p: float,
        priors: Dict[str, Union[str, Distribution]],
        max_radius: float,
        poi_bounds: Dict[str, float],
        poi_grid_size: int,
        data_sample_size: int
    ) -> None:
        super().__init__(param_dim=1, data_dim=2, data_sample_size=data_sample_size)
        self.nuisance_dim = 1

        self.max_radius = max_radius
        self.likelihood_cov = likelihood_cov
        self.mixture_p = mixture_p

        self.poi_bounds = poi_bounds
        self.poi_grid = torch.linspace(start=poi_bounds['low'], end=poi_bounds['high'], steps=poi_grid_size)
        self.poi_grid_size = poi_grid_size

        self.radius_qr_sample = lambda sample_shape: max_radius*Uniform(low=torch.Tensor([0]), high=torch.Tensor([1])).sample(sample_shape=sample_shape).sqrt()
        self.angle_qr_sample = lambda sample_shape: Uniform(low=torch.Tensor([0]), high=torch.Tensor([2*math.pi])).sample(sample_shape=sample_shape)

        if priors['radius'] == 'uniform':  # uniform in the circle of radius `max_radius`
            self.radius_sample = lambda sample_shape: max_radius*Uniform(low=torch.Tensor([0]), high=torch.Tensor([1])).sample(sample_shape=sample_shape).sqrt()
        else:
            self.radius_sample = lambda sample_shape: priors['radius'].sample(sample_shape=sample_shape)
        self.angle_sample = lambda sample_shape: priors['angle'].sample(sample_shape=sample_shape)
    
    def likelihood_sample(
        self,
        n_samples: int,
        mu: torch.Tensor, 
        nu: torch.Tensor
    ) -> torch.Tensor:
        n_batches = mu.shape[0]
        which_component = Categorical(probs=torch.Tensor([[self.mixture_p, 1-self.mixture_p]])).sample(sample_shape=(n_batches*n_samples, )).reshape(n_batches*n_samples, ).long()
        which_loc = torch.stack((
            torch.hstack((mu, nu)), torch.hstack((-mu, -nu))
        ), dim=1)[torch.repeat_interleave(torch.arange(n_batches), repeats=n_samples), which_component, :]
        return MultivariateNormal(loc=which_loc, covariance_matrix=self.likelihood_cov).sample(sample_shape=(1, )).reshape(n_batches, n_samples, self.data_dim)

    def simulate_for_test_statistic(
        self, 
        b: int,
        estimation_method: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if estimation_method == 'likelihood':
            raise NotImplementedError
        elif estimation_method == 'likelihood_bap': # best average power (Heinrich, 2022)
            # sample nulls
            radius_poi, angle_poi = self.radius_sample(sample_shape=(1, )), self.angle_sample(sample_shape=(1, ))
            radius_nuisance, angle_nuisance = self.radius_sample(sample_shape=(1, )), self.angle_sample(sample_shape=(1, ))  # independent of poi
            poi_h0, nuisance_h0 = radius_poi*torch.cos(angle_poi), radius_nuisance*torch.sin(angle_nuisance)  # polar to cartesian
            samples_h0 = self.likelihood_sample(n_samples=b//2, mu=poi_h0.reshape(1, 1), nu=nuisance_h0.reshape(1, 1))
            labels_h0 = torch.ones(b//2)

            # sample alternatives
            radius_scale, angle_scale = self.radius_sample(sample_shape=(1, )), self.angle_sample(sample_shape=(1, ))
            scale = radius_scale*torch.cos(angle_scale)
            poi_h1_left, poi_h1_right = poi_h0-scale, poi_h0+scale
            samples_h1_left = self.likelihood_sample(n_samples=b//4, mu=poi_h1_left.reshape(1, 1), nu=nuisance_h0.reshape(1, 1))
            samples_h1_right = self.likelihood_sample(n_samples=b//4, mu=poi_h1_right.reshape(1, 1), nu=nuisance_h0.reshape(1, 1))
            labels_h1 = torch.zeros((b//4)*2)

            params_samples = torch.hstack((
                torch.tile(poi_h0, dims=((b//2)+(b//4)*2, 1)),
                torch.vstack((
                    samples_h0.reshape(b//2, self.data_dim), 
                    samples_h1_left.reshape(b//4, self.data_dim), 
                    samples_h1_right.reshape(b//4, self.data_dim)
                ))
            ))

            permuted_index = torch.from_numpy(np.random.choice(
                a=range(params_samples.shape[0]), size=params_samples.shape[0], replace=False
            ))
            return params_samples[permuted_index, :], torch.concat((labels_h0, labels_h1))[permuted_index]
        elif estimation_method in ['prediction', 'posterior']:
            raise NotImplementedError
        else:
            raise ValueError(f"Only one of ['likelihood', 'likelihood_bap', 'prediction', 'posterior'] is supported, got {estimation_method}")
    
    def simulate_for_critical_values(
        self, 
        b_prime: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radius_poi, angle_poi = self.radius_sample(sample_shape=(b_prime, )).reshape(b_prime, 1), self.angle_sample(sample_shape=(b_prime, )).reshape(b_prime, 1)
        radius_nuisance, angle_nuisance = self.radius_sample(sample_shape=(b_prime, )).reshape(b_prime, 1), self.angle_sample(sample_shape=(b_prime, )).reshape(b_prime, 1)
        poi, nuisance = radius_poi*torch.cos(angle_poi), radius_nuisance*torch.sin(angle_nuisance)
        samples = self.likelihood_sample(n_samples=self.data_sample_size, mu=poi, nu=nuisance)
        assert samples.shape == (b_prime, self.data_sample_size, self.data_dim)
        return poi, nuisance, samples
    
    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)


class GaussianMixture(Simulator):
    
    def __init__(
        self, 
        mix_p: float,
        nuisance_dim: int,
        data_sample_size: int,
        param_space_bounds: Dict[str, float],
        param_grid_size: int = 1_000,
        num_eval_points_mc: int = 1_000  # for each dimension
    ) -> None:
        super().__init__(param_dim=1, data_dim=1+nuisance_dim, data_sample_size=data_sample_size)

        self.mix_p = mix_p
        self.nuisance_dim = nuisance_dim
        self.param_space_bounds = param_space_bounds
        self.param_grid = torch.linspace(start=param_space_bounds['low'], end=param_space_bounds['high'], steps=param_grid_size)

        self.poi_prior = Uniform(low=param_space_bounds['low'], high=param_space_bounds['high'])
        if nuisance_dim > 1:
            self.nuisance_prior = MultipleIndependent(dists=[
                pdist.Uniform(low=param_space_bounds['low']*torch.ones(1), high=param_space_bounds['high']*torch.ones(1)) for _ in range(nuisance_dim)
            ])
        elif nuisance_dim > 0:  # only 1 nuisance
            self.nuisance_prior = Uniform(low=param_space_bounds['low'], high=param_space_bounds['high'])

        self.num_eval_points_mc = num_eval_points_mc
    
    def likelihood_eval(
        self,
        poi: torch.Tensor,
        data: torch.Tensor,
        nuisance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Evaluate each batch in data over all (poi, nuisance) row-wise pair
        """
        poi = poi.reshape(-1, self.param_dim)
        if nuisance is not None:
            nuisance = nuisance.reshape(-1, self.nuisance_dim)
            assert poi.shape[0] == nuisance.shape[0], f"{poi.shape}, {nuisance.shape}"
        loc = poi if nuisance is None else torch.hstack((poi, nuisance))
        # make sure data has form (sample_shape, batch_shape, d)
        assert data.reshape(self.data_sample_size, -1, self.data_dim).shape == data.shape
        # make sure we can broadcast to evaluate on different parameters (first dim of poi and nuisance)
        data = data.reshape(self.data_sample_size, -1, 1, self.data_dim)
        first_comp = torch.transpose(self.mix_p*MultivariateNormal(loc=-loc, covariance_matrix=torch.eye(self.data_dim)).log_prob(value=data), 0, 1).double().exp()
        second_comp = (1-self.mix_p)*torch.transpose(MultivariateNormal(loc=loc, covariance_matrix=torch.eye(self.data_dim)).log_prob(value=data), 0, 1).double().exp()
        assert first_comp.shape == second_comp.shape == (data.shape[1], self.data_sample_size, poi.shape[0])
        return torch.prod(first_comp + second_comp, dim=1)

    def likelihood_sample(
        self,
        n_samples: int,  # samples within each batch
        poi: torch.Tensor,
        nuisance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        poi = poi.reshape(-1, 1)
        n_batches = poi.shape[0]
        if nuisance is not None:
            nuisance = nuisance.reshape(-1, self.nuisance_dim)
            assert poi.shape[0] == nuisance.shape[0]
            loc = torch.hstack((poi, nuisance))
        else:
            loc = poi
        which_component = Categorical(probs=torch.Tensor([[self.mix_p, 1-self.mix_p]])).sample(sample_shape=(n_batches*n_samples, )).reshape(n_batches*n_samples, ).long()
        which_loc = torch.stack(
            (-loc, loc)
        , dim=1)[torch.repeat_interleave(torch.arange(n_batches), repeats=n_samples), which_component, ...]
        assert which_loc.shape == (n_batches*n_samples, 1+self.nuisance_dim)
        return MultivariateNormal(loc=which_loc, covariance_matrix=torch.eye(self.data_dim)).rsample(sample_shape=(1, )).reshape(n_batches, n_samples, self.data_dim)
    
    def compute_mle(
        self, 
        data: torch.Tensor
    ) -> GaussianMixture:
        pass

    def compute_exact_LR(
        self,
        poi_null: torch.Tensor,
        data: torch.Tensor
    ) -> torch.Tensor:
        pass

    def compute_exact_BF(
        self,
        poi_null: torch.Tensor,
        data: torch.Tensor
    ) -> torch.Tensor:
        data = data.reshape(self.data_sample_size, 1, self.data_dim)
        poi_eval_points = self.poi_prior.rsample(sample_shape=(self.num_eval_points_mc, ))
        if self.nuisance_dim == 0:
            # no nuisances, no integral at numerator
            numerator = self.likelihood_eval(
                poi=poi_null, data=data
            )
            denominator =  torch.sum(self.likelihood_eval(poi=poi_eval_points, data=data), dim=1)/self.num_eval_points_mc
        else:
            num_eval_points = self.num_eval_points_mc**self.nuisance_dim
            nuisance_eval_points = self.nuisance_prior.rsample(sample_shape=(num_eval_points, ))
            numer_likelihood = self.likelihood_eval(
                poi=torch.tile(poi_null, dims=(num_eval_points, 1)), 
                data=data,
                nuisance=nuisance_eval_points
            )
            assert numer_likelihood.shape == (data.shape[1], num_eval_points)
            numerator = torch.sum(numer_likelihood, dim=1)/num_eval_points
            
            denom_eval_points = torch.cartesian_prod(poi_eval_points, *[nuisance_eval_points[:, ]])
            assert denom_eval_points.shape == (num_eval_points*self.num_eval_points_mc, 1+self.nuisance_dim)
            denom_likelihood = self.likelihood_eval(
                poi=denom_eval_points[:, 0],
                data=data,
                nuisance=denom_eval_points[:, 1:]
            )
            denominator = torch.sum(denom_likelihood, dim=1)/(num_eval_points*self.num_eval_points_mc)
        return numerator.double() / denominator.double()

    def simulate_for_test_statistic(
        self, 
        b: int,
        estimation_method: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement only after experiment with “exact” test statistics
        pass

    def simulate_for_critical_values(
        self, 
        b_prime: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = self.poi_prior.rsample(sample_shape=(b_prime, )).reshape(-1, 1)
        nuisance = self.nuisance_prior.rsample(sample_shape=(b_prime, )).reshape(-1, self.nuisance_dim) if self.nuisance_dim > 0 else None
        samples = self.likelihood_sample(n_samples=self.data_sample_size, poi=poi, nuisance=nuisance)
        return poi, nuisance, samples

    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)


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
        nuisance_param_dim: int,
        data_sample_size: int,
        nuisance_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        fixed_params: Dict[str, Union[float, torch.Tensor]] = {}
    ) -> None:
        super().__init__(param_dim=1, data_dim=3, data_sample_size=data_sample_size)

        # check every poi or (actual) nuisance has a prior and is not fixed; set fixed params
        assert len(priors.keys()) == (self.param_dim+nuisance_param_dim), \
            "Need to specify a prior distribution for each parameter of interest and nuisance parameter"
        assert len(priors.keys() & fixed_params.keys()) == 0, \
            "Parameters of interest and nuisance parameters are sampled through priors and cannot have fixed values. Remove them from `fixed_params`"
        self.fixed_params = fixed_params
        self.fixed_params.update({name: self._default_fixed_params[name] for name in self._default_fixed_params.keys() - fixed_params.keys()})

        # check that all nuisances are either given a prior or are fixed
        assert self._nuisance_names & (priors.keys() | fixed_params.keys()) == self._nuisance_names, \
            "Nuisance parameters must either be given a prior or be fixed (in `fixed_params`)."
        self.nuisance_param_dim = nuisance_param_dim

        self.poi_bounds = poi_bounds
        self.nuisance_bounds = nuisance_bounds
        self.poi_grid = torch.linspace(start=poi_bounds['low'], end=poi_bounds['high'], steps=poi_grid_size, dtype=torch.int)
        self.poi_grid_size = poi_grid_size

        self.poi_prior = priors.pop('s')
        self.nuisance_priors = priors if nuisance_param_dim > 0 else {}
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
        poi = self.qr_poi_prior.sample(sample_shape=(b_prime, )).reshape(b_prime, self.param_dim)
        nuisance = {
            nuisance: prior.sample(sample_shape=(b_prime, 1)).reshape(b_prime, 1)
            for nuisance, prior in self.qr_nuisance_priors.items()
        }
        samples = self._likelihood_sample(n_samples=self.data_sample_size, s=poi, **nuisance)
        return poi, nuisance, samples

    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)
