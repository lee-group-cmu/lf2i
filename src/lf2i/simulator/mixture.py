from typing import Optional, Union, Dict, Tuple

import math
import numpy as np
import torch
from torch.distributions import (
    Distribution, 
    MultivariateNormal,
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
        super().__init__(poi_dim=1, data_dim=2, data_sample_size=data_sample_size, nuisance_dim=1)

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
        poi_space_bounds: Dict[str, float],
        poi_grid_size: int = 1_000,
        num_eval_points_mc: int = 1_000  # for each dimension
    ) -> None:
        super().__init__(poi_dim=1, data_dim=1+nuisance_dim, data_sample_size=data_sample_size, nuisance_dim=nuisance_dim)

        self.mix_p = mix_p
        self.poi_space_bounds = poi_space_bounds
        self.poi_grid = torch.linspace(start=poi_space_bounds['low'], end=poi_space_bounds['high'], steps=poi_grid_size)

        self.poi_prior = Uniform(low=poi_space_bounds['low'], high=poi_space_bounds['high'])
        if nuisance_dim > 1:
            self.nuisance_prior = MultipleIndependent(dists=[  # same bounds as poi
                pdist.Uniform(low=poi_space_bounds['low']*torch.ones(1), high=poi_space_bounds['high']*torch.ones(1)) for _ in range(nuisance_dim)
            ])
        elif nuisance_dim > 0:  # only 1 nuisance
            self.nuisance_prior = Uniform(low=poi_space_bounds['low'], high=poi_space_bounds['high'])

        self.num_eval_points_mc = num_eval_points_mc
    
    def likelihood_eval(
        self,
        poi: torch.Tensor,
        data: torch.Tensor,
        nuisance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Evaluate each batch in data over all (poi, nuisance) row-wise pair
        """
        poi = poi.reshape(-1, self.poi_dim)
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
