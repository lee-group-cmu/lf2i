from typing import Optional, Union, Dict, Tuple, Callable

import math
import numpy as np
from scipy import integrate
import torch
from torch.distributions import (
    Distribution, 
    MultivariateNormal,
    Uniform, 
    Categorical
)
import jax
import jax.numpy as jax_np
from sklearn.mixture import GaussianMixture as skGaussianMixture
from sbi.utils import MultipleIndependent
from pyro import distributions as pdist

from lf2i.simulator._base import Simulator
from lf2i.utils.optimize import minimize_jax_grad


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
        param_grid_size: int = 1_000
    ) -> None:
        super().__init__(poi_dim=1, data_dim=1+nuisance_dim, data_sample_size=data_sample_size, nuisance_dim=nuisance_dim)

        self.mix_p = mix_p
        self.param_space_bounds = param_space_bounds
        self.param_grid = torch.linspace(start=param_space_bounds['low'], end=param_space_bounds['high'], steps=param_grid_size)

        self.poi_prior = Uniform(low=param_space_bounds['low'], high=param_space_bounds['high'])
        if nuisance_dim > 1:
            self.nuisance_prior = MultipleIndependent(dists=[
                pdist.Uniform(low=param_space_bounds['low']*torch.ones(1), high=param_space_bounds['high']*torch.ones(1)) for _ in range(nuisance_dim)
            ])
        elif nuisance_dim == 1:
            self.nuisance_prior = Uniform(low=param_space_bounds['low'], high=param_space_bounds['high'])
        else:
            self.nuisance_prior = None
    
    """
    def likelihood_eval(
        self,
        poi: torch.Tensor,
        data: torch.Tensor,
        nuisance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Evaluate each batch in data over all (poi, nuisance) row-wise pairs
        poi = poi.reshape(-1, self.param_dim)
        if nuisance is not None:
            nuisance = nuisance.reshape(-1, self.nuisance_dim)
            assert poi.shape[0] == nuisance.shape[0], f"{poi.shape}, {nuisance.shape}"
        loc = poi if nuisance is None else torch.hstack((poi, nuisance))
        # make sure data has form (sample_shape, batch_shape, d)
        assert data.reshape(self.data_sample_size, -1, self.data_dim).shape == data.shape
        # make sure we can broadcast to evaluate on different parameters (first dim of poi and nuisance)
        data = data.reshape(self.data_sample_size, -1, 1, self.data_dim)
        first_comp = self.mix_p*torch.transpose(MultivariateNormal(loc=-loc, covariance_matrix=torch.eye(self.data_dim)).log_prob(value=data), 0, 1).double().exp()
        second_comp = (1-self.mix_p)*torch.transpose(MultivariateNormal(loc=loc, covariance_matrix=torch.eye(self.data_dim)).log_prob(value=data), 0, 1).double().exp()
        assert first_comp.shape == second_comp.shape == (data.shape[1], self.data_sample_size, poi.shape[0])
        return torch.prod(first_comp + second_comp, dim=1)
    """
    def likelihood_eval(
        self, 
        *nuisances: Union[float, np.ndarray, jax_np.ndarray],
        poi: Union[float, np.ndarray, jax_np.ndarray],
        data: Union[np.ndarray, jax_np.ndarray],
        log: bool = False
    ) -> Union[np.ndarray, jax_np.ndarray]:

        def std_gauss_pdf(loc, X, d):
            jax.config.update("jax_enable_x64", True)
            return ((2*math.pi)**(-d/2)) * jax_np.exp(-0.5 * jax_np.sum(jax_np.multiply(X-loc, X-loc), axis=1).astype(jax_np.double))

        if isinstance(poi, (np.ndarray, jax_np.ndarray)):
            assert (poi.shape == (self.param_dim, )) or (poi.shape == (1, self.param_dim))
        else:
            poi = jax_np.array([poi])
        if nuisances:
            if isinstance(nuisances[0], float):
                loc = jax_np.hstack((poi.reshape(1, self.param_dim), jax_np.array([*nuisances]).reshape(1, self.nuisance_dim)))
            else:
                loc = jax_np.hstack((poi.reshape(1, self.param_dim), nuisances[0].reshape(1, self.nuisance_dim)))
        else:
            loc = poi.reshape(1, 1)
        data = data.reshape(-1, self.data_dim)
        joint_eval = self.mix_p*std_gauss_pdf(loc=-loc, X=data, d=self.param_dim+self.nuisance_dim) + \
            (1-self.mix_p)*std_gauss_pdf(loc=loc, X=data, d=self.param_dim+self.nuisance_dim)
        if log:
            return jax_np.sum(jax_np.log(joint_eval))
        else:
            return jax_np.prod(joint_eval)
    
    def likelihood_eval_np(
        self, 
        *nuisances: Union[float, np.ndarray],
        poi: Union[float, np.ndarray],
        data: np.ndarray,
        log: bool = False
    ) -> np.ndarray:

        def std_gauss_pdf(loc, X, d):
            return ((2*math.pi)**(-d/2)) * np.exp(-0.5 * np.sum(np.multiply(X-loc, X-loc), axis=1).astype(np.double))

        if isinstance(poi, np.ndarray):
            assert (poi.shape == (self.param_dim, )) or (poi.shape == (1, self.param_dim))
        else:
            poi = np.array([poi])
        if nuisances:
            if isinstance(nuisances[0], float):
                loc = np.hstack((poi.reshape(1, self.param_dim), np.array([*nuisances]).reshape(1, self.nuisance_dim)))
            else:
                loc = np.hstack((poi.reshape(1, self.param_dim), nuisances[0].reshape(1, self.nuisance_dim)))
        else:
            loc = poi.reshape(1, 1)
        data = data.reshape(-1, self.data_dim)
        joint_eval = self.mix_p*std_gauss_pdf(loc=-loc, X=data, d=self.param_dim+self.nuisance_dim) + \
            (1-self.mix_p)*std_gauss_pdf(loc=loc, X=data, d=self.param_dim+self.nuisance_dim)
        if log:
            return np.sum(np.log(joint_eval))
        else:
            return np.prod(joint_eval)
        

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
    
    def max_likelihood_eval(
        self, 
        data: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        mixture = skGaussianMixture(
                n_components=2,
                covariance_type='spherical',
                precisions_init=np.array([1, 1])
            )
        if self.data_sample_size == 1:
            # for consistency with n > 1, eval samples with known weights and cov, and means given by +- sample
            assert (data.shape == (self.data_dim, )) or (data.shape == (1, self.data_dim))
            mixture.means_ = np.vstack((
                -data.reshape(1, self.data_dim),
                data.reshape(1, self.data_dim),
            ))
            mixture.precisions_cholesky_ = np.array([1., 1.])
            mixture.weights_ = np.array([self.mix_p, 1-self.mix_p])
        else:
            mixture.fit(X=data)
        return lambda samples: mixture.score_samples(X=samples)
    
    def compute_exact_logLR(
        self,
        poi_null: torch.Tensor,
        data: torch.Tensor,
        nuisance_opt_init_guess: Optional[jax_np.ndarray] = None
    ) -> float:
        assert data.shape == (self.data_sample_size, self.data_dim)
        if self.nuisance_dim == 0:
            numerator = torch.from_numpy(self.likelihood_eval(poi=poi_null.numpy(), data=data, log=True))
            denominator = torch.sum(torch.from_numpy(self.max_likelihood_eval(data=data.numpy())(samples=data.numpy())))
        else:
            # eval max likelihood when parameter space is restricted by phi_0
            # cannot use sklearn mixture, hence maximize the log-likelihood directly (minimize -1*log-likelihood)
            numerator = -1*minimize_jax_grad(
                fun=lambda *nuisances: -1*(self.likelihood_eval(
                *nuisances,
                poi=poi_null.numpy(),
                data=data.numpy(),
                log=True
            )),
            x0=nuisance_opt_init_guess or jax_np.ones(shape=(self.nuisance_dim, ), dtype=jax_np.float32),
            bounds=[[self.param_space_bounds['low'], self.param_space_bounds['high']] for _ in range(self.nuisance_dim)]
            ).fun

            denominator = torch.sum(torch.from_numpy(self.max_likelihood_eval(data=data.numpy())(samples=data.numpy())))
        return (torch.Tensor([numerator]) - denominator).item()

    def compute_exact_BF(
        self,
        poi_null: torch.Tensor,
        data: torch.Tensor
    ) -> float:
        assert data.shape == (self.data_sample_size, self.data_dim)
        if self.nuisance_dim == 0:
            numerator = self.likelihood_eval_np(poi=poi_null.numpy(), data=data)
            denominator, _ = integrate.quad(
                lambda poi: self.likelihood_eval_np(poi=poi, data=data), 
                a=self.param_space_bounds['low'], b=self.param_space_bounds['high']
            )
        else:
            numerator, _ = integrate.nquad(
                lambda *nuisances: self.likelihood_eval_np(*nuisances, poi=poi_null.numpy(), data=data.numpy()), 
                ranges=[[self.param_space_bounds['low'], self.param_space_bounds['high']] for _ in range(self.nuisance_dim)]
            )
            denominator, _ = integrate.nquad(
                lambda poi, *nuisances: self.likelihood_eval_np(*nuisances, poi=poi, data=data.numpy()), 
                ranges=[[self.param_space_bounds['low'], self.param_space_bounds['high']] for _ in range(self.param_dim+self.nuisance_dim)]
            )
        return numerator / denominator

    def simulate_for_test_statistic(
        self, 
        b: int,
        estimation_method: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if estimation_method == 'likelihood':
            raise NotImplementedError
        elif estimation_method == 'likelihood_bap': # best average power (Heinrich, 2022)
            # sample nulls
            poi_null, nuisance_null = self.poi_prior.sample(sample_shape=(1, )), self.nuisance_prior.sample(sample_shape=(1, ))
            samples_null = self.likelihood_sample(n_samples=b//2, poi=poi_null.reshape(-1, self.param_dim), nuisance=nuisance_null.reshape(1, self.nuisance_dim))
            labels_null = torch.ones(b//2)

            # sample alternatives
            scale = self.poi_prior.sample(sample_shape=(1, )).item()  # use value in poi range to determine distance from null
            scale = min(scale, min(poi_null.item()-self.param_space_bounds['low'], self.param_space_bounds['high']-poi_null.item()))  # make sure alternatives remain in poi range
            poi_h1_left, poi_h1_right = poi_null-scale, poi_null+scale
            samples_h1_left = self.likelihood_sample(n_samples=b//4, poi=poi_h1_left.reshape(1, self.param_dim), nuisance=nuisance_null.reshape(1, self.nuisance_dim))
            samples_h1_right = self.likelihood_sample(n_samples=b//4, poi=poi_h1_right.reshape(1, self.param_dim), nuisance=nuisance_null.reshape(1, self.nuisance_dim))
            labels_h1 = torch.zeros((b//4)*2)

            params_samples = torch.hstack((
                torch.tile(poi_null, dims=((b//2)+(b//4)*2, 1)),
                torch.vstack((
                    samples_null.reshape(b//2, self.data_dim), 
                    samples_h1_left.reshape(b//4, self.data_dim), 
                    samples_h1_right.reshape(b//4, self.data_dim)
                ))
            ))

            permuted_index = torch.from_numpy(np.random.choice(
                a=range(params_samples.shape[0]), size=params_samples.shape[0], replace=False
            ))
            return params_samples[permuted_index, :], torch.concat((labels_null, labels_h1))[permuted_index]
        elif estimation_method in ['prediction', 'posterior']:
            raise NotImplementedError
        else:
            raise ValueError(f"Only one of ['likelihood', 'likelihood_bap', 'prediction', 'posterior'] is supported, got {estimation_method}")

    def simulate_for_critical_values(
        self, 
        b_prime: int,
        fixed_nuisance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = self.poi_prior.sample(sample_shape=(b_prime, )).reshape(-1, 1)
        if fixed_nuisance is None:
            nuisance = self.nuisance_prior.sample(sample_shape=(b_prime, )).reshape(-1, self.nuisance_dim) if self.nuisance_dim > 0 else None
        else:
            assert fixed_nuisance.shape == (b_prime, self.nuisance_dim)
            nuisance = fixed_nuisance
        samples = self.likelihood_sample(n_samples=self.data_sample_size, poi=poi, nuisance=nuisance)
        return poi, nuisance, samples

    def simulate_for_diagnostics(self, b_doubleprime: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.simulate_for_critical_values(b_doubleprime)
