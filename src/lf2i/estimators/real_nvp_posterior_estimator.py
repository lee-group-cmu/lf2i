from typing import Union, Any, Dict, Tuple, Optional, Callable
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

import sbibm
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import bayesflow as bf
import tensorflow as tf
from sbi.inference.posteriors.base_posterior import NeuralPosterior


class PosteriorEstimator:
    def __init__(self, poi_dim: int, x_dim: int, estimator_kwargs: dict, checkpoint_path='./checkpoints'):
        self.poi_dim = poi_dim
        self.x_dim = x_dim
        self.estimator_kwargs = estimator_kwargs
        self.checkpoint_path = checkpoint_path

        self.inference_net = bf.networks.InvertibleNetwork(
            num_params=self.poi_dim,
            num_coupling_layers=estimator_kwargs['num_coupling_layers'],
            coupling_settings={"dense_args": dict(kernel_regularizer=None), "dropout": False},
        )
        self.summary_net = bf.networks.DeepSet(summary_dim=estimator_kwargs['summary_dim'])

        self.batch_size = estimator_kwargs['batch_size']
        self.epochs = estimator_kwargs['epochs']
        self.iterations_per_epoch = estimator_kwargs['iterations_per_epoch']
        self.checkpoint_path = checkpoint_path

        self.amortizer = bf.amortizers.AmortizedPosterior(self.inference_net, self.summary_net)
        self.trainer = bf.trainers.Trainer(amortizer=self.amortizer, checkpoint_path=self.checkpoint_path) # generative_model=self.model,
        self.save_counter = self.trainer.checkpoint.save_counter
        self._trained = bool(self.save_counter)

        try:
            self.history = {
                'train_losses': self.trainer.loss_history.total_loss,
                'val_losses': self.trainer.loss_history.total_val_loss,
            }
        except:
            self.history = None


    def estimate(self, parameters, samples):
        if tf.compat.v1.train.checkpoint_exists(f'{self.checkpoint_path}/checkpoint'):
            self.trainer.load_pretrained_network()
            print("Test statistic already trained.\n")

        else:
            print("Training test statistic...")
            parameters = np.array(parameters).reshape(-1, 1, self.poi_dim)
            samples = np.array(samples).reshape(-1, 1, self.x_dim)

            simulations_dict = {
                'sim_data' : samples,
                'prior_draws' : parameters,
            }
            _ = self.trainer.train_offline(
                simulations_dict=simulations_dict,
                epochs=self.epochs,
                batch_size=self.batch_size,
                save_checkpoint=True,
            )

            try:
                self.history = {
                    'train_losses': self.trainer.loss_history.total_loss,
                    'val_losses': self.trainer.loss_history.total_val_loss,
                }
            except:
                self.history = None

            print("Training finished.\n")

        self._trained = True
        return


    def sample(self, x: Union[torch.Tensor, np.ndarray], sample_shape: Optional[Union[int, tuple]]=1, num_samples: Optional[int]=None, **kwargs):
        # assert self._trained
        assert x is not None
        x = x.reshape(-1, 1, x.shape[-1])
        x = x if isinstance(x, np.ndarray) else x.detach().numpy()

        if isinstance(sample_shape, int):
            num_samples = sample_shape
        elif isinstance(sample_shape, tuple):
            num_samples = sample_shape[0]
        else:
            assert num_samples is not None

        input_dict = {'summary_conditions': x, 'parameters': None, 'direct_conditions': None}
        samples = self.amortizer.sample(input_dict=input_dict, n_samples=num_samples)
        return torch.tensor(samples)


    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, norm_posterior: bool=False, leakage_correction_params: dict=None):
        assert theta.shape[-1] == self.poi_dim
        assert x.shape[-1] == self.x_dim

        theta = theta.reshape(-1, self.poi_dim)
        x = x.reshape(-1, self.x_dim)

        if len(x) == 1:
            x = x.expand(len(theta), -1)

        params = np.array(theta)
        samples = np.array(x).reshape(-1, 1, self.x_dim)

        inputs = {'summary_conditions': samples, 'parameters': params, 'direct_conditions': None}
        return torch.tensor(self.amortizer.log_prob(input_dict=inputs))


    def get_params(self):
        return {
            'poi_dim': self.poi_dim,
            'x_dim': self.x_dim,
            'estimator_kwargs': self.estimator_kwargs,
            'checkpoint_path': self.checkpoint_path
        }


    @staticmethod
    def from_params(params):
        return PosteriorEstimator(
            poi_dim=params['poi_dim'],
            x_dim=params['x_dim'],
            estimator_kwargs=params['estimator_kwargs'],
            checkpoint_path=params['checkpoint_path']
        )