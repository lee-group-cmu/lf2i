from typing import Optional, Union, Callable, Dict, Tuple, Any, NamedTuple, Sequence
from tqdm import tqdm
import warnings

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
import torch
import jax.numpy as jnp
import sklearn.metrics as sk_metrics


class DiffusionMapOutput(NamedTuple):
    stationary_distribution: np.ndarray
    transition_probs: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diffusion_map: np.ndarray


class ClusteringOutput(NamedTuple):
    centroids: np.ndarray  # k x n (if we retain all n eigenvectors)
    one_hot_clusters: np.ndarray  # n x k, where n is number of samples
    n_iter: int


class DDD:

    # TODO: convert everything to torch ?

    def __init__(
        self,
        poi_dim: int,
        poi_bins: np.ndarray,
        nuisance_dim: int,
        k: int,
        kernel: Union[str, Callable],
        kernel_kwargs: Dict[str, Any] = {},
        diffusion_map_type: str = 'power',
        std_method: Union[str, None] = 'z_score',
        t: Optional[int] = None,
        nystrom_subset_size: Optional[int] = None
    ) -> None:
        
        if poi_dim > 1:
            # how do we bin when poi_dim > 1
            # how do we check if poi is in bin when computing ddd
            raise NotImplementedError
        
        assert t == 1, "Check logic with eigenvalues and trans probs, especially in Nystrom method. Need to use P^t and its eigenvalues! Not P and its eigenvalues"

        self.poi_dim = poi_dim
        self.poi_bins = poi_bins
        self.nuisance_dim = nuisance_dim
        self.k = k
        self.t = t
        self.diffusion_map_type = diffusion_map_type
        self.std_method = std_method
        self.nystrom_subset_size = nystrom_subset_size
        
        if kernel == 'rbf':
            self.kernel = lambda nuisances: sk_metrics.pairwise.rbf_kernel(X=nuisances, **kernel_kwargs)
        elif isinstance(kernel, Callable):
            self.kernel = kernel
        else:
            raise NotImplementedError
    
    @staticmethod
    def normalize(
        nuisances: np.ndarray,
        method: str
    ) -> np.ndarray:
        if method is None:
            return nuisances
        elif method == 'z_score':
            # standardize each nuisance parameter (by column)
            return (nuisances - np.mean(nuisances, axis=0).reshape(1, -1)) / np.std(nuisances, axis=0).reshape(1, -1)
        else:
            raise NotImplementedError
    
    def _eig_to_diff_map(
        self,
        eigenvals: np.ndarray,
        eigenvecs: np.ndarray
    ) -> np.ndarray:
        if self.diffusion_map_type == 'power':
            # discard the imaginary part, if any. Complex data not supported in many frameworks (e.g., SciKit Learn)
            return np.real((eigenvals ** self.t).reshape(1, -1) * eigenvecs)
        elif self.diffusion_map_type == 'average':
            return np.real((eigenvals.reshape(1, -1) / (1-eigenvals.reshape(1, -1))) * eigenvecs)
        else:
            raise NotImplementedError

    def diffusion_map(
        self,
        nuisances: np.ndarray  # assumes they are shuffled already
    ) -> Tuple[np.ndarray]:
        if (nuisances.shape[0] > 10_000) and (self.nystrom_subset_size is None):
            warnings.warn(
                message=f"Eigendecomposition might be computationally infeasible for n = {nuisances.shape[0]}. Consider using Nyström method.", 
                category=UserWarning
            )

        subset_n = self.nystrom_subset_size or nuisances.shape[0]
        edge_weights = self.kernel(nuisances=self.normalize(nuisances=nuisances.reshape(-1, self.nuisance_dim), method=self.std_method))
        node_weights = np.sum(edge_weights[:subset_n, :subset_n], axis=1).reshape(-1, 1)  # needed only for subset if using Nystrom
        transition_probs = edge_weights[:subset_n, :subset_n] / node_weights
        assert np.all(np.isclose(transition_probs.sum(axis=1), 1))  # row-stochastic matrix
        
        eigenvals, eigenvecs = np.linalg.eig(transition_probs)
        diff_map = self._eig_to_diff_map(eigenvals=eigenvals, eigenvecs=eigenvecs)

        return DiffusionMapOutput(
            stationary_distribution=node_weights / node_weights.sum(),  # only for nystrom subset
            transition_probs=edge_weights[:, :subset_n] / np.sum(edge_weights[:, :subset_n], axis=1).reshape(-1, 1),  # for all nuisances (n x subset_n)
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            diffusion_map=diff_map
        )
    
    def nystrom_approximation(
        self,
        diff_map_output: DiffusionMapOutput
    ) -> np.ndarray:
        # NOTE: this implementation follows http://www.iro.umontreal.ca/~lisa/pointeurs/bengio_eigenfunctions_nc_2004.pdf
        # TODO: check https://www.cs.cmu.edu/~muli/file/nystrom_icml10.pdf for a more efficient Nyström approximation method
        # TODO: uniform sampling of column might not be “optimal”. There are smarter ways, see https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37831.pdf
        
        nystrom_eigenvecs = np.sqrt(self.nystrom_subset_size) * np.matmul(
            # NOTE: assumes first self.nystrom_subset_size rows in nuisances were used for eigendecomposition
            diff_map_output.transition_probs,  # n x self.nystrom_subset_size
            diff_map_output.eigenvectors  # self.nystrom_subset_size x self.nystrom_subset_size (if we retain all eigenvectors)
        ) / diff_map_output.eigenvalues.reshape(1, -1)
        assert nystrom_eigenvecs.shape == (diff_map_output.transition_probs.shape[0], diff_map_output.eigenvectors.shape[1])
        nystrom_eigenvals = diff_map_output.eigenvalues / self.nystrom_subset_size
        return self._eig_to_diff_map(eigenvals=nystrom_eigenvals, eigenvecs=nystrom_eigenvecs)
        
    def diffusion_k_means(
        self,
        diffusion_map: np.ndarray,
        stationary_distribution: np.ndarray,
        max_iter: int = 300,  # max_iter and tol taken from sklearn Kmeans settings
        tol: float = 1e-4
    ) -> ClusteringOutput:  
        if self.k == diffusion_map.shape[0]:
            # if k == n, no clustering
            return ClusteringOutput(
                centroids=diffusion_map,
                one_hot_clusters=np.eye(self.k),  # TODO: make this sparse
                n_iter=0
            )
        else:
            centroids = diffusion_map[np.random.choice(a=np.arange(diffusion_map.shape[0]), size=self.k, replace=False), :]
            for it in tqdm(range(max_iter), desc='Computing optimal clustering ...'):
                cluster_assignments = np.argmin(scipy_cdist(diffusion_map, centroids, metric='euclidean'), axis=1)
                # i-th centroid is sum of the diffusion map assigned to i-th cluster, weighted by corresponding stationary distribution
                new_centroids = np.array([
                    np.matmul(
                        diffusion_map[cluster_assignments == c, :].T, stationary_distribution[cluster_assignments == c]
                    ) / stationary_distribution[cluster_assignments == c].sum() 
                    for c in range(self.k)
                ]).reshape(self.k, diffusion_map.shape[1])
                if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
                    # if converged before reaching max_iter, stop
                    break
                centroids = new_centroids

            # one-hot encoding of cluster assignments
            one_hot_clusters = np.zeros(shape=(diffusion_map.shape[0], self.k), dtype=int)
            one_hot_clusters[range(one_hot_clusters.shape[0]), cluster_assignments.astype(int)] = 1
            
            return ClusteringOutput(
                centroids=new_centroids,
                one_hot_clusters=one_hot_clusters,  # TODO: make this sparse
                n_iter=it
            )
    
    def torch_ddd_regularizer(
        self,
        y_pred: torch.Tensor,
        poi_input: torch.Tensor,
        centroids_matmul: torch.Tensor,
        one_hot_clusters: torch.Tensor,
        device: Union[str, Any]
    ) -> torch.Tensor:
        """centroids_matmul == np.matmul(centroids, centroids.T)
        """
        ddd = torch.Tensor([0.]).to(device)
        for j in range(len(self.poi_bins)-1):
            poi_partition_mask = ( (poi_input >= self.poi_bins[j]) & (poi_input < self.poi_bins[j+1]) ).reshape(-1, )
            y_pred_binned_1 = y_pred[poi_partition_mask]
            soft_assign_0 = torch.sum(-torch.log(1 - y_pred_binned_1).reshape(-1, 1) * one_hot_clusters[poi_partition_mask, :], axis=0) / torch.sum(-torch.log(1 - y_pred_binned_1))
            soft_assign_1 = torch.sum(-torch.log(y_pred_binned_1).reshape(-1, 1) * one_hot_clusters[poi_partition_mask, :], axis=0) / torch.sum(-torch.log(y_pred_binned_1))
            soft_assign_0, soft_assign_1 = soft_assign_0.reshape(1, one_hot_clusters.shape[1]), soft_assign_1.reshape(1, one_hot_clusters.shape[1])
            ddd_j = torch.matmul(
                torch.matmul((soft_assign_0 - soft_assign_1), centroids_matmul),
                (soft_assign_0 - soft_assign_1).T
            )
            ddd += ddd_j.reshape(1, )
        return ddd

    def xgb_ddd_objective(
        loss: Union[str, Callable[[np.ndarray, np.ndarray], float]],
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if loss == 'logistic':
            pass
        loss_fun = lambda y_pred: loss(y_true=y_true, y_pred=y_pred)

        grad = None
        hess = None

        return grad, hess


class NNClassifier(torch.nn.Module):

    def __init__(
        self,
        input_d: int,
        hidden_layer_shapes: Sequence[int],
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_d = input_d
        self.hidden_layer_shapes = hidden_layer_shapes
        self.hidden_activation = hidden_activation
        self.dropout_p = dropout_p
        
        self.build_model()

    def build_model(self) -> None:
        # input
        self.model = [
            torch.nn.Linear(self.input_d, self.hidden_layer_shapes[0]), 
            self.hidden_activation, 
            #torch.nn.Dropout(p=self.dropout_p)
        ]
        # hidden 
        for i in range(0, len(self.hidden_layer_shapes)-1):
            self.model += [
                torch.nn.Linear(self.hidden_layer_shapes[i], self.hidden_layer_shapes[i+1]), 
                self.hidden_activation, 
                #torch.nn.Dropout(p=self.dropout_p)
            ]
        # output
        self.model += [torch.nn.Linear(self.hidden_layer_shapes[-1], 1), torch.nn.Sigmoid()]
        self.model = torch.nn.Sequential(*self.model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Learner:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        ddd: bool = False,
        device: str = "cpu",
        **kwargs
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss = loss.to(self.device)
        if ddd:
            self.ddd = DDD(**kwargs)
            self.ddd_loss_trajectory = []
        else:
            self.ddd = None
        self.loss_trajectory = []
    
    #def __getstate__(self):
    #    # return state values to be pickled. Don't save self.ddd because unpicklable
    #    return self.device, self.model, self.optimizer, self.loss, self.ddd_loss_trajectory, self.loss_trajectory
    #
    #def __setstate__(self, state):
    #    # restore state from the unpickled state values
    #    self.device, self.model, self.optimizer, self.loss, self.ddd_loss_trajectory, self.loss_trajectory = state
    
    def fit(
        self,
        X: torch.Tensor, # NOTE: assumes order poi, nu, data
        y: torch.Tensor, # NOTE: labels 1/0
        epochs: int, 
        batch_size: int,
        ddd_gamma: Optional[float] = None
    ) -> None:
        
        # TODO: move this to dedicated on_torch_training_start method in DDD
        if self.ddd is not None:
            print("Constructing diffusion map ...", flush=True)
            # NOTE: assumes POIs come first, nuisances second, and then data
            diff_map_output = self.ddd.diffusion_map(nuisances=X[:, self.ddd.poi_dim:self.ddd.poi_dim+self.ddd.nuisance_dim].numpy())
            print("Computing optimal partition ...", flush=True)
            clustering_output = self.ddd.diffusion_k_means(diffusion_map=diff_map_output.diffusion_map, stationary_distribution=diff_map_output.stationary_distribution)
            print(f"Converged after {clustering_output.n_iter} iterations", flush=True)
            
            if self.ddd.nystrom_subset_size:
                estimated_diff_map = self.ddd.nystrom_approximation(diff_map_output=diff_map_output)
                print(f"Computing Nystrom cluster assignments ...", flush=True)
                cluster_assignments = np.argmin(scipy_cdist(estimated_diff_map, clustering_output.centroids, metric='euclidean'), axis=1)
                print(np.unique(cluster_assignments), flush=True)
                one_hot_clusters = np.zeros(shape=(estimated_diff_map.shape[0], self.ddd.k), dtype=int)
                one_hot_clusters[range(one_hot_clusters.shape[0]), cluster_assignments.astype(int)] = 1

            one_hot_clusters = one_hot_clusters if self.ddd.nystrom_subset_size else clustering_output.one_hot_clusters
            one_hot_clusters = torch.from_numpy(one_hot_clusters).to(self.device)

            centroids_matmul = torch.from_numpy(np.matmul(clustering_output.centroids, clustering_output.centroids.T)).to(self.device)
            assert centroids_matmul.shape[0] == centroids_matmul.shape[1]
        
        print(f"Start training ...", flush=True)
        self.model.train()
        for _ in tqdm(range(epochs), desc="Training NN Classifier"):
            shuffle_idx = torch.from_numpy(np.random.choice(a=np.arange(X.shape[0]), size=X.shape[0], replace=False))
            epoch_joint_losses = []
            if self.ddd is not None:
                epoch_ddd_losses = []
            for idx in range(0, X.shape[0], batch_size):
                self.optimizer.zero_grad()
                
                batch_shuffle_idx = shuffle_idx[idx: min(idx + batch_size, y.shape[0])]
                batch_X = X[shuffle_idx, :][idx: min(idx + batch_size, X.shape[0]), :].float().to(self.device).requires_grad_(False)
                batch_y = y[shuffle_idx][idx: min(idx + batch_size, y.shape[0])].reshape(-1, 1).float().to(self.device).requires_grad_(False)
                
                batch_predictions = self.model(batch_X)
                batch_loss = self.loss(batch_predictions, batch_y)
                if self.ddd is not None:
                    ddd_loss = self.ddd.torch_ddd_regularizer(
                        y_pred=batch_predictions,
                        poi_input=batch_X[:, :self.ddd.poi_dim],
                        centroids_matmul=centroids_matmul,
                        one_hot_clusters=one_hot_clusters[batch_shuffle_idx.to(self.device), :],
                        device=self.device
                    )
                    batch_loss = batch_loss.reshape(1, ) + ddd_gamma*ddd_loss
                    print(batch_loss, ddd_loss, flush=True)
                batch_loss.backward()
                self.optimizer.step()
                epoch_joint_losses.append(batch_loss.cpu().detach().numpy())
                if self.ddd is not None:
                    epoch_ddd_losses.append(ddd_loss.cpu().detach().numpy())
            self.loss_trajectory.append(np.mean(epoch_joint_losses))
            if self.ddd is not None:
                self.ddd_loss_trajectory.append(np.mean(epoch_ddd_losses))

    def predict(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        return self.model(X.to(self.device).requires_grad_(False)).cpu().detach()

    def predict_proba(
        self, 
        X: torch.Tensor
    ) -> torch.Tensor:
        proba_positive_class = self.predict(X=X).reshape(-1, 1)
        return torch.hstack((1-proba_positive_class, proba_positive_class))
