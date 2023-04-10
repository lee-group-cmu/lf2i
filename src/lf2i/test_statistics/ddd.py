from typing import Optional, Union, Callable, Dict, Tuple, Any, NamedTuple, Sequence
from tqdm import tqdm

import numpy as np
import torch
import sklearn.metrics as sk_metrics


class ClusteringOutput(NamedTuple):
    centroids: np.ndarray
    one_hot_clusters: np.ndarray
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
        std_method: str = 'z_score',
        t: Optional[int] = None
    ) -> None:
        
        if poi_dim > 1:
            # how do we bin when poi_dim > 1
            # how do we check if poi is in bin when computing ddd
            raise NotImplementedError

        self.poi_dim = poi_dim
        self.poi_bins = poi_bins
        self.nuisance_dim = nuisance_dim
        self.k = k
        self.t = t
        self.diffusion_map_type = diffusion_map_type
        self.std_method = std_method
        
        if kernel == 'rbf':
            self.kernel = lambda nuisances: sk_metrics.pairwise.rbf_kernel(X=nuisances, **kernel_kwargs)
        elif isinstance(kernel, Callable):
            self.kernel = kernel
        else:
            raise NotImplementedError
    
    @staticmethod
    def standardize(
        nuisances: np.ndarray,
        method: str
    ) -> np.ndarray:
        if method == 'z_score':
            # standardize each nuisance parameter (by column)
            return (nuisances - np.mean(nuisances, axis=0).reshape(1, -1)) / np.std(nuisances, axis=0).reshape(1, -1)
        else:
            raise NotImplementedError

    def diffusion_map(
        self,
        nuisances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return stationary distribution and diffusion map
        """
        nuisances = nuisances.reshape(-1, self.nuisance_dim)
        edge_weights = self.kernel(nuisances=self.standardize(nuisances=nuisances, method=self.std_method))
        node_weights = np.sum(edge_weights, axis=1).reshape(-1, 1)
        transition_probs = edge_weights / node_weights
        assert np.all(np.isclose(transition_probs.sum(axis=1), 1))  # row-stochastic matrix
        
        eigenvals, eigenvecs = np.linalg.eig(transition_probs)
        eigenvals, eigenvecs = np.sort(eigenvals), eigenvecs[:, np.argsort(eigenvals)]
        if self.diffusion_map_type == 'power':
            diff_map = (eigenvals ** self.t).reshape(1, -1) * eigenvecs
        else:
            raise NotImplementedError
        return node_weights / node_weights.sum(), diff_map
        
    def optimal_partition(
        self,
        diffusion_map: np.ndarray,
        stationary_distribution: np.ndarray,
        max_iter: int = 300,
        tol: float = 1e-4
    ) -> ClusteringOutput:  
        if self.k == diffusion_map.shape[0]:
            # if k == n, no clustering
            return ClusteringOutput(
                centroids=diffusion_map,
                one_hot_clusters=np.eye(self.k),
                n_iter=0
            )
        
        centroids = diffusion_map[np.random.choice(a=np.arange(diffusion_map.shape[0]), size=self.k, replace=False), :]
        for it in range(max_iter):
            distances_from_centroids = sk_metrics.pairwise_distances(diffusion_map, centroids, metric='euclidean')
            cluster_assignments = np.argmin(distances_from_centroids, axis=1)
            # i-th centroid is sum of the diffusion map assigned to i-th cluster, weighted by corresponding stationary distribution
            new_centroids = np.array([
                np.matmul(
                    diffusion_map[cluster_assignments == c, :].T, stationary_distribution[cluster_assignments == c]
                ) / stationary_distribution[cluster_assignments == c].sum() 
                for c in range(self.k)
            ])
            assert new_centroids.shape == (self.k, diffusion_map.shape[1]), f'{new_centroids.shape}'
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
                # if converged before reaching max_iter, stop
                break
            centroids = new_centroids

        # one-hot encoding of cluster assignments
        one_hot_clusters = np.zeros(shape=(diffusion_map.shape[0], self.k), dtype=int)
        one_hot_clusters[range(one_hot_clusters.shape[0]), cluster_assignments.astype(int)] = 1
        
        return ClusteringOutput(
            centroids=np.real(new_centroids),  # discard the imaginary part
            one_hot_clusters=one_hot_clusters,
            n_iter=it
        )
    
    def compute_ddd(
        self,
        y_pred: torch.Tensor,
        poi_input: torch.Tensor,
        centroids_matmul: torch.Tensor,
        one_hot_clusters: torch.Tensor
    ) -> torch.Tensor:
        """centroids_matmul == np.matmul(centroids, centroids.T)
        """
        # print(one_hot_clusters.shape, flush=True)
        ddd = torch.Tensor([0.])
        for j in range(len(self.poi_bins)-1):
            # print(f'partition {j}', flush=True)
            poi_partition_mask = ( (poi_input >= self.poi_bins[j]) & (poi_input < self.poi_bins[j+1]) ).reshape(-1, )
            y_pred_binned_1 = y_pred[poi_partition_mask]
            soft_assign_0 = torch.sum((1 - y_pred_binned_1).reshape(-1, 1) * one_hot_clusters[poi_partition_mask, :], axis=0) / torch.sum(1 - y_pred_binned_1)
            soft_assign_1 = torch.sum(y_pred_binned_1.reshape(-1, 1) * one_hot_clusters[poi_partition_mask, :], axis=0) / torch.sum(y_pred_binned_1)
            soft_assign_0, soft_assign_1 = soft_assign_0.reshape(1, one_hot_clusters.shape[1]), soft_assign_1.reshape(1, one_hot_clusters.shape[1])
            # print(soft_assign_0.shape, soft_assign_1.shape, centroids.shape, flush=True)
            ddd_j = torch.matmul(
                torch.matmul((soft_assign_0 - soft_assign_1), centroids_matmul),
                (soft_assign_0 - soft_assign_1).T
            )
            ddd += ddd_j.reshape(1, )
            # print(ddd_j, flush=True)
        
        return ddd


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
    
    def fit(
        self,
        X: torch.Tensor, 
        y: torch.Tensor,
        epochs: int, 
        batch_size: int,
        ddd_gamma: Optional[float] = None
    ) -> None:
        
        if self.ddd is not None:
            print("Constructing diffusion map ...", flush=True)
            # NOTE: assumes POIs come first, nuisances second, and then data
            stationary_dist, diff_map = self.ddd.diffusion_map(nuisances=X[:, self.ddd.poi_dim:self.ddd.poi_dim+self.ddd.nuisance_dim].numpy())
            print("Computing optimal partition ...", flush=True)
            clustering_output = self.ddd.optimal_partition(
                diffusion_map=diff_map, stationary_distribution=stationary_dist,
            )
            one_hot_clusters = torch.from_numpy(clustering_output.one_hot_clusters).to(self.device)
            # compute this only one time for less computations
            centroids_matmul = torch.from_numpy(np.matmul(clustering_output.centroids.astype(np.double), clustering_output.centroids.T.astype(np.double)))
            assert centroids_matmul.shape[0] == centroids_matmul.shape[1]
            print(f"Converged after {clustering_output.n_iter} iterations", flush=True)

        self.model.train()
        for _ in tqdm(range(epochs), desc="Training NNClassifier"):
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
                    ddd_loss = self.ddd.compute_ddd(
                        y_pred=batch_predictions,
                        poi_input=batch_X[:, :self.ddd.poi_dim],  # NOTE: assumes POIs come first
                        centroids_matmul=centroids_matmul,
                        one_hot_clusters=one_hot_clusters[batch_shuffle_idx.to(self.device), :]
                    )
                    batch_loss = batch_loss.reshape(1, ) + ddd_gamma*ddd_loss
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
