import os
from typing import List, Tuple, Union
import numpy as np
from datasets.base_dataset import BaseDataset


class SyntheticDataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print(f"Loading synthetic dataset with following parameters: {config}")
        self.x_data, self.y_data = self.generate_data_points(**config)
        self.x_data = self.x_data.astype(np.float32)
        print(f"Data shape : {self.x_data.shape}")
        print(f"Target shape : {self.y_data.shape}")
        print("Synthetic dataset loaded.")
    
    def generate_data_points(self,
            seed : Union[int, None],
            n_samples : int,
            dimension : int,
            n_clusters : int,
            proportion : Union[str, List[float]] = 'uniform',
            means : Union[str, List[np.ndarray]] = 'random',
            stds : Union[List[List[float]], List[float], float] = .1,
            min_std : float = .1,
            max_std : float = .5,
            overlapping : str = 'none', # none, ellipsoidal, non_gaussian_laplace, non_gaussian_uniform, nested, bridging
            pre_visualization : bool = False,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data points. The data points are generated from a mixture of gaussians.
        The means of the Gaussian are themselves random (following N(0, In)) and their std are uniformly sampled
        from the stds parameters.

        Args:
            seed (Union[int, None]): the seed of the random generator
            n_samples (int): the number of datapoints generated
            dimension (int): the dimension of the datapoints
            n_clusters (int): the number of clusters
            proportion (Union[str, List[float]], optional): the proportion of each cluster. If 'uniform', each cluster will have the same proportion. If 'random', the proportion will be sampled from a normal distribution. Defaults to 'uniform'.
            means (np.ndarray, optional): the means of the gaussians as a list of d-dimensional vectors. Defaults to 'random' (means are sampled from N(0, I_d)).
            stds (Union[List[List[float, float]], List[float], float], optional): the stds of the gaussians (either as a float, as a list of std, or as a list of d-tuple of std). If None, they will be sampled uniformly from [min_std, max_std].
            min_std (float, optional): the minimum std in the case stds is None. Defaults to .1.
            max_std (float, optional): the maximum std in the case stds is None. Defaults to .5.
            overlapping (bool, optional): <!> Not used for now. <!>. If False, the clusters will be placed so that each ball B(x, std) is disjoint. Defaults to True.
            pre_visualization (bool, optional): if True, the data points will be plotted. Defaults to False.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: a synthetically generated dataset
        """
        x_data = np.zeros((n_samples, dimension))
        y_data = np.zeros(n_samples)
        
        # Set the seed
        if seed is not None:
            np.random.seed(seed)
            
        # Create the proportion list
        if not isinstance(proportion, list):
            if proportion == 'uniform':
                proportion = [1/n_clusters] * n_clusters
            elif proportion == 'random':
                proportion = np.exp(np.random.normal(size=n_clusters))
                proportion = proportion / proportion.sum()
            else:
                raise ValueError(f"Unknown proportion {proportion}")
        assert len(proportion) == n_clusters, f"proportion must be of length {n_clusters}"
        
        # Create the means
        if not isinstance(means, list):
            if means == 'random':
                means = np.random.normal(size=(n_clusters, dimension))
            else:
                raise ValueError(f"Unknown means {means}")
        
        # Create the stds
        if stds is None:
            stds = np.random.uniform(min_std, max_std, n_clusters)
        elif isinstance(stds, list) and isinstance(stds[0], list): # stds is a list of d-tuple of std
            stds = np.array(stds)
        elif isinstance(stds, list) and isinstance(stds[0], float): # stds is a list of std
            stds = np.array(stds)
        else: # stds is a float
            stds = np.ones(n_clusters) * stds

        # Handle different overlapping types
        if overlapping == 'ellipsoidal':
            # Code for rotated ellipsoidal clusters
            for i in range(n_samples):
                cluster = np.random.choice(n_clusters, p=proportion)
                if isinstance(stds[cluster], float):
                    cov_matrix = np.eye(dimension) * stds[cluster]
                else:
                    cov_matrix = np.diag(stds[cluster])
                x_data[i] = np.random.multivariate_normal(means[cluster], cov_matrix)
                y_data[i] = cluster

        elif overlapping == 'non_gaussian_laplace':
            # Code for Laplace
            for i in range(n_samples):
                cluster = np.random.choice(n_clusters, p=proportion)
                x_data[i] = np.random.laplace(means[cluster], stds[cluster])
                y_data[i] = cluster

        elif overlapping == 'non_gaussian_uniform':
            # Code for uniform on a ball
            for i in range(n_samples):
                cluster = np.random.choice(n_clusters, p=proportion)
                x_data[i] = np.random.uniform(means[cluster] - stds[cluster], means[cluster] + stds[cluster])
                y_data[i] = cluster

        elif overlapping == 'nested':
            # Code for nested clusters
            for i in range(n_samples):
                main_cluster = np.random.choice(n_clusters, p=proportion)
                # Determine if this point belongs to the main cluster or a sub-cluster
                if np.random.rand() < 0.5:  # 50% chance to belong to the main cluster
                    x_data[i] = np.random.normal(means[main_cluster], stds[main_cluster])
                else:  # Belongs to a sub-cluster
                    # Sub-cluster parameters: slightly offset mean and reduced std
                    sub_cluster_mean = means[main_cluster] + np.random.normal(scale=0.1, size=dimension)
                    sub_cluster_std = stds[main_cluster] * 0.5  # Smaller std for sub-cluster
                    x_data[i] = np.random.normal(sub_cluster_mean, sub_cluster_std)
                y_data[i] = main_cluster

        elif overlapping == 'bridging':
            # Assuming two clusters to bridge for simplicity; can be extended for more
            assert n_clusters >= 2, "Cluster bridging requires at least two clusters"

            # Split the dataset into three parts: two clusters and one bridge
            n_cluster_samples = int(n_samples * 0.8) // 2  # 40% of data in each cluster
            n_bridge_samples = n_samples - 2 * n_cluster_samples  # 20% of data in the bridge

            # Generate two main clusters
            for i in range(2):
                cluster_indices = range(i * n_cluster_samples, (i + 1) * n_cluster_samples)
                cluster_mean = means[i] if isinstance(means, list) else np.random.normal(size=(dimension,))
                cluster_std = stds[i] if isinstance(stds, list) else stds
                x_data[cluster_indices] = np.random.normal(cluster_mean, cluster_std, (n_cluster_samples, dimension))
                y_data[cluster_indices] = i

            # Generate the bridging points
            bridge_indices = range(2 * n_cluster_samples, 2 * n_cluster_samples + n_bridge_samples)
            for i, index in enumerate(bridge_indices):
                # Interpolate between the means of the first two clusters
                ratio = i / n_bridge_samples
                bridge_point = ratio * means[0] + (1 - ratio) * means[1]
                bridge_std = min(stds) * 0.5  # Smaller std for bridge points
                x_data[index] = np.random.normal(bridge_point, bridge_std)
                y_data[index] = -1  # Special label for bridge points

        else:
            # Add check that it's 'none' and not something else
            # Default behavior for no specific overlap
            for i in range(n_samples):
                cluster = np.random.choice(n_clusters, p=proportion)
                x_data[i] = np.random.normal(means[cluster], stds[cluster])
                y_data[i] = cluster
            
        # Plot the data points
        if pre_visualization and dimension == 2:
            import matplotlib.pyplot as plt
            plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
            plt.show()
        
        # Return the data points
        return x_data, y_data
    
    
    def get_x_data(self):
        return self.x_data
    
    def get_labels(self):
        return self.y_data