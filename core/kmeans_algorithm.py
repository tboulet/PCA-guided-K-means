


from typing import Callable, Dict, List, Union
import numpy as np


class KMeansAlgorithm:
    
    def __init__(self, 
                 n_clusters : int,
                 initial_centroids : np.ndarray,
                 n_init : int,
                 max_iter : int,
                 random_state : int,
                 use_sklearn : bool,
                 distance_computation_method : str,
                 centroid_computation_method : str,
                 precompute_distances : bool,
                 ):
        """Initialize our handcrafted KMeans algorithm.

        Args:
            n_clusters (int): the number of clusters
            initial_centroids (np.ndarray): the initialized centroids, a numpy array of shape (n_clusters, n_features)
            n_init (int): the number of initializations, i.e. the number of times the algorithm is run with different initializations
            max_iter (int): the maximum number of iterations
            random_state (int): the random state
            use_sklearn (bool): whether to use the sklearn implementation or the handcrafted implementation
            distance_computation_method (str): the distance metric to use for cluster assignment. Possible values are "L2", "L1", "cosine" or "precompute". Defaults to "L2".
            centroid_computation_method (str): the centroid definition to use for centroid update. Possible values are "mean", "mesoid" (the point in the cluster minimizing inertia) or "median" (the coordinate-wise median).
            precompute_distances (bool, optional): whether to precompute the distances between the data points.
        """
        self.n_clusters = n_clusters
        self.initial_centroids = initial_centroids
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_sklearn = use_sklearn
        self.precompute_distances = precompute_distances
        self.distances_data : np.ndarray = None # (n_data, n_data)
        self.cluster_idx_to_idx_cluster_in_data : np.ndarray = None  # (n_clusters,)
        
        # Sklearn implementation, for comparison
        if use_sklearn:
            from sklearn.cluster import KMeans
            self.sklearn_kmeans = KMeans(
                n_clusters=n_clusters, 
                init=initial_centroids, 
                n_init=n_init, 
                max_iter=max_iter,
                random_state=random_state,
                copy_x=True,  # keep as True, otherwise the data may be overwritten
                )
            return
            
            
        # Dictionary mapping distance_computation_method to distance_function : (n_features,) and (n_features,) -> ()
        distance_computation_method_to_distance_function : Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
            "L2" : lambda data_point_1, data_point_2 : np.linalg.norm(data_point_1 - data_point_2),
            "L1" : lambda data_point_1, data_point_2 : np.sum(np.abs(data_point_1 - data_point_2)),
            "cosine" : lambda data_point_1, data_point_2 : 1 - np.dot(data_point_1, data_point_2) / (np.linalg.norm(data_point_1) * np.linalg.norm(data_point_2)),
        }
        assert distance_computation_method in distance_computation_method_to_distance_function.keys(), f"Invalid value for distance_computation_method: {distance_computation_method}. Possible values are {list(distance_computation_method_to_distance_function.keys())}"
        self.distance_computation_method = distance_computation_method
        self.distance_function = distance_computation_method_to_distance_function[distance_computation_method]
        
        # Dictionary mapping centroid_computation_method to centroid_function : (n_data, n_features) and (n_data,) -> (n_features,)
        centroid_computation_method_to_centroid_function : Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "mean" : lambda x_data, indexes_in_cluster : np.mean(x_data[indexes_in_cluster], axis=0),
            "mesoid" : lambda x_data, indexes_in_cluster : self._compute_mesoid(x_data, indexes_in_cluster),
            "median" : lambda x_data, indexes_in_cluster : np.median(x_data[indexes_in_cluster], axis=0),
        }
        assert centroid_computation_method in centroid_computation_method_to_centroid_function.keys(), f"Invalid value for centroid_computation_method: {centroid_computation_method}. Possible values are {list(centroid_computation_method_to_centroid_function.keys())}"
        self.centroid_computation_method = centroid_computation_method
        self.centroid_function = centroid_computation_method_to_centroid_function[centroid_computation_method]
        
        
    def fit_predict(self, x_data : np.ndarray) -> np.ndarray:
        """Apply the KMeans algorithm to the data.

        Args:
            x_data (np.ndarray): the data points, of shape (n_data, n_features)

        Returns:
            np.ndarray: the labels of each data point, of shape (n_data,)
        """
        
        # Sklearn implementation, for comparison
        if self.use_sklearn:
            return self.sklearn_kmeans.fit_predict(x_data)
        
        
        
        # Our handcrafted implementation
        
        # Set the random state. Our implementation is based on numpy.
        n_data, n_features = x_data.shape
        np.random.seed(self.random_state)
        
        # Precompute the distances between the data points if needed
        if self.precompute_distances and self.distances_data is None:
            self.distances_data = np.zeros((n_data, n_data))
            for idx_data_point_1 in range(n_data):
                for idx_data_point_2 in range(idx_data_point_1 + 1, n_data):
                    dist = self.distance_function(data_point_1 = x_data[idx_data_point_1], data_point_2 = x_data[idx_data_point_2])
                    self.distances_data[idx_data_point_1, idx_data_point_2] = dist
                    self.distances_data[idx_data_point_2, idx_data_point_1] = dist  # we assume the distance function is symmetric
                    
        # Initialize the centroids
        centroids = self.initial_centroids  # (n_clusters, n_features)
        
        # Run the algorithm
        previous_labels = None
        for t in range(self.max_iter):
            
            # Assign each data point to the closest centroid
            distances_to_centroids = np.zeros((self.n_clusters, n_data))
            for cluster_idx in range(self.n_clusters):
                centroid = centroids[cluster_idx]
                for idx_data_point in range(n_data):
                    data_point = x_data[idx_data_point]
                    distances_to_centroids[cluster_idx, idx_data_point] = self.distance_function(
                        data_point_1 = data_point, 
                        data_point_2 = centroid,)
            labels = np.argmin(distances_to_centroids, axis=0)  # (n_data,)
            if np.array_equal(labels, previous_labels): # if the labels have not changed, we have converged
                break
            previous_labels = labels
            
            # Update the centroids
            centroids = np.zeros((self.n_clusters, n_features))
            for cluster_idx in range(self.n_clusters):
                indexes_in_cluster = (labels == cluster_idx).nonzero()[0]
                if indexes_in_cluster.shape[0] == 0:
                    print("WARNING: a cluster was found to have 0 elements. This is likely because of a bad random data partitionning initialization. We will select a random data point as the centroid.")
                    centroids[cluster_idx] = x_data[np.random.choice(n_data, size=1, replace=False)]
                else:
                    centroids[cluster_idx] = self.centroid_function(x_data = x_data, indexes_in_cluster = indexes_in_cluster)
                
        return labels

    
    def _compute_mesoid(self, x_data : np.ndarray, indexes_in_cluster : np.ndarray) -> np.ndarray:
        """Compute the mesoid of a cluster, i.e. the point in the cluster minimizing inertia.

        Args:
            x_data (np.ndarray): the data points, of shape (n_data, n_features)
            indexes_in_cluster (np.ndarray): the indexes of the data points in the cluster, of shape (n_data_in_cluster,)

        Returns:
            np.ndarray: the mesoid, of shape (n_features,)
        """

        best_inertia = np.inf
        best_idx_data_point = None
        # We iterate on the data points in the cluster
        for idx_mesoid_candidate in indexes_in_cluster:
            # We compute the inertia of the cluster if the mesoid was the current data point
            if self.precompute_distances:
                inertia_candidate = np.sum([
                    self.distances_data[idx_mesoid_candidate, idx_data_point_2] 
                    for idx_data_point_2 in indexes_in_cluster
                ])
            else:
                mesoid_candidate = x_data[idx_mesoid_candidate]
                inertia_candidate = np.sum([
                    self.distance_function(
                        data_point_1 = mesoid_candidate, 
                        data_point_2 = x_data[idx_data_point_2]) 
                    for idx_data_point_2 in indexes_in_cluster
                    ])
            # We update the index of the best mesoid if needed
            if inertia_candidate < best_inertia:
                best_inertia = inertia_candidate
                best_idx_data_point = idx_mesoid_candidate
        # We return the best mesoid
        assert best_idx_data_point is not None, "No best mesoid was found. This should not happen."
        return x_data[best_idx_data_point]
    
