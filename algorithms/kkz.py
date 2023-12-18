from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans
from core.kmeans_algorithm import KMeansAlgorithm

from core.utils import labels_to_clustering_result  


class KKZ_Algorithm(BaseInitForKMeansAlgorithm):
    
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        self.clustering_result = None
    
    
    def get_kkz_init_centroids(self, x_data : np.ndarray) -> np.ndarray:
        """Compute the initial centroids using the KKZ algorithm.

        Args:
            x_data (np.ndarray): the data points, of shape (n_data, n_features)

        Returns:
            np.ndarray: the initial centroids, of shape (n_clusters, n_features)
        """
        # compute the norm of each point
        norms = np.linalg.norm(x_data, axis=1)
        # extract the first centroid being the maximal norm point
        centroids = [np.argmax(norms)]
        distances = np.empty((x_data.shape[0], self.config["k"]))
        # extract the other centroids
        for i in range(1, self.config["k"]):
            # compute the distance between each point and the centroids
            distances[:, i-1] = np.linalg.norm(x_data - x_data[centroids[i-1]], axis=1)
            # compute the minimal distance between each point and the centroids
            min_distances = np.min(distances[:, :i], axis=1)
            # compute the argmax of the minimal distances
            centroids.append(np.argmax(min_distances))
        # cluster the data using the centroids
        centroids = x_data[centroids]
        return centroids
        
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        # Since the KKZ algorithm is deterministic, we can compute the clustering result only once.
        if self.config["compute_only_once"] and self.clustering_result is not None:
            sleep(0.01)
            return self.clustering_result
        
        # Compute the initial centroids using the KKZ algorithm.
        centroids = self.get_kkz_init_centroids(x_data)
        
        # Cluster the original data using the centroids computed by KKZ
        kmeans_algo = KMeansAlgorithm(
            n_clusters=self.config['k'],
            init=centroids,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)