from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans, AgglomerativeClustering
from core.kmeans_algorithm import KMeansAlgorithm

from core.utils import labels_to_clustering_result  



class HAC_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        self.clustering_result = None
    
    
    def get_hac_init_centroids(self, x_data : np.ndarray) -> np.ndarray:
        """Compute the initial centroids using the HAC algorithm.

        Args:
            x_data (np.ndarray): the data points, of shape (n_data, n_features)

        Returns:
            np.ndarray: the initial centroids, of shape (n_clusters, n_features)
        """
        # Cluister the data using Hierarchical Agglomerative Clustering
        labels = AgglomerativeClustering(n_clusters=self.config["k"], linkage="ward").fit_predict(x_data)
        # Compute the new cluster centers in the original space
        centroids = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            centroids[i] = np.mean(x_data[labels == i], axis=0)
        return centroids
    
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        # Since the HAC algorithm is deterministic, if the clustering result is already computed, return it.
        if self.config["compute_only_once"] and self.clustering_result is not None:
            sleep(0.01)
            return self.clustering_result
        
        # Compute the initial centroids using the KKZ algorithm.
        centroids = self.get_hac_init_centroids(x_data)
        
        # Cluster the original data using the centroids computed by KKZ
        kmeans_algo = KMeansAlgorithm(
            n_clusters=self.config['k'],
            init=centroids,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)