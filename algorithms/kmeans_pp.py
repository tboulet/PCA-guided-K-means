from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans
from core.kmeans_algorithm import KMeansAlgorithm

from core.utils import labels_to_clustering_result  




class KMeansPlusPlusAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
                
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        
        # Initialize centroids using the kmeans++ algorithm
        n_clusters = self.config['k']
        n_data, n_features = x_data.shape
        
        centroids = [x_data[np.random.randint(n_data)]] # Pick a random point as the first centroid
        
        for cluster_idx in range(1, n_clusters):
            # For each point, compute the distance to the nearest centroid
            distances_to_centroids = np.array([np.min(np.linalg.norm(x_data - c, axis=1)) for c in centroids])  # (n_centroids,)
            
            # Pick the point with the largest distance to the nearest centroid as the next centroid
            next_centroid_idx = np.argmax(distances_to_centroids)
            next_centroid = x_data[next_centroid_idx]
            centroids.append(next_centroid)
            
        centroids = np.array(centroids)
        
        self.kmeans_algo = KMeansAlgorithm(
            n_clusters=self.config['k'],
            initial_centroids=centroids,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)   
    