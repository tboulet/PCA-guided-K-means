from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans
from core.kmeans_algorithm import KMeansAlgorithm

from core.utils import labels_to_clustering_result



class RandomR1_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        
    def fit(self, x_data):
        random_assignment = np.random.randint(self.config['K_number_of_clusters'], size=x_data.shape[0])
        centroids = np.zeros((self.config['K_number_of_clusters'], x_data.shape[1]))
        for i in range(self.config['K_number_of_clusters']):
            centroids[i] = np.mean(x_data[random_assignment == i], axis=0)
        kmeans_algo = KMeansAlgorithm(
            n_clusters=self.config['K_number_of_clusters'],
            initial_centroids=centroids,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)



class RandomR2_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        centroids = x_data[np.random.choice(x_data.shape[0], self.config['K_number_of_clusters'], replace=False)]
        kmeans_algo = KMeansAlgorithm(
            n_clusters=self.config['K_number_of_clusters'],
            initial_centroids=centroids,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)    
