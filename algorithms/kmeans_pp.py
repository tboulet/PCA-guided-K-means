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
        kmeans_algo = KMeansAlgorithm(
            n_clusters=self.config['k'],
            initial_centroids='k-means++',
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)   