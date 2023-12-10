from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans

from core.utils import labels_to_clustering_result



class RandomR1_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data):
        random_assignment = np.random.randint(self.config['K_number_of_clusters'], size=x_data.shape[0])
        centroids = np.zeros((self.config['K_number_of_clusters'], x_data.shape[1]))
        for i in range(self.config['K_number_of_clusters']):
            centroids[i] = np.mean(x_data[random_assignment == i], axis=0)
        kmeans = KMeans(
            n_clusters=self.config['K_number_of_clusters'], 
            init=centroids,
            n_init=1,
            max_iter=300,
        ).fit_predict(x_data)
        return labels_to_clustering_result(kmeans)

class RandomR2_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        print("Running RandomR2...")
        kmeans = KMeans(
            n_clusters=self.config['K_number_of_clusters'], 
            init='random',
            n_init=1,
            random_state=np.random.randint(1000),
        ).fit(x_data)
        print("RandomR2 finished.")
        labels = kmeans.labels_
        return labels_to_clustering_result(labels)
    
