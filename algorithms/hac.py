from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans, AgglomerativeClustering

from core.utils import labels_to_clustering_result  


    
class HAC_AlgorithmWithoutKMeans(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        self.clustering_result = None
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        if self.clustering_result is not None:
            sleep(0.01)
            return self.clustering_result
        self.clustering_result = labels_to_clustering_result(AgglomerativeClustering(n_clusters=self.config["k"], linkage="ward").fit_predict(x_data))
        return self.clustering_result
    


class HAC_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        self.clustering_result = None
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        if self.clustering_result is not None:
            sleep(0.01)
            return self.clustering_result
        # cluster the data using HAC
        labels = AgglomerativeClustering(n_clusters=self.config["k"], linkage="ward").fit_predict(x_data)
        # compute the new cluster centers in the original space
        cluster_centers = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            cluster_centers[i] = np.mean(x_data[labels == i], axis=0)
        # cluster the original data using the new cluster centers
        labels = KMeans(n_clusters=self.config["k"], init=cluster_centers, n_init=1, max_iter=300).fit_predict(x_data)
        self.clustering_result = labels_to_clustering_result(labels)
        return self.clustering_result