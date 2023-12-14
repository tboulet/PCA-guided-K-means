from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans

from core.utils import labels_to_clustering_result  


class KKZ_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        self.clustering_result = None
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        if self.clustering_result is not None:
            sleep(0.01)
            return self.clustering_result
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
        labels = KMeans(n_clusters=self.config["k"], init=centroids, n_init=1, max_iter=300).fit_predict(x_data)
        self.clustering_result = labels_to_clustering_result(labels)
        return self.clustering_result




class KKZ_AlgorithmWithoutKmeans(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        self.clustering_result = None
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        if self.clustering_result is not None:
            sleep(0.01)
            return self.clustering_result
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
        distances[:, self.config["k"]-1] = np.linalg.norm(x_data - x_data[centroids[self.config["k"]-1]], axis=1)
        labels = np.argmin(distances, axis=1)
        self.clustering_result = labels_to_clustering_result(labels)
        return self.clustering_result
