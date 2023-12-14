from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from core.utils import labels_to_clustering_result    


class PCA_GuidedSearchAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        # reduce data dimension using PCA the reduced dimension is the number of clusters
        PCA_reduced_data = PCA(n_components=self.config["k"]).fit_transform(x_data)
        # cluster the reduced data using kmeans
        kmeans_result = KMeans(n_clusters=self.config["k"], init="random", n_init=1, max_iter=300).fit_predict(PCA_reduced_data)
        # compute the new cluster centers in the original space
        cluster_centers = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            cluster_centers[i] = np.mean(x_data[kmeans_result == i], axis=0)
        # cluster the original data using the new cluster centers
        clustering_result = KMeans(n_clusters=self.config["k"], init=cluster_centers, n_init=1, max_iter=300).fit_predict(x_data)
        return labels_to_clustering_result(clustering_result)


    
class Normalized_PCA_GuidedSearchAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        x_data = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
        # reduce data dimension using PCA the reduced dimension is the number of clusters
        PCA_reduced_data = PCA(n_components=self.config["k"]).fit_transform(x_data)
        # cluster the reduced data using kmeans
        kmeans_result = KMeans(n_clusters=self.config["k"], init="random", n_init=1, max_iter=300).fit_predict(PCA_reduced_data)
        # compute the new cluster centers in the original space
        cluster_centers = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            cluster_centers[i] = np.mean(x_data[kmeans_result == i], axis=0)
        # cluster the original data using the new cluster centers
        clustering_result = KMeans(n_clusters=self.config["k"], init=cluster_centers, n_init=1, max_iter=300).fit_predict(x_data)
        return labels_to_clustering_result(clustering_result)

