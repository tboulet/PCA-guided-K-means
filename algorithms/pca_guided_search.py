from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from core.kmeans_algorithm import KMeansAlgorithm

from core.utils import labels_to_clustering_result    


class PCA_GuidedSearchAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        
        # Reduce data dimension using PCA. The reduced dimension is the number of clusters.
        x_data_pca_reduced = PCA(n_components=self.config["k"]).fit_transform(x_data)
        
        # Cluster the reduced data using KMeans.
        n_clusters = self.config["k"]
        random_assignment_on_pca_subspace = np.random.randint(n_clusters, size=x_data_pca_reduced.shape[0])
        centroids_on_pca_subspace = np.zeros((n_clusters, x_data_pca_reduced.shape[1]))
        for i in range(n_clusters):
            centroids_on_pca_subspace[i] = np.mean(x_data_pca_reduced[random_assignment_on_pca_subspace == i], axis=0)
        kmeans_algo_on_pca_subspace = KMeansAlgorithm(
            n_clusters=n_clusters,
            initial_centroids=centroids_on_pca_subspace,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels_from_pca_subspace = kmeans_algo_on_pca_subspace.fit_predict(x_data_pca_reduced)
        
        # Compute the new centroids in the original space
        centroids = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            centroids[i] = np.mean(x_data[labels_from_pca_subspace == i], axis=0)
        
        # Cluster the original data using the new cluster centers
        self.kmeans_algo = KMeansAlgorithm(
            n_clusters=n_clusters,
            initial_centroids=centroids,
            random_state=np.random.randint(1000),
            **self.kmeans_config,
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)


    
class Normalized_PCA_GuidedSearchAlgorithm(PCA_GuidedSearchAlgorithm):
    """Normalize the data before applying PCA_GuidedSearchAlgorithm."""
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        x_data_normalized = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
        return super().fit(x_data_normalized)

