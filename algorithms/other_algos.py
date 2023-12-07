from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from core.utils import labels_to_clustering_result    

class KMeansPlusPlusAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        return labels_to_clustering_result(KMeans(n_clusters=self.config["k"], init="k-means++", n_init=1, max_iter=300).fit_predict(x_data))
    
    
    
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

class KKZ_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        # compute the norm of each point
        norms = np.linalg.norm(x_data, axis=1)
        # extract the first centroid being the maximal norm point
        centroids = [np.argmax(norms)]
        # extract the other centroids
        for i in range(1, self.config["k"]):
            # compute the distance between each point and the centroids
            distances = np.linalg.norm(x_data - x_data[centroids][:, np.newaxis], axis=2)
            # compute the minimal distance between each point and the centroids
            min_distances = np.min(distances, axis=1)
            # compute the argmax of the minimal distances
            centroids.append(np.argmax(min_distances))
        # cluster the data using the centroids
        distances = np.linalg.norm(x_data - x_data[centroids][:, np.newaxis], axis=2)
        clustering_result = np.argmin(distances, axis=0)
        return labels_to_clustering_result(clustering_result)   
    
class HAC_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        return labels_to_clustering_result(AgglomerativeClustering(n_clusters=self.config["k"], linkage="ward").fit_predict(x_data))

class HAC_Kmeans_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        # cluster the data using HAC
        clustering_result = AgglomerativeClustering(n_clusters=self.config["k"], linkage="ward").fit_predict(x_data)
        # compute the new cluster centers in the original space
        cluster_centers = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            cluster_centers[i] = np.mean(x_data[clustering_result == i], axis=0)
        # cluster the original data using the new cluster centers
        clustering_result = KMeans(n_clusters=self.config["k"], init=cluster_centers, n_init=1, max_iter=300).fit_predict(x_data)
        return labels_to_clustering_result(clustering_result)
    
class KR_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        raise NotImplementedError("KR is not implemented yet.")