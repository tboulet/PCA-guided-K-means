from typing import Dict, List
import numpy as np

from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from core.utils import labels_to_clustering_result

class SK_PCA_GuidedSearchAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        
        # Reduce data dimension using PCA. The reduced dimension is the number of clusters.
        x_data_pca_reduced = PCA(n_components=self.config["k"]).fit_transform(x_data)
        
        # Cluster the reduced data using KMeans.
        n_clusters = self.config["k"]

        kmeans_algo_on_pca_subspace = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=1,
            random_state=np.random.randint(1000),
        )
        labels_from_pca_subspace = kmeans_algo_on_pca_subspace.fit_predict(x_data_pca_reduced)
        
        # Compute the new centroids in the original space
        centroids = np.zeros((self.config["k"], x_data.shape[1]))
        for i in range(self.config["k"]):
            centroids[i] = np.mean(x_data[labels_from_pca_subspace == i], axis=0)
        
        # Cluster the original data using the new cluster centers
        self.kmeans_algo = KMeans(
            n_clusters=n_clusters,
            init=centroids,
            n_init=1,
            random_state=np.random.randint(1000),
        )

        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)

class SK_RandomR1_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        
    def fit(self, x_data):
        random_assignment = np.random.randint(self.config['K_number_of_clusters'], size=x_data.shape[0])
        centroids = np.zeros((self.config['K_number_of_clusters'], x_data.shape[1]))
        for i in range(self.config['K_number_of_clusters']):
            centroids[i] = np.mean(x_data[random_assignment == i], axis=0)

        self.kmeans_algo = KMeans(
            n_clusters=self.config['K_number_of_clusters'],
            init=centroids,
            n_init=1,
            random_state=np.random.randint(1000),
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)

class SK_RandomR2_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        self.kmeans_algo = KMeans(
            n_clusters=self.config['K_number_of_clusters'],
            init="random",
            n_init=1,
            random_state=np.random.randint(1000),
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)    

class SK_KMeansPlusPlusAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
                
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        
        self.kmeans_algo = KMeans(
            n_clusters=self.config['k'],
            init="k-means++",
            n_init=1,
            random_state=np.random.randint(1000),
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)

class SK_HAC_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)    
    
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

        # Compute the initial centroids using the KKZ algorithm.
        centroids = self.get_hac_init_centroids(x_data)
        
        # Cluster the original data using the centroids computed by KKZ
        self.kmeans_algo = KMeans(
            n_clusters=self.config['k'],
            init=centroids,
            n_init=1,
            random_state=np.random.randint(1000),
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)
    
class SK_KKZ_Algorithm(BaseInitForKMeansAlgorithm):
    
    
    def __init__(self, config: dict, kmeans_config: dict):
        super().__init__(config, kmeans_config)
    
    def get_kkz_init_centroids(self, x_data : np.ndarray) -> np.ndarray:
        """Compute the initial centroids using the KKZ algorithm.

        Args:
            x_data (np.ndarray): the data points, of shape (n_data, n_features)

        Returns:
            np.ndarray: the initial centroids, of shape (n_clusters, n_features)
        """
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
        return centroids
        
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        # Compute the initial centroids using the KKZ algorithm.
        centroids = self.get_kkz_init_centroids(x_data)
        
        # Cluster the original data using the centroids computed by KKZ
        self.kmeans_algo = KMeans(
            n_clusters=self.config['k'],
            init=centroids,
            n_init=1,
            random_state=np.random.randint(1000),
        )
        labels = self.kmeans_algo.fit_predict(x_data)
        return labels_to_clustering_result(labels)
