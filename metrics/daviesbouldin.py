from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric
import numpy as np

class DaviesBouldinMetric(BaseMetric):
    """
    This metric computes the Davies-Bouldin Index for clustering results.
    A lower Davies-Bouldin index indicates a better clustering.
    """

    def __init__(self, config: dict):
        self.config = config

    def compute_metrics(self, 
                        dataset: BaseDataset, 
                        clustering_result: Dict[int, List[int]],
                        algo: BaseInitForKMeansAlgorithm,
                        ) -> Dict[str, float]:
        x_data = dataset.get_x_data()

        # Compute centroids and spreads for each cluster
        centroids = {}
        spreads = {}
        for cluster_index, cluster_indices in clustering_result.items():
            cluster_points = x_data[cluster_indices]
            centroid = cluster_points.mean(axis=0)
            spread = np.mean([np.linalg.norm(point - centroid) for point in cluster_points])
            centroids[cluster_index] = centroid
            spreads[cluster_index] = spread

        # Compute similarity measure for each cluster pair and find the maximum similarity for each cluster
        max_similarities = []
        for cluster_index in clustering_result:
            max_similarity = 0
            for other_cluster_index in clustering_result:
                if cluster_index != other_cluster_index:
                    centroid_distance = np.linalg.norm(centroids[cluster_index] - centroids[other_cluster_index])
                    similarity = (spreads[cluster_index] + spreads[other_cluster_index]) / centroid_distance
                    max_similarity = max(max_similarity, similarity)
            max_similarities.append(max_similarity)

        # Compute Davies-Bouldin index
        db_index = np.mean(max_similarities)
        return {"davies_bouldin_index": db_index}
