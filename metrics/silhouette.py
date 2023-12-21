from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric
import numpy as np

class SilhouetteScoreMetric(BaseMetric):
    """This metric computes the Silhouette Score for clustering.
    The Silhouette Score is a measure of how similar an object is to its own cluster (cohesion) 
    compared to other clusters (separation).
    """

    def __init__(self, config: dict):
        self.config = config

    def compute_metrics(self, 
                        dataset: BaseDataset, 
                        clustering_result: Dict[int, List[int]],
                        algo: BaseInitForKMeansAlgorithm,
                        ) -> Dict[str, float]:
        x_data = dataset.get_x_data()
        silhouette_scores = []

        for cluster_index, cluster_indices in clustering_result.items():
            for point_index in cluster_indices:
                # Calculate 'a': Mean distance to points in the same cluster
                a = np.mean([np.linalg.norm(x_data[point_index] - x_data[other_index]) 
                            for other_index in cluster_indices if other_index != point_index])

                # Calculate 'b': Smallest mean distance to points in other clusters
                b = float('inf')
                for other_cluster_index, other_cluster_indices in clustering_result.items():
                    if other_cluster_index != cluster_index:
                        b = min(b, np.mean([np.linalg.norm(x_data[point_index] - x_data[other_point_index]) 
                                            for other_point_index in other_cluster_indices]))
                
                # Compute the silhouette score for the individual point
                silhouette_score = (b - a) / max(a, b) if max(a, b) > 0 else 0
                silhouette_scores.append(silhouette_score)

        # Average Silhouette Score for all points
        average_silhouette_score = np.mean(silhouette_scores)
        return {"silhouette_score": average_silhouette_score}
