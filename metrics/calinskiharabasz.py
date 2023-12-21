from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric
import numpy as np

from sklearn.metrics import calinski_harabasz_score

class CalinskiHarabaszMetric(BaseMetric):
    def __init__(self, config: dict):
        self.config = config
    
    def compute_metrics(self,
                        dataset: BaseDataset,
                        clustering_result: Dict[int, List[int]],
                        algo: BaseInitForKMeansAlgorithm,
                        ) -> Dict[str, float]:
        x_data = dataset.get_x_data()
        labels = np.zeros(len(x_data))
        for cluster_index, cluster_indices in clustering_result.items():
            labels[cluster_indices] = cluster_index
        return {"calinski_harabasz": calinski_harabasz_score(x_data, labels)}

# class CalinskiHarabaszMetric(BaseMetric):
#     """
#     This metric computes the Calinski-Harabasz Index for clustering results.
#     A higher Calinski-Harabasz score indicates better clustering.
#     """

#     def __init__(self, config: dict):
#         self.config = config

#     def compute_metrics(self, 
#                         dataset: BaseDataset, 
#                         clustering_result: Dict[int, List[int]],
#                         algo: BaseInitForKMeansAlgorithm,
#                         ) -> Dict[str, float]:
#         x_data = dataset.get_x_data()
#         N = x_data.shape[0]
#         k = len(clustering_result)

#         overall_mean = np.mean(x_data, axis=0)

#         # Compute within-cluster dispersion W_k
#         W_k = np.sum([np.sum((x_data[indices] - np.mean(x_data[indices], axis=0))**2) 
#                       for indices in clustering_result.values()])

#         # Compute between-cluster dispersion B_k
#         B_k = np.sum([len(indices) * np.sum((np.mean(x_data[indices], axis=0) - overall_mean)**2) 
#                       for indices in clustering_result.values()])

#         # Calinski-Harabasz Index
#         ch_index = (B_k / (k - 1)) / (W_k / (N - k))
#         return {"calinski_harabasz_index": ch_index}
