from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric
import numpy as np

from sklearn.metrics import davies_bouldin_score

class DaviesBouldinMetric(BaseMetric):
    """Compute the davies bouldin score.
    It measures the average similarity between each cluster and its most similar one.
    Value in [0, +inf]. The lower the better.
    """
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
        return {"davies_bouldin": davies_bouldin_score(x_data, labels)}