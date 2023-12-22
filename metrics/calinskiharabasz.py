from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric
import numpy as np

from sklearn.metrics import calinski_harabasz_score

class CalinskiHarabaszMetric(BaseMetric):
    """Compute the calinski harabasz score.
    This metric is a ratio between the within-cluster dispersion and the between-cluster dispersion.
    It traduces how well separated and dense the clusters are.
    Value in [0, +inf]. The higher the better.
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
        return {"calinski_harabasz": calinski_harabasz_score(x_data, labels)}
