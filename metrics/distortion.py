from typing import Dict, List
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric


class DistortionMetric(BaseMetric):
    """This is the reference metric for clustering. It is the sum of the squared distances from each point to its cluster center.
    The lower the distortion, the better the clustering.
    
    This object log the distortion of the clustering, but also the best distortion so far.
    """
    def __init__(self, config : dict):
        self.config = config
        self.best_distortion = float('inf')
    
    def compute_metrics(self, dataset : BaseDataset, clustering_result : Dict[int, List[int]]) -> Dict[str, float]:
        distortion = 0
        for cluster_index, cluster_indices in clustering_result.items():
            cluster_center = dataset.get_x_data()[cluster_indices].mean(axis=0)
            for point_index in cluster_indices:
                point = dataset.get_x_data()[point_index]
                distortion += ((point - cluster_center) ** 2).sum()
        self.best_distortion = min(self.best_distortion, distortion)
        return {"distortion" : distortion, "best_distortion" : self.best_distortion}