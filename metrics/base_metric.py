


from abc import ABC, abstractmethod
from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm

from datasets.base_dataset import BaseDataset


class BaseMetric(ABC):
        
    @abstractmethod
    def compute_metrics(self, 
                        dataset : BaseDataset, 
                        clustering_result : Dict[int, List[int]],
                        algo : BaseInitForKMeansAlgorithm,
                        ) -> Dict[str, float]:
        """Computes the metric for the given clustering result.

        Args:
            dataset (BaseDataset): the dataset used to get the clustering dataset to cluster on.
            clustering_result (Dict[int, List[int]]): the clustering result.

        Returns:
            Dict[str, float]: a dictionary mapping metric name to the metric value.
        """