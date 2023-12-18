from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric


class NumberOfIterationsForConvergenceMetric(BaseMetric):
    """Computes the number of iterations for convergence of the algorithm.
    """
    def __init__(self, config : dict):
        self.config = config
        self.has_failed = False
    
    def compute_metrics(self, 
                        dataset : BaseDataset, 
                        clustering_result : Dict[int, List[int]],
                        algo : BaseInitForKMeansAlgorithm,
                        ) -> Dict[str, float]:
        if self.has_failed:
            return {}
        try:
            return {"number_of_iterations_for_convergence" : algo.kmeans_algo.iteration}
        except:
            self.has_failed = True
            return {}