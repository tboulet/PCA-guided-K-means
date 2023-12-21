from typing import Dict, List
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from datasets.base_dataset import BaseDataset
from metrics.base_metric import BaseMetric
import numpy as np

class ConfusionMatrixMetric(BaseMetric):
    """This metric computes the confusion matrix for clustering results against true labels.
    Note: This implementation does not align cluster labels with true labels.
    """

    def __init__(self, config: dict):
        self.config = config

    def compute_metrics(self, 
                        dataset: BaseDataset, 
                        clustering_result: Dict[int, List[int]],
                        algo: BaseInitForKMeansAlgorithm,
                        ) -> Dict[str, np.ndarray]:
        true_labels = dataset.get_labels()  # Assuming the dataset has true labels
        predicted_labels = np.empty(len(true_labels), dtype=int)

        # Assign predicted cluster labels based on clustering_result
        for cluster_label, data_indices in clustering_result.items():
            for index in data_indices:
                predicted_labels[index] = cluster_label

        # Manually compute the confusion matrix
        unique_true_labels = np.unique(true_labels)
        unique_predicted_labels = np.unique(predicted_labels)
        cm = np.zeros((len(unique_true_labels), len(unique_predicted_labels)), dtype=int)

        for i, true_label in enumerate(unique_true_labels):
            for j, predicted_label in enumerate(unique_predicted_labels):
                cm[i, j] = np.sum((true_labels == true_label) & (predicted_labels == predicted_label))

        return {"confusion_matrix": cm}
