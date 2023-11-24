from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from datasets.base_dataset import BaseDataset


class BaseInitForKMeansAlgorithm(ABC):
    """An interface for any initialization algorithm method for k-means."""
    def __init__(self, config : dict):
        self.config = config
        
    @abstractmethod
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        """Clusters the x data into k clusters.

        Args:
            x_data (np.ndarray): the dataset used to get the clustering dataset to cluster on.

        Returns:
            Dict[int, List[int]]: a dictionary mapping cluster index to the list of indices of the points in the cluster.
            This fully represent the clustering result.
        """