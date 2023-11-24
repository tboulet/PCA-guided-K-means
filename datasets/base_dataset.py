from abc import ABC, abstractmethod
from typing import Union
import numpy as np



class BaseDataset(ABC):
    def __init__(self, config) -> None:
        self.config = config
    
    @abstractmethod
    def get_x_data(self) -> np.ndarray:
        """Get the x matrix of the dataset to cluster on.
        
        Returns:
            np.ndarray: the x matrix of the dataset to cluster on, of shape (n_samples, n_features).
        """
        
    @abstractmethod
    def get_labels(self) -> Union[np.ndarray, None]:
        """Get the labels of the dataset to cluster on, or None if the dataset is unsupervised.
        
        Returns:
            np.ndarray: the dataset to cluster on, of shape (n_samples,) or None.
        """