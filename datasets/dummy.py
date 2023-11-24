


import numpy as np
from datasets.base_dataset import BaseDataset


class DummyDataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        
    def get_x_data(self):
        return np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11,12], [13, 14], [15, 16], [17, 18], [19, 20]])
    
    def get_labels(self):
        return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])