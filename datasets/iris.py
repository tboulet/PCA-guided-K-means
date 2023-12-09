from datasets.base_dataset import BaseDataset

from sklearn.datasets import load_iris
import numpy as np


class IrisDataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        self.dataset = load_iris()
        print(f"Data shape : {self.dataset.data.shape}")
        print(f"Target shape : {self.dataset.target.shape}")
            
    def get_x_data(self):
        return self.dataset.data.astype(np.float32)
    
    def get_labels(self):
        return self.dataset.target