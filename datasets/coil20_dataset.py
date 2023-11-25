import os
import numpy as np
from datasets.base_dataset import BaseDataset


class Coil20_Dataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print("Loading Coil20 dataset...")
        if not os.path.exists('data/coil20_x.npy'):
            raise FileNotFoundError("Coil20 dataset not found. Please download it by running load_datasets_scripts\load_coil20.py from the root directory.")
        self.x_data, self.y_data = np.load('data/coil20_x.npy', allow_pickle=True), np.load('data/coil20_y.npy', allow_pickle=True)
        print(f"Data shape : {self.x_data.shape}")
        print(f"Target shape : {self.y_data.shape}")
        print("Coil20 dataset loaded.")
        
    def get_x_data(self):
        return self.x_data
    
    def get_labels(self):
        return self.y_data