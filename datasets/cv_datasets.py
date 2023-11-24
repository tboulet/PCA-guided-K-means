import os
import numpy as np
from datasets.base_dataset import BaseDataset
from sklearn.datasets import fetch_openml


class MNISTDataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        if not os.path.exists('data/mnist_data.npy'):
            raise FileNotFoundError("MNIST dataset not found. Please download it by running load_datasets_scripts\load_mnist.py from the root directory.")
        data = np.load('data/mnist_data.npy', allow_pickle=True).item()
        self.x_data, self.y_data = data['images'].to_numpy(), data['labels'].to_numpy()
        print(f"Data shape : {self.x_data.shape}")
        print(f"Target shape : {self.y_data.shape}")
            
    def get_x_data(self):
        return self.x_data
    
    def get_labels(self):
        return self.y_data
    
    
    
    
class Cifar10Dataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print("Loading CIFAR-10 dataset...")
        self.dataset = fetch_openml('CIFAR_10', cache=True)
        print("CIFAR-10 dataset loaded.")
        print(f"Data shape : {self.dataset.data.shape}")
        print(f"Target shape : {self.dataset.target.shape}")
            
    def get_x_data(self):
        return self.dataset.data
    
    def get_labels(self):
        return self.dataset.target
    
    

class Cifar100Dataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print("Loading CIFAR-100 dataset...")
        self.dataset = fetch_openml('CIFAR_100', cache=True)
        print("CIFAR-100 dataset loaded.")
        print(f"Data shape : {self.dataset.data.shape}")
        print(f"Target shape : {self.dataset.target.shape}")
            
    def get_x_data(self):
        return self.dataset.data.to_numpy()
    
    def get_labels(self):
        return self.dataset.target.to_numpy()