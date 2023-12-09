import os
import numpy as np
from datasets.base_dataset import BaseDataset
from sklearn.datasets import fetch_openml


class MNISTDataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print("Loading MNIST dataset...")
        if not os.path.exists('data/mnist_data.npy'):
            raise FileNotFoundError("MNIST dataset not found. Please download it by running load_datasets_scripts\load_mnist.py from the root directory.")
        data = np.load('data/mnist_data.npy', allow_pickle=True).item()
        self.x_data, self.y_data = data['images'].to_numpy(), data['labels'].to_numpy()
        self.x_data = self.x_data.astype(np.float32)
        print(f"Data shape : {self.x_data.shape}")
        print(f"Target shape : {self.y_data.shape}")
        print("MNIST dataset loaded.")
            
    def get_x_data(self):
        return self.x_data
    
    def get_labels(self):
        return self.y_data
    
    
    
    
class Cifar10Dataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print("Loading CIFAR-10 dataset...")
        if not os.path.exists('data/cifar10_x.npy') or not os.path.exists('data/cifar10_y.npy'):
            raise FileNotFoundError("CIFAR-10 dataset not found. Please download it by running load_datasets_scripts\load_cifar10.py from the root directory.")
        self.x_data = np.load('data/cifar10_x.npy', allow_pickle=True)
        self.y_data = np.load('data/cifar10_y.npy', allow_pickle=True)
        print("CIFAR-10 dataset loaded.")
        print(f"Data shape : {self.x_data.shape}")
        print(f"Target shape : {self.y_data.shape}")
        print("CIFAR-10 dataset loaded.")
            
    def get_x_data(self):
        return self.x_data
    
    def get_labels(self):
        return self.y_data
    
    

class Cifar100Dataset(BaseDataset):
    
    def __init__(self, config : dict):
        raise NotImplementedError("CIFAR-100 dataset not implemented yet.")
        super().__init__(config)
        print("Loading CIFAR-100 dataset...")
        self.dataset = fetch_openml('CIFAR_100', cache=True)
        print("CIFAR-100 dataset loaded.")
        print(f"Data shape : {self.dataset.data.shape}")
        print(f"Target shape : {self.dataset.target.shape}")
        print("CIFAR-100 dataset loaded.")
            
    def get_x_data(self):
        return self.dataset.data.to_numpy()
    
    def get_labels(self):
        return self.dataset.target.to_numpy()