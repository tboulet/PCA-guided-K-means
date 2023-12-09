import os
import numpy as np
from datasets.base_dataset import BaseDataset


class BinaryAlphabetDataset(BaseDataset):
    
    def __init__(self, config : dict):
        super().__init__(config)
        print("Loading the Binary Alphabet dataset...")
        if not os.path.exists('data/ba_x.npy'):
            raise FileNotFoundError("Binary Alphabet dataset not found. Please download it by running load_datasets_scripts\load_binary_alphabet.py from the root directory.")
        self.x_data, self.y_data = np.load('data/ba_x.npy', allow_pickle=True), np.load('data/ba_y.npy', allow_pickle=True)
        
        # The BA dataset consist of the 10 digits from 0 to 9 and the 26 letters from A to Z, in that order. There is 39 examples for each character and each example is a 20 x 16 pixels image.
        # We can keep only the numbers by removing the letters.
        if not config["keep_alphabets"] and not config["keep_digits"]:
            raise ValueError("Both keep_alphabets and keep_digits are False. At least one of them must be True.")
        elif not config["keep_alphabets"]:
            self.x_data = self.x_data[:10 * 39]
            self.y_data = self.y_data[:10 * 39]
        elif not config["keep_digits"]:
            self.x_data = self.x_data[10 * 39:]
            self.y_data = self.y_data[10 * 39:]
        else:
            pass
        self.x_data = self.x_data.astype(np.float32)
        print(f"Data shape : {self.x_data.shape}")
        print(f"Target shape : {self.y_data.shape}")
        print("Binary Alphabet dataset loaded.")
        
    def get_x_data(self):
        return self.x_data
    
    def get_labels(self):
        return self.y_data