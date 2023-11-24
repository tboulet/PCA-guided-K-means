import numpy as np
from sklearn.datasets import fetch_openml
import os

# Function to download MNIST dataset and save it as a NumPy array
def download_and_save_mnist():
    
    # Load MNIST dataset using scikit-learn
    save_path = os.path.join('data', 'mnist_data.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mnist = fetch_openml('mnist_784')
    x_data, y_data = mnist.data.astype(np.uint8), mnist.target.astype(np.uint8)

    # Save the data as a NumPy array
    np.save(save_path, {'images': x_data, 'labels': y_data})

    print(f'MNIST dataset saved as {save_path}')
    
    
if __name__ == "__main__":
    download_and_save_mnist()