# Utils
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns


# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Code
from datasets import dataset_name_to_DatasetClass

method_to_function = {
        "tsne" : lambda x, y : TSNE(n_components=2).fit_transform(x),
        "pca" : lambda x, y : PCA(n_components=2).fit_transform(x),
        "lda" : lambda x, y : LinearDiscriminantAnalysis(n_components=2).fit_transform(x, y),
    }

@hydra.main(config_path="configs", config_name="visualization_config.yaml")
def main(config : DictConfig):

    # Get the config values from the config object.
    config = OmegaConf.to_container(config, resolve=True)
    dataset_name : str = config["dataset"]["name"]
    
    # Get the dataset.
    DatasetClass = dataset_name_to_DatasetClass[dataset_name]
    dataset = DatasetClass(config["dataset"]["config"])
        
    # Get the x dataset
    n_data_max = 1000
    x_data = dataset.get_x_data()[:n_data_max]
    y_data = dataset.get_labels()[:n_data_max]
    
    
    

    # Perform visualization
    if config["method"] not in method_to_function:
        raise ValueError(f"Unknown method {config['method']}. Must be in {list(method_to_function.keys())}")
    x_data_visu = method_to_function[config["method"]](x_data, y_data)
    
    # Create a DataFrame for the t-SNE results
    visu_df = pd.DataFrame(data=x_data_visu, columns=['Component 1', 'Component 2'])
    visu_df['Labels'] = y_data

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Component 1', y='Component 2', hue='Labels', data=visu_df, palette='viridis', legend='full', alpha=0.7)
    plt.title(f'{config["method"]} visualization of dataset {dataset_name}')
    plt.show()


if __name__ == "__main__":
    main()