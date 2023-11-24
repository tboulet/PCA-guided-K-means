from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans

from core.utils import labels_to_clustering_result



class RandomR1_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data):
        raise NotImplementedError("RandomR1 is not implemented yet.")



class RandomR2_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        print("Running RandomR2...")
        kmeans = KMeans(
            n_clusters=self.config['K_number_of_clusters'], 
            init='random',
            random_state=np.random.randint(1000),
        ).fit(x_data)
        print("RandomR2 finished.")
        labels = kmeans.labels_
        return labels_to_clustering_result(labels)
    
