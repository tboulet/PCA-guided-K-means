from time import sleep
from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans

from core.utils import labels_to_clustering_result  




class KMeansPlusPlusAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        return labels_to_clustering_result(KMeans(n_clusters=self.config["k"], init="k-means++", n_init=1, max_iter=300).fit_predict(x_data))
