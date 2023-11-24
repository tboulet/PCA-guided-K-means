from typing import Dict, List

import numpy as np
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from sklearn.cluster import KMeans

from core.utils import labels_to_clustering_result


    

class KMeansPlusPlusAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        raise NotImplementedError("KMeansPlusPlus is not implemented yet.")
    
    
    
class PCA_GuidedSearchAlgorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        raise NotImplementedError("PCAGuidedSearch is not implemented yet.")
    
    

class KKZ_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        raise NotImplementedError("KKZ is not implemented yet.")
    
    
    
class HAC_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        raise NotImplementedError("HAC is not implemented yet.")
    
    
    
class KR_Algorithm(BaseInitForKMeansAlgorithm):
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(self, x_data : np.ndarray) -> Dict[int, List[int]]:
        raise NotImplementedError("KR is not implemented yet.")