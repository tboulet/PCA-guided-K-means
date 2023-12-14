from typing import Dict, Type
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm

from algorithms.random import RandomR1_Algorithm, RandomR2_Algorithm
from algorithms.pca_guided_search import PCA_GuidedSearchAlgorithm, Normalized_PCA_GuidedSearchAlgorithm
from algorithms.kkz import KKZ_Algorithm
from algorithms.hac import HAC_Algorithm
from algorithms.kmeans_pp import KMeansPlusPlusAlgorithm



# This maps the name of the algorithm to the class of the algorithm.
algo_name_to_AlgoClass : Dict[str, Type[BaseInitForKMeansAlgorithm]] = {
    "R1" : RandomR1_Algorithm,
    "R2" : RandomR2_Algorithm,
    "KMeansPP" : KMeansPlusPlusAlgorithm,
    "PCA_GuidedSearch" : PCA_GuidedSearchAlgorithm,
    "KKZ" : KKZ_Algorithm,
    "HAC": HAC_Algorithm,    
    
    
    
    "Normalized_PCA_GuidedSearch": Normalized_PCA_GuidedSearchAlgorithm, # TODO(Thomas): include normalization rather as an algo-agnostic preprocessing step, parameterized by a boolean flag in the config
}
