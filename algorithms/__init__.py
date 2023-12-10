from typing import Dict, Type
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from algorithms.random import RandomR1_Algorithm, RandomR2_Algorithm
from algorithms.other_algos import KMeansPlusPlusAlgorithm, PCA_GuidedSearchAlgorithm, KKZ_Algorithm, HAC_Algorithm, HAC_Kmeans_Algorithm, Normalized_PCA_GuidedSearchAlgorithm

# This maps the name of the algorithm to the class of the algorithm.
algo_name_to_AlgoClass : Dict[str, Type[BaseInitForKMeansAlgorithm]] = {
    "R1" : RandomR1_Algorithm,
    "R2" : RandomR2_Algorithm,
    "KMeans++" : KMeansPlusPlusAlgorithm,
    "PCA_GuidedSearch" : PCA_GuidedSearchAlgorithm,
    "KKZ" : KKZ_Algorithm,
    "HAC": HAC_Algorithm,
    "HAC_Kmeans": HAC_Kmeans_Algorithm,
    "Normalized_PCA_GuidedSearch": Normalized_PCA_GuidedSearchAlgorithm
}
