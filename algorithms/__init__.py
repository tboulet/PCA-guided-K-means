from typing import Dict, Type
from algorithms.base_algorithm import BaseInitForKMeansAlgorithm
from algorithms.random import RandomR1_Algorithm, RandomR2_Algorithm

# This maps the name of the algorithm to the class of the algorithm.
algo_name_to_AlgoClass : Dict[str, Type[BaseInitForKMeansAlgorithm]] = {
    "R1" : RandomR1_Algorithm,
    "R2" : RandomR2_Algorithm,
}
    