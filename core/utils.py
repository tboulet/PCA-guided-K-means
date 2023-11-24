

from typing import Dict, List
import numpy as np


def labels_to_clustering_result(labels : np.ndarray) -> Dict[int, List[int]]:
    """Converts the labels to a clustering result, ie a dictionary mapping cluster index to the list of indices of the points in the cluster.

    Args:
        labels (np.ndarray): the labels of the clustering result.

    Returns:
        Dict[int, List[int]]: the clustering result.
    """
    clustering_result = {}
    for i in range(len(labels)):
        if labels[i] not in clustering_result:
            clustering_result[labels[i]] = []
        clustering_result[labels[i]].append(i)
    return clustering_result