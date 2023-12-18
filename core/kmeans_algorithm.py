


from typing import Dict, List, Union
import numpy as np


class KMeansAlgorithm:
    
    def __init__(self, 
                 n_clusters : int,
                 init : Union[str, np.ndarray],
                 n_init : int,
                 max_iter : int,
                 random_state : int,
                 use_sklearn : bool,
                 ):
        """Initialize a handcrafted KMeans algorithm.

        Args:
            n_clusters (int): the number of clusters
            init (Union[str, np.ndarray]): the initialization method, either "random" or a numpy array of shape (n_clusters, n_features)
            n_init (int): the number of initializations, i.e. the number of times the algorithm is run with different initializations
            max_iter (int): the maximum number of iterations
            random_state (int): the random state
            use_sklearn (bool): whether to use the sklearn implementation or the handcrafted implementation
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_sklearn = use_sklearn
        
        if use_sklearn:
            from sklearn.cluster import KMeans
            self.sklearn_kmeans = KMeans(
                n_clusters=n_clusters, 
                init=init, 
                n_init=n_init, 
                max_iter=max_iter,
                random_state=random_state,
                )
        else:
            pass
        
        
    def fit_predict(self, x_data : np.ndarray) -> np.ndarray:
        """Apply the KMeans algorithm to the data.

        Args:
            x_data (np.ndarray): the data points, of shape (n_data, n_features)

        Returns:
            np.ndarray: the labels of each data point, of shape (n_data,)
        """
        if self.use_sklearn:
            return self.sklearn_kmeans.fit_predict(x_data)
        else:
            raise NotImplementedError()