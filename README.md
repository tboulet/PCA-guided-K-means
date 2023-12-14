# PCA-guided K-means
A benchmark on various K-means initialization method, in particular PCA-guided search, as explained in the paper "PCA-guided search for K-means" by Qin Xu et al.

This is a student project by Timothé Boulet, Thomas Lemercier and Théo Saulus.

# Installation

Clone, create a venv and install the requirements :

```bash
git clone https://github.com/tboulet/PCA-guided-K-means.git
cd PCA-guided-K-means
python -m venv venv
source venv/bin/activate # or venv/Scripts/activate on Windows
pip install -r requirements.txt
```


# Run the benchmark

For running the algorithm `algo_name` on the dataset `dataset_name`, run the following command:

```bash
python run.py algo=algo_name dataset=dataset_name
```

For example, to run the random R2 method for initializing K-means on the dataset `mnist`, run the following command:

```bash
python run.py algo=R2 dataset=mnist
```

The results will be logged on WandB. You can modify the code to log to your own WandB account. They will also appear in the console.

### Configuration

We use Hydra as our configuration manager. You can modify the configuration of the benchmark by modifying the `configs/config_default.yaml` file, or you can override the configuration by specifying the corresponding field in the command line. For example :

```bash
python run.py n_iterations=50
```

By default, algorithm R1 will be used on the dataset `iris`.


# Dataset used

The datasets used in the paper are the following:
- AT&T Face
- MNIST
- Binary Alphabet
- Coil20

We intend to reuse these datasets, but also add more datasets to the benchmark.

For downloading the datasets, you can run the following command, you will need to run the corresponding script in the `load_datasets_scripts` folder, as specified below.

Datasets currently available in the code are the following:
- `iris` : The Iris dataset
- `att` : The AT&T Face dataset. To load the dataset, you will need to first download it in Kaggle at https://www.kaggle.com/datasets/kasikrit/att-database-of-faces and place the s1-s40 folders in `./data/at&t_data` and then run ``load_datasets_scripts\load_att.py`` (from the root directory of this repo). This will download the dataset and save it in ``data/`` as an .npy file.
- `mnist` : The MNIST dataset. To load the dataset, you will need to run ``load_datasets_scripts\load_mnist.py``.
- `ba` : The Binary Alphabet dataset. To load the dataset, you will need to run ``load_datasets_scripts\load_binary_alphabet.py``.
- `coil` : The Coil20 dataset. To load the dataset, you will need to run ``load_datasets_scripts\load_coil20.py``.
- `cifar10` : The CIFAR10 dataset. To load the dataset, you will need to run ``load_datasets_scripts\load_cifar10.py``.
- `synth` : A synthetic dataset that will be created on the fly. You can control parameters of this dataset through the config file of the `synth` dataset.


For adding a new dataset, you need to :
- code a class that implements the `datasets.base_dataset.BaseDataset` interface, i.e. that implements the get_x_data() and get_y_data() methods, which returns `numpy.ndarray` objects of shape `(n_samples, n_features)` and `(n_samples,)` respectively.
- add the dataset in the `datasets/__init__.py` file, in the `algo_name_to_AlgoClass` dictionary, with the key being the name of the dataset, and the value being the class you just coded.
- add its config file in `configs/datasets/<dataset_name>.yaml`, specifying the field `name`, `n_classes` and `config`.


# Algorithms implemented

The K-means initialization methods implemented are the following:

- `R1` : Random (R1 version), we partition randomly the dataset into `k` clusters, and take the centroids of these clusters as the initial centroids.
- `R2` : Random (R2 version), we sample `k` points uniformly at random from the dataset, they are the initial centroids.
- `KMeansPP` : K-means++ algorithm. The
first centroid is chosen randomly, and subsequent centroids are
selected with a probability proportional to the square of the shortest
distance from a data point to the closest centroid.
- `PCA_GuidedSearch` : PCA-guided search, as described in the paper "PCA-guided search for K-means" by Qin Xu et al. The algorithm is the following:
    - We first compute the PCA of the dataset, and project the dataset on the first `k` principal components.
    - We then run the K-means++ algorithm on the projected dataset.
    - We obtain our clusters according to our projected dataset. We finally apply the K-means algorithm on the original dataset, with the centroids obtained from those clusters as initial centroids.
- `HAC` : Hierarchical Agglomerative Clustering. The HAC algorithm is applied on the dataset : employing a "bottom-
up" approach, begins with numerous small clusters and iteratively
merges the closest clusters until a single cluster is formed. The centroids of the `k` clusters obtained are taken as the initial centroids of an additional K-means algorithm.
- `KKZ` : The KKZ algorithm initiates by selecting the first centroid with the maximum norm. Successive centroids are chosen based on the maxi-
mal distance from previously selected centroids to candidate points,
iteratively repeating this process until K centroids are identified

TODO (Thomas) : add more algorithms.

For adding a new algorithm, you need to :
- code a class that implements the `algorithms.base_algorithm.BaseInitForKMeansAlgorithm` interface, i.e. that implements the `fit(x_data : np.nd_array)` method, which returns a dictionnary mapping cluster index to list of indices of datapoints.
- add the algorithm in the `algorithms/__init__.py` file, in the `algo_name_to_AlgoClass` dictionary, with the key being the name of the algorithm, and the value being the class you just coded.
- add its config file in `configs/algorithms/<algo_name>.yaml`, specifying the field `name` and `config`.

# Metrics

The metrics used to measure the quality of the initialization are the following:
- `distortion` : The distortion of the clustering, i.e. the sum of the squared distances between each point and its centroid.
- `best distortion` : The best distortion obtained so far, i.e. since the beginning of the algorithm running. This is used for stochastic algorithms, that may give different results at each run.

TODO (Théo) : eventually add other metrics, (including the Pareto front ?)

For adding a new metric, you need to :
- code a class that implements the `metrics.base_metric.BaseMetric` interface, i.e. that implements the `compute_metrics(dataset : BaseDataset, clustering_result : Dict[int, List[int]])` method, which return a dictionary mapping metric name to metric value.
- add the metric in the `metrics/__init__.py` file, in the `metric_name_to_MetricClass` dictionary, with the key being the name of the metric, and the value being the class you just coded.
- add its config in the `yaml` file used for global configuration.