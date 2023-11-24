from typing import Dict, Type
from datasets.base_dataset import BaseDataset
from datasets.dummy import DummyDataset
from datasets.iris import IrisDataset
from datasets.cv_datasets import MNISTDataset, Cifar10Dataset, Cifar100Dataset


# This maps the name of the dataset to the class of the dataset.
dataset_name_to_DatasetClass : Dict[str, Type[BaseDataset]] = {
    "dummy" : DummyDataset,
    "iris" : IrisDataset,
    "mnist" : MNISTDataset,
    "cifar10" : Cifar10Dataset,
    "cifar100" : Cifar100Dataset,
}
