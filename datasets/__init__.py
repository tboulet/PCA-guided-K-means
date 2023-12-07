from typing import Dict, Type
from datasets.base_dataset import BaseDataset

from datasets.dummy import DummyDataset
from datasets.iris import IrisDataset
from datasets.cv_datasets import MNISTDataset, Cifar10Dataset, Cifar100Dataset
from datasets.AT_and_T_dataset import AT_and_T_Dataset
from datasets.BA_dataset import BinaryAlphabetDataset
from datasets.coil20_dataset import Coil20_Dataset
from datasets.syntethic_dataset import SyntheticDataset

# This maps the name of the dataset to the class of the dataset.
dataset_name_to_DatasetClass : Dict[str, Type[BaseDataset]] = {
    "dummy" : DummyDataset,
    "iris" : IrisDataset,
    "mnist" : MNISTDataset,
    "cifar10" : Cifar10Dataset,
    "cifar100" : Cifar100Dataset,
    "att" : AT_and_T_Dataset,
    "ba" : BinaryAlphabetDataset,
    "coil" : Coil20_Dataset,
    "synth" : SyntheticDataset,
}
