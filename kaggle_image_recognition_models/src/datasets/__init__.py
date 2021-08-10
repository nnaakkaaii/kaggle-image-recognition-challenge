import argparse
from typing import Any, Callable, Dict

from .base_dataset import BaseDataset
# from . import mnist_dataset
#
# datasets: Dict[str, Callable[[Any, bool, argparse.Namespace], BaseDataset]] = {
#     'mnist': mnist_dataset.create_dataset,
# }
#
# dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
#     'mnist': mnist_dataset.dataset_modify_commandline_options,
# }

from . import religious_art_dataset

datasets: Dict[str, Callable[[Any, bool, argparse.Namespace], BaseDataset]] = {
    'train_dataset': religious_art_dataset.create_train_dataset,
    'test_dataset': religious_art_dataset.create_test_dataset,

}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'train_dataset': religious_art_dataset.dataset_modify_commandline_options,
    'test_dataset': religious_art_dataset.dataset_modify_commandline_options,
}
