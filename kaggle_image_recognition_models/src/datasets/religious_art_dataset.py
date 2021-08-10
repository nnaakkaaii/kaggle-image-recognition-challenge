import argparse
import os
from typing import Any, Dict

import numpy as np
import torch
# from torchvision.datasets import MNIST

from . import base_dataset


def create_train_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return TrainDataset(transform, opt.max_dataset_size, opt.img_dir, opt.target_dir, is_train)


def create_test_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return TestDataset(transform, opt.max_dataset_size, opt.img_dir, is_train)


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--img_dir', type=str, default=os.path.join('inputs', 'mnist'), help='mnistデータを保存する場所')
    return parser


# class MnistDataset(base_dataset.BaseDataset):
#     """torchvisionのMNISTを利用したデータセット
#     MNISTのDatasetの並び順が変更しないというメタ知識を前提としている
#     渡すtransformも変更する
#     """
#     def __init__(self, transform: Any, max_dataset_size: int, img_dir: str, is_train: bool, train_ratio: float) -> None:
#         dataset = MNIST(img_dir, download=True, transform=transform)
#         if is_train:
#             self.dataset = dataset[:int(len(dataset) * train_ratio)]
#         else:
#             self.dataset = dataset[int(len(dataset) * train_ratio):]
#
#         super().__init__(max_dataset_size, len(self.dataset), is_train)
#
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         x, t = self.dataset[idx]
#         return {'x': x, 't': t}


class TrainDataset(base_dataset.BaseDataset):
    def __init__(self, transform: Any, max_dataset_size: int, img_dir: str, target_dir: str, is_train: bool) -> None:
        self.image = np.load(img_dir)['arr_0']
        self.target = np.load(target_dir)['arr_0']
        self.transform = transform
        super().__init__(max_dataset_size, len(self.image), is_train)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.image[idx]
        target = self.target[idx]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = image[np.newaxis, :, :]
        return {
            'x': torch.tensor(image, dtype=torch.float),
            't': torch.tensor(target, dtype=torch.float).long(),
        }


class TestDataset(base_dataset.BaseDataset):
    # train_ratioはcross validationに任せるため消去
    def __init__(self, transform: Any, max_dataset_size: int, test_dir: str, is_train: bool) -> None:
        self.image = np.load(test_dir)['arr_0']
        self.transform = transform
        super().__init__(max_dataset_size, len(self.image), is_train)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.image[idx]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = image[np.newaxis, :, :]
        return {
            'x': torch.tensor(image, dtype=torch.float),
        }


