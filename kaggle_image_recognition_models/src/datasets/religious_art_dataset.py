import argparse
import os
from typing import Any, Dict

import numpy as np
import torch

from . import base_dataset


def create_dataset(transform: Any, is_train: bool, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return ReligiousArtDataset(transform, opt.max_dataset_size, opt.img_path, opt.target_path, is_train, opt.train_ratio)


def dataset_modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--img_path', type=str, default=os.path.join('inputs', 'religious_art', 'christ-train-imgs.npz'), help='入力データへのパス')
    parser.add_argument('--target_path', type=str, default=os.path.join('inputs', 'religious_art', 'christ-train-labels.npz'), help='ラベルへのパス, test時は利用しない')
    return parser


class ReligiousArtDataset(base_dataset.BaseDataset):
    """宗教画コンペに利用するtrain-val用データセット
    """
    def __init__(self, transform: Any, max_dataset_size: int, img_path: str, target_path: str, is_train: bool, train_ratio: float) -> None:
        """
        :param transform: 渡すtransformをtrain-val別に指定。
        :param max_dataset_size:
        :param img_path: 画像データのパスを指定。
        :param target_path: targetラベルのパスを指定
        :param is_train: train-valをboolで使い分け
        :param train_ratio: train-valを何%をtrainにするかを指定
        """
        image = np.load(img_path)['arr_0']
        target = np.load(target_path)['arr_0']
        self.transform = transform
        if is_train:
            self.image = image[:int(len(image) * train_ratio)]
            self.target = target[:int(len(target) * train_ratio)]
        else:
            self.image = image[int(len(image) * train_ratio):]
            self.target = target[int(len(target) * train_ratio):]
        super().__init__(max_dataset_size, len(self.image), is_train)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = torch.from_numpy(self.image[idx]).float()
        target = self.target[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {
            'x': image,
            't': torch.tensor(target, dtype=torch.float).long(),
        }


if __name__ == '__main__':
    # python3 -m kaggle_image_recognition_models.src.datasets.religious_art_dataset
    from torchvision.transforms import transforms

    parser = argparse.ArgumentParser()
    parser = dataset_modify_commandline_options(parser)
    parser.add_argument('--max_dataset_size', type=int, default=10**8)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    vanilla_dataset = create_dataset(None, True, args)
    for i in range(len(vanilla_dataset)):
        if i % (len(vanilla_dataset) // 2 + 1) == 0:
            print(vanilla_dataset[i]['x'].shape, vanilla_dataset[i]['t'])

    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    simple_dataset = create_dataset(transform, True, args)
    for i in range(len(simple_dataset)):
        if i % (len(simple_dataset) // 2 + 1) == 0:
            print(simple_dataset[i]['x'].shape, simple_dataset[i]['t'])
