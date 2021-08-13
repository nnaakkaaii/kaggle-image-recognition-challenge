import argparse
from typing import Any

from torchvision.transforms import transforms


def create_transform(opt: argparse.Namespace) -> Any:
    return create_horizontal_flip_transform()


def create_horizontal_flip_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
