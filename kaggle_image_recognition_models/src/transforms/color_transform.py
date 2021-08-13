import argparse
from typing import Any

from torchvision.transforms import transforms


def create_transform(opt: argparse.Namespace) -> Any:
    return create_color_transform(
        opt.transform_brightness, opt.transform_contrast, opt.transform_saturation, opt.transform_hue)


def transform_modify_commandline_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--transform_brightness', type=float, default=0.5, help='明るさの変動幅 0以上')
    parser.add_argument('--transform_contrast', type=float, default=0.5, help='コントラストの変動幅 0以上')
    parser.add_argument('--transform_saturation', type=float, default=0.5, help='彩度の変動幅 0以上')
    parser.add_argument('--transform_hue', type=float, default=0.5, help='色相の変動幅 0~0.5')
    return parser


def create_color_transform(brightness: float, contrast: float, saturation: float, hue: float):
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
