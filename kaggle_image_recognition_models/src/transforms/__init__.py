import argparse
from typing import Any, Callable, Dict

from . import affine_transform, no_transform, horizontal_flip_transform, color_transform

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'no': no_transform.create_transform,
    'affine': affine_transform.create_transform,
    'horizontal_flip': horizontal_flip_transform.create_transform,
    'color': color_transform.create_transform
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'no': lambda x: x,
    'affine': affine_transform.transform_modify_commandline_option,
    'horizontal_flip': lambda x: x,
    'color': color_transform.transform_modify_commandline_option,
}
