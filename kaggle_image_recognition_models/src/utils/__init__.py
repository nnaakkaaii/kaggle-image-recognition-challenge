import argparse
from typing import Any, Callable, Dict

from . import fix_seed


utils: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'seed': fix_seed.fix_seed
}

utils_optins: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser ]] = {
    'seed': fix_seed.seed_modify_commandline_options
}
