"""
Main Module import all submodules.
"""
#
#
#
__author__ = "Janos Borst"
try:
    from mlmc._version import __version__
except:
    __version__ = "unknown"

import mlmc.data
import mlmc.models
import mlmc.graph
import mlmc.metrics
import mlmc.representation
import mlmc.modules
import mlmc.callbacks
import mlmc.loss
# Save and load models for inference
from .save_and_load import save, load

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import subprocess
import sys
import torch


def install_torch_geometric(cuda=None):
    if cuda is None:
        cuda = torch.version.cuda
    if cuda is None:
        cuda = "cpu"
    else:
        cuda = f"cu{cuda.replace('.','')}"

    version = torch.__version__.replace("+cpu", "")
    pckgs = [
        ["torch-scatter", "-f", f"https://pytorch-geometric.com/whl/torch-{version}+{cuda}.html"],
        ["torch-sparse", "-f", f"https://pytorch-geometric.com/whl/torch-{version}+{cuda}.html"],
        ["torch-cluster", "-f", f"https://pytorch-geometric.com/whl/torch-{version}+{cuda}.html"],
        ["torch-spline-conv", "-f", f"https://pytorch-geometric.com/whl/torch-{version}+{cuda}.html"],
        [f"torch-geometric"]
    ]

    for p in pckgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + p)

