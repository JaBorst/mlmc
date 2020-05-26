"""
Main Module import all submodules.
"""


import pkg_resources  # part of setuptools
#
# __version__ = pkg_resources.require("plenpy")[0].version
# __author__ = "Maximilian Schambach"
#

import mlmc.data
import mlmc.models
import mlmc.graph
import mlmc.metrics
import mlmc.representation
# Save and load models for inference without the language models representaition to save disc space
from .save_and_load import save, load