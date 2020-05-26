"""
Main Module import all submodules.
"""
#
#
# import pkg_resources  # part of setuptools
#
# try:
#     __version__ = pkg_resources.require("mlmc")[0].version
#     __author__ = "Janos Borst"
# except:
#     __version__ = "local"
__version__ = "0.1.0"
__author__ = "Janos Borst"

import mlmc.data
import mlmc.models
import mlmc.graph
import mlmc.metrics
import mlmc.representation
# Save and load models for inference without the language models representaition to save disc space
from .save_and_load import save, load