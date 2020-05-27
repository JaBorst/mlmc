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
# Save and load models for inference without the language models representaition to save disc space
from .save_and_load import save, load