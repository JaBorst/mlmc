"""
Submodule containing implementations of neural network models for Multilabel Architectures.
"""

# CNN Variants
from .standard import *

#LSAN Variants
# Basic transformers

from .zeroshot import *


# Geometric Models
try:
    from .standard.ZAGCNNLM import ZAGCNNLM
    from .standard.SKGLM import SKGLM
except:
    print("pytorch_geometric not installed.")
    pass


def get(name: str):
    import mlmc.models as mm
    fct = getattr(mm, name.capitalize())
    return fct
