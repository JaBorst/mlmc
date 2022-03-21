"""
Submodule containing implementations of neural network models for Multilabel Architectures.
"""

# CNN Variants
from mlmc.models.standard.KimCNN import KimCNN

#LSAN Variants
# Basic transformers

from mlmc.models.abstracts.bayes_model import  BayesNetwork

from .zeroshot import *


# Geometric Models
try:
    from mlmc.models.standard.ZAGCNNLM import ZAGCNNLM
    from mlmc.models.standard.SKGLM import SKGLM
except:
    print("pytorch_geometric not installed.")
    pass


def get(name: str):
    import mlmc.models as mm
    fct = getattr(mm, name.capitalize())
    return fct
