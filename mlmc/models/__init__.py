"""
Submodule containing implementations of neural network models for Multilabel Architectures.
"""

# CNN Variants
from .XMLCNN import XMLCNN
from .KimCNN import KimCNN
from .MoKimCNN import MoKimCNN

#LSAN Variants
from .LSAN import LSAN
from .LSANNC import LSANNC
from .MoLSANNC import MoLSANNC
# Basic transformers
from .MoTransformer import MoTransformer
from .Transformer import Transformer

from .bayes_model import  BayesNetwork

from .zeroshot import *


# Geometric Models
try:
    from .ZAGCNNLM import ZAGCNNLM
    from .SKGLM import SKGLM
except:
    print("pytorch_geometric not installed.")
    pass


def get(name: str):
    import mlmc.models as mm
    fct = getattr(mm, name.capitalize())
    return fct
