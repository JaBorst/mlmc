"""
Submodule containing implementations of neural network models for Multilabel Architectures.
"""

# Well Tested
from .XMLCNN import XMLCNN
from .KimCNN import KimCNN

#LSAN Variants
from .LSAN import LSANOriginal
from .LSAN_reimplementation import LabelSpecificAttention
from .LSANOriginalTransformer import LSANOriginalTransformer


# Geometric Models
try:
    from .ZAGCNN import ZAGCNN
    from .ZAGCNNLM import ZAGCNNLM
    from .SKG_ML import SKG
except:
    print("pytorch_geometric not installed.")
    pass


from .lm_mogrifier import MogrifierLMCharacter, MogrifierLMWord