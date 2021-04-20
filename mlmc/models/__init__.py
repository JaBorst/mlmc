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

from .Encoder import SimpleEncoder

# Geometric Models
try:
    from .ZAGCNNLM import ZAGCNNLM
    from .SKGLM import SKGLM
except:
    print("pytorch_geometric not installed.")
    pass

def finetune_mixed_precision_model(model, finetune=True):
    """
    Sets a model to use FP16 where appropriate to save memory and speed up training.

    :param model: A model instance
    :return: A model with initialized Automatic Mixed Precision
    """
    try:
        from apex import amp
        model.use_amp=True
        opt = model.optimizer.__class__(filter(lambda p: p.requires_grad, model.parameters()), **model.optimizer_params)
        model, opt = amp.initialize(model, opt,opt_level="O2",
   keep_batchnorm_fp32=True, loss_scale="dynamic")
        model.optimizer = opt
    except ModuleNotFoundError:
        model.use_amp = False
    return model
