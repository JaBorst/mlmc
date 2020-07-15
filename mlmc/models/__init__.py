"""
Submodule containing implementations of neural network models for Multilabel Architectures.
"""

# Well Tested
from .XMLCNN import XMLCNN
from .KimCNN import KimCNN
from .MoKimCNN import MoKimCNN

#LSAN Variants
from .LSAN import LSANOriginal
from .LSANOriginalTransformer import LSANOriginalTransformer

# Geometric Models
try:
    from mlmc.models.experimental.ZAGCNNLMAttention import ZAGCNNAttention
    from .ZAGCNN import ZAGCNN
    from .MoZAGCNNLM import MoZAGCNNLM
    from .MoTransformer import MoTransformer
    from .ZAGCNNLM import ZAGCNNLM
    from .SKG_ML import SKG
except:
    print("pytorch_geometric not installed.")
    pass

from .LSANOriginalTransformerNoClasses import LSANOriginalTransformerNoClasses
from .lm_mogrifier import MogrifierLMCharacter, MogrifierLMWord


def finetune_mixed_precision_model(model, finetune=True):
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