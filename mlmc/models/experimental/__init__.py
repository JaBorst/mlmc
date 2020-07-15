try:
    from .SKG_MLLM import SKGLM
    from .ZAGCNNLMAttention import ZAGCNNAttention
    from .SKGLM import SKGLMGGC
    from .SKGLMConv import SKGLMConv
except:
    print("pytorch_geometric not installed.")
    pass
