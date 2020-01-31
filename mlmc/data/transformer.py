"""A collection of frequently used tranforming functions for strings or tensors"""

import torch

def clean(x):
    import string
    return "".join([c for c in x if c in string.ascii_letters + string.punctuation + " "])

def label_smoothing(x):
    alpha = 0.1
    K = x.shape[-1]
    return (1 - alpha) * x + alpha / K

def label_smoothing_random(x, alpha=0.1):
    K = x.shape[-1]
    return (1 - alpha) * x + alpha * torch.rand(x.shape)
