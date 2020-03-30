import torch

def threshold_hard(x, tr=0.5):
    """
    Transform input tensor into a multiple-hot tensor using a threshold.

    Args:
        x: Tensor
        tr: Threshold

    Returns:
        Multiple-Hot Tensor
    """
    return (x > tr).int()

def threshold_mcut(x):
    """
    Transform input tensor into a multiple-hot tensor using a threshold

    The threshold is estimated from the largest decay in probability between two successively ranked labels.

    Args:
        x: Input Tensor

    Returns:
        Multiple-Hot Tensor
    """
    x_sorted = torch.sort(x, -1)[0]
    thresholds = (x_sorted[:, 1:] - x_sorted[:, :-1]).max(-1)[0]
    return (x > thresholds[:, None]).float()

def threshold_max(x):
    return torch.zeros_like(x).scatter(1, torch.topk(x, k=1)[1], 1)