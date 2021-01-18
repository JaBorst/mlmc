import torch
from ..thresholds.scaling import mean_scaling

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
    thresholds = (x_sorted[:, 1:] - x_sorted[:, :-1]).max(-1)
    values = torch.gather(x_sorted, 1,thresholds[1][:,None])
    thresholds = values + 0.5 * thresholds[0][:,None]
    return (x > thresholds).float()


def threshold_mean(x):
    """
    Transform input tensor into a multiple-hot tensor using a threshold

    The threshold is calculated by taking the mean of the input tensor.

    Args:
        x: Input Tensor

    Returns:
        Multiple-Hot Tensor
    """
    return (x > x.mean(-1).unsqueeze(-1)).float()

def threshold_scaling_mcut(x, ind):
    """
    Transform input tensor into a multiple-hot tensor using a threshold

    The threshold is estimated from the largest decay in probability between two successively ranked labels.
    Before applying the threshold the mean of the values indicated by 0 and 1 is scaled to be the same.

    Args:
        x: Input Tensor
        ind: Index Tensor

    Returns:
        Multiple-Hot Tensor
    """
    x = mean_scaling(x,ind)
    return threshold_mcut(x)


def threshold_max(x):
    """
    Transform input tensor into a one-hot tensor using a threshold

    The threshold chooses the maximum value in the input tensor.

    Args:
        x: Input Tensor

    Returns:
        One-Hot Tensor
    """
    return torch.zeros_like(x).scatter(1, torch.topk(x, k=1)[1], 1)






