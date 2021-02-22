import torch

def mean_scaling(x, ind):
    """
    Takes a vector of scores as input and an multi-hot encoded index vector.
    This function will rescale the values of two classes indicated by the index vector such that the mean of of two are equal.
    Args:
        x: input tensor  with batch size b and dimension d(b, d)
        ind: vector of zeroes and ones of dimension (d,)

    Returns:
        Tensor of shape (b,d)
    """
    if (ind==1).all(): return x

    if (ind==0).all():
        ones_mean = 0.5
        zeros_mean = x[:, torch.where(ind == 0)[0]].mean(-1, keepdim=True)
        ratio = ones_mean / zeros_mean
    else:
        ones_mean = x[:, torch.where(ind == 1)[0]].mean(-1, keepdim=True)
        zeros_mean = x[:, torch.where(ind == 0)[0]].mean(-1, keepdim=True)
        ratio = ones_mean / zeros_mean
    return x * ((-ratio * (ind - 1)[None] + ind[None]))
