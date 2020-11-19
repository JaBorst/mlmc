from mlmc.thresholds.thresholds import *
from mlmc.thresholds.scaling import mean_scaling

thresholds_dict = {
    "hard": threshold_hard,
    "max": threshold_max,
    "mcut": threshold_mcut,
    "scaling_mcut": threshold_scaling_mcut,
}

class ThresholdWrapper():
    def __init__(self, name, *args, **kwargs):
        self._fct = thresholds_dict[name]
        self._args = args
        self._kwargs = kwargs

    def __call__(self, x):
        return self._fct(x, *self._args, **self._kwargs)


def get(name: str, *args, **kwargs):
    """
    Instantiate any kind of thresholdind function with hyper parameters
    Args:
        name: name of the thresholding function (max, hard, mcut)
        *args:
        **kwargs:

    Returns:
        A callable threshold Object
    """
    return ThresholdWrapper(name, *args, **kwargs)

#
# get("scaling_mcut", torch.tensor([0,0,1]))(torch.tensor([[0.00001,0.00001,0.9]]))