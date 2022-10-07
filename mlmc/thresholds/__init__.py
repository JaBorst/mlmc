from mlmc.thresholds.thresholds import *

thresholds_dict = {
    "hard": threshold_hard,
    "max": threshold_max,
    "mcut": threshold_mcut,
    "mean": threshold_mean,
}

class ThresholdWrapper():
    """
    A Wrapper around thresholding functions
    """
    def __init__(self, name, *args, **kwargs):
        """
        Initializes threshold
        :param name: Name of the threshold (see thresholds_dict.keys())
        :param args: Threshold specific arguments
        :param kwargs: Threshold specific keyword arguments
        """
        self._fct = thresholds_dict[name]
        self._args = args
        self._kwargs = kwargs

    def __call__(self, x):
        """
        Executes initialized threshold
        :param x: Tensor
        :return: Tensor with applied threshold
        """
        return self._fct(x, *self._args, **self._kwargs)


def get(name: str, *args, **kwargs):
    """
    Instantiate any kind of thresholding function with hyperparameters
    Args:
        name: Name of the thresholding function (see thresholds_dict.keys())
        *args: Threshold specific arguments
        **kwargs: Threshold specific keyword arguments

    Returns:
        A callable threshold Object
    """
    return ThresholdWrapper(name, *args, **kwargs)
