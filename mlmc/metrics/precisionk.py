from ignite.metrics import Precision, Accuracy
import torch


class PrecisionK(Precision):
    """
    Implements Precision@k metric using ignite's precision metric.
    """
    def __init__(self, k, *args, **kwargs):
        self.k = k
        super(PrecisionK, self).__init__(*args, **kwargs)

    def update(self, output):
        """
        Adds classification output to class for computation of metric.

        :param output: output of classification task in form (scores, truth, pred)
        """
        transformed = torch.zeros_like(output[0]).scatter(1, torch.topk(output[0], k=self.k)[1], 1)
        super(PrecisionK, self).update((transformed.int(), output[1].int()))

    def print(self):
        """
        Computes metric.

        :return: Precision@k score
        """
        return self.compute()


class AccuracyTreshold(Accuracy):
    """
    Class for calculating a threshold regarding accuracy. Uses ignite's accuracy metric.
    """
    def __init__(self, trf, args_dict=None, *args, **kwargs):
        if args_dict is None:
            args_dict = {}
        self.trf = trf
        self.args_dict = args_dict
        super(AccuracyTreshold, self).__init__(*args, **kwargs)

    def update(self, output):
        """
        Adds classification output to class for computation of metric.

        :param output: output of classification task in form (scores, truth, pred)
        """
        super(AccuracyTreshold, self).update((self.trf(x=output[0]).int(), output[1].int()))

    def print(self):
        """
        Computes metric.

        :return: Accuracy threshold
        """
        return self.compute()
