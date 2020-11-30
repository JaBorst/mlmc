from ignite.metrics import Precision, Accuracy
import torch


class PrecisionK(Precision):
    def __init__(self, k, *args, **kwargs):
        self.k = k
        super(PrecisionK, self).__init__(*args, **kwargs)

    def update(self, output):
        transformed = torch.zeros_like(output[0]).scatter(1, torch.topk(output[0], k=self.k)[1], 1)
        super(PrecisionK, self).update((transformed.int(), output[1].int()))

    def print(self):
        return self.compute()


class AccuracyTreshold(Accuracy):
    def __init__(self, trf, args_dict=None, *args, **kwargs):
        if args_dict is None:
            args_dict = {}
        self.trf = trf
        self.args_dict = args_dict
        super(AccuracyTreshold, self).__init__(*args, **kwargs)

    def update(self, output):
        super(AccuracyTreshold, self).update((self.trf(x=output[0]).int(), output[1].int()))

    def print(self):
        return self.compute()
