from ignite.metrics import Precision, Accuracy
import torch


class PrecisionK(Precision):
    def __init__(self, k, *args, **kwargs):
        self.k = k
        super(PrecisionK, self).__init__(*args, **kwargs)

    def update(self, output):
        transformed = torch.zeros_like(output[0]).scatter(1, torch.topk(output[0], k=self.k)[1], 1)
        super(PrecisionK, self).update((transformed, output[1]))


class AccuracyTreshold(Accuracy):
    def __init__(self, trf, args_dict=None, activation=torch.sigmoid, *args, **kwargs):
        self.trf = trf
        self.args_dict = args_dict
        self.activation = activation
        super(AccuracyTreshold, self).__init__(*args, **kwargs)

    def update(self, output):
        super(AccuracyTreshold, self).update((self.trf(x=self.activation(output[0]), **self.args_dict), output[1]))
