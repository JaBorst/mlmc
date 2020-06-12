from ignite.metrics import Precision, Accuracy
import torch


class PrecisionK(Precision):
    def __init__(self, k, *args, **kwargs):
        self.k = k
        super(PrecisionK, self).__init__(*args, **kwargs)

    def update(self, output):
        transformed = torch.zeros_like(output[0]).scatter(1, torch.topk(output[0], k=self.k)[1], 1)
        super(PrecisionK, self).update((transformed, output[1]))


class AccuracyTreshold():
    def __init__(self, trf, args_dict={}):
        self.trf = trf
        self.args_dict = args_dict
        self.l = []
    def update(self, output):
        self.l.extend((self.trf(output[0], **self.args_dict) == output[1]).all(-1).tolist())
    def compute(self):
        return sum(self.l)/len(self.l)