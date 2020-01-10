import sklearn.metrics as skm
import torch

class MultiLabelReport():
    def __init__(self, classes):
        self.classes = classes
        self.truth = []
        self.pred = []
    def update(self, batch):
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        self.truth.append(batch[0])
        self.pred.append(batch[1])
    def compute(self):
        return skm.classification_report(torch.cat(self.truth), torch.cat(self.pred), output_dict=True, target_names=list(self.classes.keys()))

