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

class AUC_ROC():
    def __init__(self,n_classes):
        self.values = torch.arange(0,1.01, 0.01)
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.true_positives = torch.zeros((len(self.values), self.n_classes, ))
        self.false_positives = torch.zeros((len(self.values), self.n_classes,))
        self.all_positives = torch.zeros((self.n_classes,))
        self.all_negatives = torch.zeros((self.n_classes,))
        self.n = 0

    def update(self, batch):
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        self.n += batch[0].shape[0]
        self.reduction = "macro"
        positives_mask = batch[1]==1
        negatives_mask = batch[1]==0
        self.all_positives += (positives_mask).sum(0)
        self.all_negatives += (negatives_mask).sum(0)

        for i, value in enumerate(self.values):
            prediction = torch.zeros_like(batch[0])
            prediction[batch[0] > value] = 1
            self.true_positives[i] +=  ((prediction / positives_mask) == 1).sum(0)
            self.false_positives[i] += ((prediction / positives_mask) == float("inf")).sum(0)

    def compute(self):

        if self.reduction == "micro":
            tpr = self.true_positives.sum(-1) / self.all_positives.sum()
            fpr = self.false_positives.sum(-1) / self.all_negatives.sum()

        if self.reduction == "macro":
            tpr = self.true_positives / self.all_positives[None, :]
            tpr[torch.isnan(tpr)] = 0
            fpr = self.false_positives / self.all_negatives[None, :]
            tpr=tpr.mean(-1)
            fpr = fpr.mean(-1)

        return skm.auc(fpr, tpr), (fpr, tpr)

