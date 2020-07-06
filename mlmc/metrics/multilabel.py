import sklearn.metrics as skm
import torch


class MultiLabelReport():
    """Multilabel iterative F1/Precision/Recall. Ignite API like"""
    def __init__(self, classes, trf=lambda x : x,  check_zeros=False, **kwargs):
        self.classes = classes
        self.truth = []
        self.pred = []
        self.check_zeros = check_zeros
        self.trf = trf
        self.kwargs = kwargs

    def update(self, batch):
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        if self.check_zeros:
            non_zero_rows = (((batch[1]==0).sum(-1)==batch[1].shape[-1]).int()) ==0
            self.truth.append(batch[1][non_zero_rows])
            self.pred.append(batch[0][non_zero_rows])
        else:
            self.truth.append(batch[1])
            self.pred.append(batch[0])
    def compute(self):
        pred = self.trf(torch.cat(self.pred),**self.kwargs)
        return skm.classification_report((torch.cat(self.truth)).numpy(),
                                         pred.numpy(),
                                         output_dict=True,
                                         target_names=list(self.classes.keys()))

class AUC_ROC():
    """Multilabel iterative AUC_ROC. Ignite API like"""
    def __init__(self,n_classes, return_roc=False, reduction="macro"):
        self.values = torch.arange(0,1.01, 0.01)
        self.n_classes = n_classes
        self.reset()
        self.reduction = reduction
        self.return_roc = return_roc

    def reset(self):
        self.true_positives = torch.zeros((len(self.values), self.n_classes, ))
        self.false_positives = torch.zeros((len(self.values), self.n_classes,))
        self.all_positives = torch.zeros((self.n_classes,))
        self.all_negatives = torch.zeros((self.n_classes,))
        self.n = 0

    def update(self, batch):
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        self.n += batch[0].shape[0]
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
            tpr = tpr.mean(-1)
            fpr = fpr.mean(-1)

        return (skm.auc(fpr, tpr), (fpr.tolist(), tpr.tolist()) )if self.return_roc else skm.auc(fpr, tpr)




