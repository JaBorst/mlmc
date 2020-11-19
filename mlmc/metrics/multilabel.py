import sklearn.metrics as skm
import torch


class MultiLabelReport():
    """Multilabel iterative F1/Precision/Recall. Ignite API like"""
    def __init__(self, check_zeros=False, is_multilabel=True,**kwargs):
        self.check_zeros = check_zeros
        self.is_multilabel = is_multilabel
        self.reset()

    def init(self, classes, _threshold_fct, target, **kwargs):
        "an extra function for model specific parameters of the metric"
        self.classes = classes
        self.trf = _threshold_fct
        self.target = target

    def reset(self):
        self.truth = []
        self.pred = []

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
        pred = self.trf(torch.cat(self.pred))
        truth = (torch.cat(self.truth)).numpy()
        if not self.is_multilabel:
            truth = torch.nn.functional.one_hot(torch.cat(self.truth), len(self.classes)).numpy()
        return skm.classification_report(truth,
                                         pred.numpy(),
                                         output_dict=True,
                                         target_names=list(self.classes.keys()))
    def print(self):
        r = self.compute()
        return {k:v for k,v in r.items() if "micro" in k or "macro" in k}

class AUC_ROC():
    """Multilabel iterative AUC_ROC. Ignite API like"""
    def __init__(self,return_roc=True, reduction="macro"):
        self.values = torch.arange(0,1.01, 0.01)
        self.reduction = reduction
        self.return_roc = return_roc
        self.n=0

    def reset(self):
        if hasattr(self,"n_classes"):
            self.true_positives = torch.zeros((len(self.values), self.n_classes, ))
            self.false_positives = torch.zeros((len(self.values), self.n_classes,))
            self.all_positives = torch.zeros((self.n_classes,))
            self.all_negatives = torch.zeros((self.n_classes,))
        self.n = 0

    def update(self, batch):
        if self.n == 0:
            # If this is the first batch update, infer the size of the class set
            self.n_classes = batch[0].shape[-1]
            self.reset()
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

    def rates(self):
        if self.reduction == "micro":
            tpr = self.true_positives.sum(-1) / self.all_positives.sum()
            fpr = self.false_positives.sum(-1) / self.all_negatives.sum()

        if self.reduction == "macro":
            tpr = self.true_positives / self.all_positives[None, :]
            tpr[torch.isnan(tpr)] = 0
            fpr = self.false_positives / self.all_negatives[None, :]
            tpr = tpr.mean(-1)
            fpr = fpr.mean(-1)
        return fpr,tpr

    def compute(self):
        fpr, tpr = self.rates()
        return (skm.auc(fpr, tpr), (fpr.tolist(), tpr.tolist()) )if self.return_roc else skm.auc(fpr, tpr)


    def print(self):
        fpr, tpr = self.rates()
        return skm.auc(fpr, tpr)
