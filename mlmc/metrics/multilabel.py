import sklearn.metrics as skm
import torch


class MultiLabelReport():
    """Multilabel iterative F1/Precision/Recall. Ignite API like"""
    def __init__(self, check_zeros=False, is_multilabel=True,**kwargs):
        self.check_zeros = check_zeros
        self.is_multilabel = is_multilabel
        self.reset()

    def init(self, classes, target, **kwargs):
        "an extra function for model specific parameters of the metric"
        self.classes = classes

    def reset(self):
        """
        Clears previously added truth and pred instance attributes.
        """
        self.truth = []
        self.pred = []

    def update(self, batch):
        """
        Adds classification output to class for computation of metric

        :param batch: Output of classification task in form (scores, truth, pred)
        """
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        if self.check_zeros:
            non_zero_rows = (((batch[1]==0).sum(-1)==batch[1].shape[-1]).int()) ==0
            self.truth.append(batch[1][non_zero_rows])
            self.pred.append(batch[2][non_zero_rows])
        else:
            self.truth.append(batch[1])
            self.pred.append(batch[2])

    def compute(self):
        """
        Computes metric.

        :return: Classification report
        """
        pred = torch.cat(self.pred)
        truth = torch.cat(self.truth)
        if not self.is_multilabel:
            truth = torch.nn.functional.one_hot(truth, len(self.classes))
        return skm.classification_report(truth,
                                         pred.numpy(),
                                         output_dict=True,
                                         target_names=list(self.classes.keys()))
    def print(self):
        """
        Computes metric.

        :return: Classification report
        """
        r = self.compute()
        return {k:v for k,v in r.items() if "micro" in k or "macro" in k}



class AUC_ROC():
    """Multilabel iterative AUC_ROC. Ignite API like"""
    def __init__(self,return_roc=True, reduction="macro"):
        """
        Initializes metric.

        :param return_roc: If true returns ROC curve in list form
        :param reduction: "micro" for micro-averaging or "macro" for macro-averaging
        """
        self.values = torch.arange(0,1.01, 0.01)
        self.reduction = reduction
        self.return_roc = return_roc
        self.n=0

    def reset(self):
        """
        Clears previously added truth and pred instance attributes.
        """
        if hasattr(self,"n_classes"):
            self.true_positives = torch.zeros((len(self.values), self.n_classes, ))
            self.false_positives = torch.zeros((len(self.values), self.n_classes,))
            self.all_positives = torch.zeros((self.n_classes,))
            self.all_negatives = torch.zeros((self.n_classes,))
        self.n = 0

    def update(self, batch):
        """
        Adds classification output to class for computation of metric.

        :param batch: Output of classification task in form (scores, truth, pred)
        """
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
        """
        Calculates false positive rate (fpr) and true positive rate (tpr).

        :return: false positive rate and true positive rate
        """
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
        """
        Computes metric.

        :return: AUC score and ROC curve if return_roc is True, else only AUC score
        """
        fpr, tpr = self.rates()
        return (skm.auc(fpr, tpr), (fpr.tolist(), tpr.tolist()) ) if self.return_roc else skm.auc(fpr, tpr)


    def print(self):
        """
        Computes metric.

        :return: AUC score
        """
        fpr, tpr = self.rates()
        return skm.auc(fpr, tpr)
