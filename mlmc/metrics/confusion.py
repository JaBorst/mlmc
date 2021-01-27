from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import torch

class ConfusionMatrix():
    """
    Implementation of sklearn confusion matrix for singlelabel and multilabel usage.
    """
    def __init__(self, is_multilabel=False, **kwargs):
        self.is_multilabel = is_multilabel
        self.check_zeros = False
        self.reset()

    def reset(self):
        """Clears previously added truth and pred instance attributes."""
        self.truth = []
        self.pred = []

    def update(self, batch):
        """
        Adds classification output to class for computation of metric.

        :param batch: Output of classification task in form (scores, truth, pred)
        """
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        if self.check_zeros:
            non_zero_rows = (((batch[1] == 0).sum(-1) == batch[1].shape[-1]).int()) == 0
            self.truth.append(batch[1][non_zero_rows])
            self.pred.append(batch[0][non_zero_rows])
        else:
            self.truth.append(batch[1])
            self.pred.append(batch[0])

    def compute(self):
        """
        Computes metric.

        :return: Confusion matrix
        """
        pred = torch.cat(self.pred, 0)
        truth = torch.cat(self.truth, 0)
        if self.is_multilabel:
            return multilabel_confusion_matrix(truth.numpy(), pred.numpy())
        else:
            return confusion_matrix(truth.numpy(), pred.max(-1)[1].numpy())
