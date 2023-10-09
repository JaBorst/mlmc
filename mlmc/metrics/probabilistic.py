from scipy.stats import beta
import torch
import numpy as np



class ProbabilisticReport():
    """Multilabel iterative F1/Precision/Recall. Ignite API like"""
    def __init__(self, check_zeros=False, is_multilabel=False, probability=0.9, n=5000, **kwargs):
        self.check_zeros = check_zeros
        self.is_multilabel = is_multilabel
        self.probability = probability
        self.n = n
        self.truth = []
        self.pred = []
        self.reset()

    def init(self, classes, **kwargs):
        "an extra function for model specific parameters of the metric"
        self.classes = classes

    def reset(self):
        """Clears previously added truth and pred instance attributes."""
        # self.truth = []
        # self.pred = []
        pass

    def update(self, batch):
        """
        Adds classification output to class for computation of metric.

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

    def _plot(self, m="f1-score"):
        pred = torch.cat(self.pred)
        truth = torch.cat(self.truth)

        truth = torch.nn.functional.one_hot(truth, len(self.classes))

        tp = (pred * truth).sum(0)
        fp = (pred * (1 - truth)).sum(0)
        fn = ((1 - pred) * (truth)).sum(0)

        precision = beta(1 + tp, 1 + fp)
        recall = beta(1 + tp, 1 + fn)
        f1 = beta(2 * tp, 1 + fp + fn)

        import matplotlib.pyplot as plt
        r = precision if m == "precision" else recall if m=="recall" else f1

        x = np.linspace(r.ppf(0.001),
                        r.ppf(0.999), 100)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, r.pdf(x), lw=2, alpha=0.6, label=self.classes.keys())
        ax.legend(loc='best', frameon=False)
        ax.set_xlim((0,1))
        plt.show()

    def compute(self,*args, **kwargs):
        """
        Computes metric.

        :return: Classification report
        """
        pred = torch.cat(self.pred)
        truth = torch.cat(self.truth)

        truth = torch.nn.functional.one_hot(truth, len(self.classes))

        tp = (pred*truth).sum(0)
        fp = (pred * (1-truth)).sum(0)
        fn = ((1-pred)*(truth)).sum(0)



        precision  = beta(1+tp, 1+fp)
        recall  = beta(1+tp, 1+fn)
        f1  = beta(2*tp, 1+ fp + fn)
        s = truth.sum(0)

        _val = lambda x: (x.mean(), x.ppf((1-self.probability)/2), x.ppf(1-(1-self.probability)/2))

        classwise = [dict(zip(["precision", "recall", "f1-score","support"],x)) for x  in zip(*_val(precision), *_val(recall), *_val(f1), s.tolist())]
        r = dict(zip(self.classes.keys(),classwise))


        r["micro avg"] = {"precision": _val(beta(1 + tp.sum(), 1 + fp.sum())) ,
                          "recall": _val(beta(1 + tp.sum(), 1 + fn.sum())) ,
                          "f1-score": _val(beta(2 * tp.sum(), 1 + fp.sum() + fn.sum())),
                          "support": pred.shape[0]
                          }

        _avg = lambda x: beta(*beta.fit((x.rvs((  self.n, len(self.classes),)).mean(1))))

        r["macro avg"] ={
            "precision": _val(_avg(precision)),
            "recall":_val(_avg(recall)),
            "f1-score": _val(_avg(f1)),
            "support": pred.shape[0]
        }
        return r

    def print(self, *args, **kwargs):
        """
        Computes metric.

        :return: Classification report
        """
        r = self.compute(*args, **kwargs)
        return {k:v for k,v in r.items() if "micro" in k or "macro" in k}
