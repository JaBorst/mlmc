import mlmc
from mlmc.data import SingleLabelDataset, MultiLabelDataset
from mlmc.ensembles.descision_criteria import *


class Ensemble:
    def __init__(self, m, device="cpu"):
        self.m = m if isinstance(m, list) else [m]
        self.m = [m.cpu() for m in self.m]
        self.vote = MajorityDecision()
        self.set_train()
        self.device = device


    def set_train(self, t=None):
        if t==None:
            self.t=[True]*len(self.m)

    def fit(self, *args, **kwargs):
        for m, train in zip(self.m, self.t):
            if train:
                m = m.set_device(self.device)
                m.fit(*args, **kwargs)
                m.reset_memory()
                m = m.set_device("cpu")

    def evaluate(self, *args, **kwargs):
        return [m.evaluate(*args, **kwargs) for m in self.m]

    def _init_metrics(self, metrics=None):
        """
        Initializes metrics to be used. If no metrics are specified then depending on the target the default metrics
        for this target will be used. (see mlmc.metrics.metrics_config.items())

        :param metrics: Name of the metrics (see mlmc.metrics.metrics_dict.keys() and mlmc.metrics.metrics_config.keys())
        :return: A dictionary containing the initialized metrics
        """
        if metrics is None:
            metrics = ["accuracy", "multilabel_report" if self.m[0]._config["target"]=="multi" else "singlelabel_report"]
        metrics = mlmc.metrics.MetricsDict(metrics)
        metrics.init(self.m[0].__dict__)
        metrics.reset()
        metrics.rename({"multilabel_report": "report", "singlelabel_report": "report"})
        return metrics


    def evaluate_ensemble(self, data, batch_size=50, metrics=None, _fit=False):
        """
        Evaluation, return accuracy and loss and some multilabel measure

        Returns p@1, p@3, p@5, AUC, loss, Accuracy@0.5, Accuracy@mcut, ROC Values, class-wise F1, Precision and Recall.
        Args:
            data: A MultilabelDataset with the data for evaluation
            batch_size: The batch size of the evaluation loop. (Larger is often faster, but it should be small enough
            to fit into GPU memory. In general it can be larger than batch_size in training.
            return_roc: If True, the return dictionary contains the ROC values.
            return_report: If True, the return dictionary will contain a class wise report of F1, Precision and Recall.
            metrics: Additional metrics
        Returns:
            A dictionary with the evaluation measurements.
        """
        [m.eval() for m in self.m]  # set mode to evaluation to disable dropout

        assert not (type(data) == SingleLabelDataset and self.m[0]._config["target"] == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(data) == MultiLabelDataset and self.m[0]._config["target"] == "single"), \
            "You inserted a MultiLabelDataset but chose single as target."
        initialized_metrics = self._init_metrics(metrics)
        _, output, pred = self.predict_ensemble(data.x, batch_size=batch_size)
        initialized_metrics.update_metrics((output,torch.stack([x["labels"] for x in data]), pred))

        [m.train() for m in self.m]  # set mode to evaluation to disable dropout

        if _fit:
            return initialized_metrics
        else:
            return initialized_metrics.compute()

    def predict_ensemble(self, *args, vote=True, **kwargs):
        scores = self._get_ensemble_single(*args, **kwargs)

        if vote:
            idx = self.vote(scores[1])
            s = torch.stack([x[:, i] for x, i in zip(scores[1], idx.tolist())], 0)
            p = torch.stack([x[:, i] for x, i in zip(scores[2], idx.tolist())], 0)
            l = [x[i] for x, i in zip(scores[0], idx)]
            return l, s, p
        else:
            return scores

    def _get_ensemble_single(self, *args, **kwargs):
        [m.eval() for m in self.m]  # set mode to evaluation to disable dropout
        kwargs["return_scores"] = True
        scores = []
        for m in self.m:
            m.set_device(self.device)
            scores.append(m.predict_batch(*args, **kwargs))
            m.set_device("cpu")
            m.reset_memory()
        def _combine(t):
            if isinstance(t[0], list):
                return list(zip(*t))
            else:
                return torch.stack(t,-1)

        tup = [_combine([x[i] for x in scores]) for i in range(len(scores[0]))]
        return tup

    def single(self, *args, **kwargs):
        [m.single(*args,**kwargs) for m in self.m]
    def multi(self, *args, **kwargs):
        [m.multi(*args,**kwargs) for m in self.m]
    def entailment(self, *args, **kwargs):
        [m.entailment(*args,**kwargs) for m in self.m]
    def set_sformatter(self, *args, **kwargs):
        [m.set_sformatter(*args,**kwargs) for m in self.m]
    def create_labels(self, *args, **kwargs):
        [m.create_labels(*args,**kwargs) for m in self.m]
#
# r = "google/bert_uncased_L-4_H-256_A-4"
# from mlmc_lab import mlmc_experimental as mlmce
# d = mlmce.data.get("agnews")
# device="cuda:0"
# m = [
#     mlmc.models.Siamese(representation=r, sformatter=mlmce.data.SFORMATTER["agnews"], finetune="all", classes=d["classes"], target="single", loss=mlmce.loss.EncourageLoss(0.75), device=device),
#     mlmc.models.Siamese(representation=r, sformatter=mlmce.data.SFORMATTER["agnews"],  finetune="all", classes=d["classes"], target="single",loss=mlmce.loss.EncourageLoss(0.75), device=device),
#     mlmc.models.Siamese(epresentation=r, sformatter=mlmce.data.SFORMATTER["agnews"], finetune="all", classes=d["classes"], target="single",loss=mlmce.loss.EncourageLoss(0.75),device=device),
# ]
# # m = m+[mlmc.models.SimpleEncoder(representation="roberta-large-mnli", sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single", device=device)]
#
# e = Ensemble(m)
# # e.t[-1] = False
# # e.fit(mlmc.data.sampler(d["train"], absolute=100), epochs=50)
# test=mlmc.data.sampler(d["test"], absolute=100)
#
# print(e.evaluate(test))
# e.predict_ensemble(test.x, vote=False)
#
#
