import torch
import mlmc
from mlmc.data import SingleLabelDataset, MultiLabelDataset
from mlmc.ensembles.descision_criteria import *
d = mlmc.data.get("agnews")
device="cuda:1"
# m = [mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device)]



class Ensemble:
    def __init__(self, m):
        self.m = m if isinstance(m, list) else [m]
        self.vote = MajorityDecision()
        self.set_train()

    def set_train(self, t=None):
        if t==None:
            self.t=[True]*len(self.m)
    def fit(self, *args, **kwargs):

        for m, train in zip(self.m, self.t):
            if train:
                m.fit(*args, **kwargs)

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
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                _, output, pred = self.predict_ensemble(b["text"])
                initialized_metrics.update_metrics((output, y, pred))

        [m.train() for m in self.m]  # set mode to evaluation to disable dropout

        if _fit:
            return initialized_metrics
        else:
            return initialized_metrics.compute()

    def predict_ensemble(self, *args, **kwargs):
        [m.eval() for m in self.m]  # set mode to evaluation to disable dropout
        kwargs["return_scores"]=True
        scores=[m.predict(*args,**kwargs) for m in self.m]
        idx = self.vote([s[1] for s in scores])
        s = torch.stack([x[i]  for x,i in zip(zip(*[s[1] for s in scores]), idx.tolist())],0)
        p = torch.stack([x[i]  for x,i in zip(zip(*[s[2] for s in scores]), idx.tolist())],0)
        [m.train() for m in self.m]
        return [x[i] for x,i in zip(zip(*[s[0] for s in scores]),idx)], s, p

    def predict_ensemble_batch(self, *args, **kwargs):
        [m.eval() for m in self.m]  # set mode to evaluation to disable dropout
        kwargs["return_scores"]=True
        scores=[m.predict_batch(*args,**kwargs) for m in self.m]
        idx = self.vote([s[1] for s in scores])
        s = torch.stack([x[i]  for x,i in zip(zip(*[s[1] for s in scores]), idx.tolist())],0)
        p = torch.stack([x[i]  for x,i in zip(zip(*[s[2] for s in scores]), idx.tolist())],0)
        [m.train() for m in self.m]
        return [x[i] for x,i in zip(zip(*[s[0] for s in scores]),idx)], s, p


    def single(self, *args, **kwargs):
        [m.single(*args,**kwargs) for m in self.m]
    def multi(self, *args, **kwargs):
        [m.multi(*args,**kwargs) for m in self.m]
    def entailment(self, *args, **kwargs):
        [m.entailment(*args,**kwargs) for m in self.m]
#
# r = "google/bert_uncased_L-4_H-256_A-4"
# from mlmc_lab import mlmc_experimental as mlmce
# m = [mlmc.models.EmbeddingBasedWeighted(mode="vanilla", representation=r, sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single", loss=mlmce.loss.EncourageLoss(0.75), device=device),
#      mlmc.models.EmbeddingBasedWeighted(mode="max", representation=r, sformatter=mlmc.data.SFORMATTER["agnews"],  finetune=True, classes=d["classes"], target="single",loss=mlmce.loss.EncourageLoss(0.75), device=device),
#      mlmc.models.EmbeddingBasedWeighted(mode="mean", representation=r, sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single",loss=mlmce.loss.EncourageLoss(0.75),device=device),
#      mlmc.models.EmbeddingBasedWeighted(mode="max_mean",representation=r,  sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single", loss=mlmce.loss.EncourageLoss(0.75),device=device),
#      mlmc.models.EmbeddingBasedWeighted(mode="attention_max_mean", representation=r, sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single", loss=mlmce.loss.EncourageLoss(0.75),device=device)]
# # m = m+[mlmc.models.SimpleEncoder(representation="roberta-large-mnli", sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single", device=device)]
#
# e = Ensemble(m)
# # e.t[-1] = False
# # e.fit(mlmc.data.sampler(d["train"], absolute=100), epochs=50)
# test=mlmc.data.sampler(d["test"], absolute=1000)
#
# print(e.evaluate(test))
# e.predict_ensemble_batch(test)
#

