import torch
import mlmc
from mlmc.data import SingleLabelDataset, MultiLabelDataset

d = mlmc.data.get("agnews")
device="cuda:3"
# m = [mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device),
#      mlmc.models.Transformer(classes=d["classes"], target="single", device=device)]

m = [mlmc.models.EmbeddingBasedWeighted(mode="vanilla", finetune=True, classes=d["classes"], target="single", device=device),
     mlmc.models.EmbeddingBasedWeighted(mode="max", finetune=True, classes=d["classes"], target="single", device=device),
     mlmc.models.EmbeddingBasedWeighted(mode="mean", finetune=True, classes=d["classes"], target="single", device=device),
     mlmc.models.EmbeddingBasedWeighted(mode="max_mean", finetune=True, classes=d["classes"], target="single", device=device),
     mlmc.models.EmbeddingBasedWeighted(mode="attention_max_mean", finetune=True, classes=d["classes"], target="single", device=device)]


m = m+[mlmc.models.SimpleEncoder(representation="roberta-large-mnli", sformatter=mlmc.data.SFORMATTER["agnews"], finetune=True, classes=d["classes"], target="single", device=device)]

class Decision(torch.nn.Module):
    def forward(self, scores):
        pass

class MajorityDecision(torch.nn.Module):
    def forward(self, scores):
        scores_stack = torch.stack(scores, -1)
        return scores_stack.argmax(-2).mode().indices

class ConfidenceDecision(torch.nn.Module):
    def forward(self, scores):
        scores_stack = torch.stack(scores,-1)
        return scores_stack.max(-2)[0].argmax(-1)

class EntropyDecision(torch.nn.Module):
    def forward(self, scores):
        scores_stack = torch.stack(scores,-1)
        return (-(scores_stack.softmax(-2)*scores_stack.log_softmax(-2)).sum(-2)).argmin(-1)


class BinaryEnsemble:
    """Train a binary model for every class"""
    def __init__(self, m, classes):
        self.classes
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

    def single(self, *args, **kwargs):
        [m.single(*args,**kwargs) for m in self.m]
    def multi(self, *args, **kwargs):
        [m.multi(*args,**kwargs) for m in self.m]
    def entailment(self, *args, **kwargs):
        [m.entailment(*args,**kwargs) for m in self.m]

e = Ensemble(m)
e.t[-1] = False
e.fit(mlmc.data.sampler(d["train"], absolute=1000), epochs=5)
e.evaluate(mlmc.data.sampler(d["test"], absolute=1000))
e.evaluate_ensemble(mlmc.data.sampler(d["test"], absolute=1000))


