import torch
import mlmc
from mlmc.data import SingleLabelDataset, MultiLabelDataset
from mlmc.ensembles.descision_criteria import *
from tqdm import tqdm
d = mlmc.data.get("agnews")
device="cuda:1"




class BinaryEnsemble:
    """Train a binary model for every class"""
    def __init__(self, factory, classes, treshold="max",loss = torch.nn.CrossEntropyLoss(), target="single", zero=None):
        self._config={"target":target, "loss":loss}
        self.classes=classes
        self.m = [factory() for _ in range(len(classes)) ]
        for i, cls in enumerate(classes.keys()): self.m[i].create_labels({cls:0})
        self.loss = loss
        self.set_train()
        self.set_threshold(treshold)


    def set_train(self, t=None):
        if t==None:
            self.t=[True]*len(self.m)

    def fit(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=1e-2,
            return_roc=False, return_report=False, callbacks=None, metrics=None, lr_schedule=None, lr_param={},
            log_mlflow=False):

        self.validation = []
        self.train_history = {"loss": []}
        for e in range(epochs):
            losses = {"loss": str(0.)}
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                loss = self._epoch(train_loader, pbar=pbar)
                self.train_history["loss"].append(loss)
                if log_mlflow:
                    import mlflow
                    mlflow.log_metric("loss", loss, step=e)

    def _epoch(self, train, pbar=None):
        """Combining into training loop"""
        from ignite.metrics import Average
        average = Average()
        for i, b in enumerate(train):
            for m in self.m : m.optimizer.zero_grad()
            l, _ = self._step(b)
            l.backward()
            for m in self.m : m.optimizer.step()
            average.update(l.item())

            if pbar is not None:
                pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                pbar.update()
        return average.compute().item()

    def _scores(self, x):
        return torch.cat([m(m.transform(x)) for m in self.m],-1)

    def _step(self, b):
        """
        This method gets input and output for of one batch and calculates output and predictions
        Args:
            x: input tensor
            y: tensor of truth indices

        Returns:
            loss, output: loss tensor, and the raw prediction output of the network
        """
        y = b["labels"].to(self.m[0].device)
        output = self._scores(b["text"])
        l = self._loss(output, y)
        l = self._regularize(l)
        return l, output

    def _loss(self, x, y):
        """
        Calculating the loss getting  of two tensors using the initiated loss function
        When implementing new models with more complex loss functions, you can reimplement this method in the
        child class to apply them.
        Args:
            x: ouput tensor of the foward pass
            y: true labels

        Returns:
            loss tensor
        """
        return self.loss(x, y)

    def _regularize(self, l):
        """
        This function checks if a regularize method is implemented. If so the method is exectuted and the
        result will be added to the loss function.
        Args:
            l: the previously calculated raw loss

        Returns:
            The loss with the added regularization term
        """
        if hasattr(self, "regularize"):
            return l + self.regularize()
        else:
            return l

    def _init_metrics(self, metrics=None):
        """
        Initializes metrics to be used. If no metrics are specified then depending on the target the default metrics
        for this target will be used. (see mlmc.metrics.metrics_config.items())

        :param metrics: Name of the metrics (see mlmc.metrics.metrics_dict.keys() and mlmc.metrics.metrics_config.keys())
        :return: A dictionary containing the initialized metrics
        """
        if metrics is None:
            metrics = ["accuracy", "multilabel_report" if self._config["target"]=="multi" else "singlelabel_report"]
        metrics = mlmc.metrics.MetricsDict(metrics)
        metrics.init(self.__dict__)
        metrics.reset()
        metrics.rename({"multilabel_report": "report", "singlelabel_report": "report"})
        return metrics


    def evaluate(self, data, batch_size=50, metrics=None, _fit=False):
        """
        Evaluation, return accuracy and loss and some multilabel measure

        Returns p@1, p@3, p@5, AUC, loss, Accuracy@0.5, Accuracy@mcut, ROC Values, class-wise F1, Precision and Recall.
        Args:
            data: A MultilabelDataset with the data for evaluation
            batch_size: The batch size of the evaluation loop. (Larger is often faster, but it e.single()should be small enough
            to fit into GPU memory. In general it can be larger than batch_size in training.
            return_roc: If True, the return dictionary contains the ROC values.
            return_report: If True, the return dictionary will contain a class wise report of F1, Precision and Recall.
            metrics: Additional metrics
        Returns:
            A dictionary with the evaluation measurements.
        """

        initialized_metrics = self._init_metrics(metrics)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                output = self.predict(b["text"])
                pred = self._threshold_fct(output)
                initialized_metrics.update_metrics((output.cpu(), y, pred.cpu()))

        if _fit:
            return initialized_metrics
        else:
            return initialized_metrics.compute()

    def set_threshold(self, name):
        """
        Sets the threshold function which will be used to as a decision threshold.

        :param name: Name of the threshold (see mlmc.thresholds.threshold_dict.keys())
        """
        from mlmc.thresholds import get as thresholdget
        self._config["threshold"] = name
        if isinstance(name, str):
            self._threshold_fct = thresholdget(self._config["threshold"])
        elif callable(name):
            self._threshold_fct = name
        else:
            Warning("Threshold is neither callable nor a string")

    def predict(self, x):
        [m.eval() for m in self.m]  # set mode to evaluation to disable dropout
        scores = self._scores(x)
        [m.train() for m in self.m]
        return scores

    def single(self, *args, **kwargs):
        [m.single(*args,**kwargs) for m in self.m]
    def multi(self, *args, **kwargs):
        [m.multi(*args,**kwargs) for m in self.m]
    def entailment(self, *args, **kwargs):
        [m.entailment(*args,**kwargs) for m in self.m]

def create_model():
    return mlmc.models.EmbeddingBasedWeighted(mode="max",
                                              sformatter=mlmc.data.SFORMATTER["agnews"],
                                              finetune=True, classes={}, target="multi", device=device)


zeromodel=[mlmc.models.SimpleEncoder(representation="roberta-large-mnli", sformatter=mlmc.data.SFORMATTER["agnews"],
                                     finetune=True, classes=d["classes"], target="single", device=device)]
from  mlmc_lab import mlmc_experimental as mlmce
e = BinaryEnsemble(create_model, classes=d["classes"], loss=mlmce.loss.EncourageLoss(0.75), zero=None)
e.fit(mlmc.data.sampler(d["train"], absolute=100), epochs=50)
e.evaluate(mlmc.data.sampler(d["test"], absolute=1000))


