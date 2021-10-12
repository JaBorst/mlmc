import torch
from tqdm import tqdm

from mlmc.metrics.multilabel import MultiLabelReport, AUC_ROC
from mlmc.representation import is_transformer, get
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.thresholds import get as  thresholdget
from ...metrics import MetricsDict
from ...representation.character import  makemultilabels
from mlmc.data import MultiOutputMultiLabelDataset, SingleLabelDataset
from ...data.dataset_classes import MultiOutputSingleLabelDataset
import re


class TextClassificationAbstractMultiOutput(TextClassificationAbstract):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """

    def __init__(self, aggregation="mean", class_weights=None, **kwargs):
        """
        Abstract initializer of a Text Classification network.
        Args:
            target: single label oder multilabel mode. defined by keystrings: ("single", "multi"). Sets some basic
            options, like loss function, activation and
                    metrics to sensible defaults.
            activation: The activation function applied to the output. Only used for metrics and when you want to
            return scores in predict. (default: torch.softmax for "single", torch.sigmoid for "multi")
            loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss for "multi" and
            torch.nn.CrossEntropyLoss for "single")
            optimizer:  One of toch.optim (default: torch.optim.Adam)
            optimizer_params: A dictionary of optimizer parameters
            device: torch device, destination of training (cpu or cuda:0)
        """
        super(TextClassificationAbstractMultiOutput, self).__init__(**kwargs)
        self.class_weights = class_weights
        self.aggregation = aggregation

        self.n_classes = [len(x) for x in self.classes]
        self.n_outputs = len(self.classes)

        self._config["class_weights"] = class_weights
        self._config["aggregation"] = aggregation
        self._config["n_classes"] = self.n_classes
        self._config["n_outputs"] = self.n_outputs

    def set_loss(self, loss):
        self._config["loss"] = loss

    def build(self):
        """
            Internal build method.
        """
        if isinstance(self._config["loss"], type) and self._config["loss"] is not None:
            if self._config.get("class_weights", None) is not None:
                self.loss = [self._config["loss"](torch.FloatTensor(w).to(self.device)) for w in self.class_weights]
            else:
                self.loss = [self._config["loss"]() for _ in range(self.n_outputs)]
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(
                filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)

    def act(self, x):
        """
        Applies activation function to output tensor.

        :param x: An input tensor
        :return: A tensor
        """
        if "softmax" in self.activation.__name__ or "softmin" in self.activation.__name__:
            return [self.activation(o, -1) for o in x]
        else:
            return [self.activation(o) for o in x]

    def _init_metrics(self, metrics=None):
        """
        Initializes metrics to be used. If no metrics are specified then depending on the target the default metrics
        for this target will be used. (see mlmc.metrics.metrics_config.items())

        :param metrics: Name of the metrics (see mlmc.metrics.metrics_dict.keys() and mlmc.metrics.metrics_config.keys())
        :return: A dictionary containing the initialized metrics
        """
        from copy import deepcopy
        if metrics is None:
            metrics=f"default_{self.target}label"
        metrics = [MetricsDict(metrics) for i in range(len(self.classes))]

        for i, m in enumerate(metrics):
            m.rename({k: f"{k}_{i}" for k in m.map.keys()})

        for m, i in zip(metrics, self.classes):
            m.init({"classes":i, "_threshold_fct":self._threshold_fct, "target":self.target})
        return metrics

    def evaluate_classes(self, classes_subset=None, **kwargs):
        """wrapper for evaluation function if you just want to evaluate on subsets of the classes."""
        if classes_subset is None:
            return self.evaluate(**kwargs)
        else:
            mask = makemultilabels([list(classes_subset.values())], maxlen=len(self.classes))
            return self.evaluate(**kwargs, mask=mask)

    def evaluate(self, data, batch_size=50, mask=None, metrics=None, _fit=False):
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
        self.eval()  # set mode to evaluation to disable dropout

        initialized_metrics = self._init_metrics(metrics)
        from ignite.metrics import Average
        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                if isinstance(b["labels"], list):
                    y = [y.to(self.device) for y in b["labels"]]
                else:
                    y = b["labels"].to(self.device).t()
                l, output = self._step(x=self.transform(b["text"]).to(self.device), y=y)
                output = self.act(output)
                pred = [self._threshold_fct(o) for o in output]

                average.update(l.item())
                for p, o, t, m in zip(pred, output, y, initialized_metrics):
                    m.update_metrics((o.cpu(), t.cpu(), p.cpu()))

        self.train()
        if _fit:
            return average.compute().item(), initialized_metrics
        else:
            return average.compute().item(), [x.compute() for x in initialized_metrics]

    def fit(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=1e-2,
            return_roc=False, return_report=False, callbacks=None, metrics=None, lr_schedule=None, lr_param={}):
        """
        Training function

        Args:
            train: MultilabelDataset used as training data
            valid: MultilabelDataset to keep track of generalization
            epochs: Number of epochs (times to iterate the train data)
            batch_size: Number of instances in one batch.
            valid_batch_size: Number of instances in one batch  of validation.
            patience: (default -1) Early Stopping Arguments.
            Number of epochs to wait for performance improvements before exiting the training loop.
            tolerance: (default 1e-2) Early Stopping Arguments.
            Minimum improvement of an epoch over the best validation loss so far.

        Returns:
            A history dictionary with the loss and the validation evaluation measurements.

        """
        if callbacks is None:
            callbacks = []
        import datetime
        id = str(hash(datetime.datetime.now()))[1:7]
        if isinstance(train, SingleLabelDataset) and self.target != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        self.validation = []
        self.train_history = {"loss": []}


        if lr_schedule is not None:
            scheduler = lr_schedule(self.optimizer, **lr_param)
        for e in range(epochs):
            self._callback_epoch_start(callbacks)

            # An epoch
            losses = {"loss": str(0.)}
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                loss = self._epoch(train_loader, pbar=pbar)
                if lr_schedule is not None: scheduler.step()
                self.train_history["loss"].append(loss)

                # Validation if available
                if valid is not None:
                    valid_loss, result_metrics = self.evaluate(
                        data=valid,
                        batch_size=valid_batch_size,
                        _fit=True)

                    valid_loss_dict= {"valid_loss": valid_loss}
                    valid_loss_dict.update({k:v for d in result_metrics for k,v in d.compute().items()})
                    self.validation.append(valid_loss_dict)

                    printables= {"valid_loss": valid_loss}
                    printables.update({k:v for i,d in enumerate(result_metrics) for k,v in d.print().items()})
                    pbar.postfix[0].update(printables)
                    pbar.update()

            # Callbacks
            self._callback_epoch_end(callbacks)

        self._callback_train_end(callbacks)
        if patience > -1:
            self.load_state_dict(torch.load(id + "_checkpoint.pt"))
        # Load best
        from copy import copy
        return_copy = {"train": copy(self.train_history), "valid": copy(self.validation)}
        return return_copy

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
        l = torch.stack([l(o, t) for l, o, t in zip(self.loss, x, y)])
        if self.aggregation == "sum":
            l = l.sum()
        if self.aggregation == "mean":
            l = l.mean()
        return l

    def predict(self, x, return_scores=False):
        """
        Classify sentence string  or a list of strings.

        Args:
            x:  A list of the text instances.
            return_scores:  If True, the labels are returned with their corresponding confidence scores
            tr: The threshold at which the labels are returned.
            method: Method of thresholding
                    (hard will cutoff at ``tr``, mcut will look for the largest distance in
                    confidence between two labels following each other and will return everything above)

        Returns:
            A list of the labels or a tuple of (labels, scores, mask) if return_scores is True

        """

        output = self.scores(x)
        with torch.no_grad():
            predictions = [self._threshold_fct(o) for o in output]
        labels = [[[d[i.item()] for i in torch.where(p == 1)[0]] for p in prediction] for d, prediction in
                      zip(self.classes_rev, predictions)]
        labels = list(zip(*labels))

        if return_scores:
            return labels, output, predictions
        return labels

    def scores(self, x):
        self.eval()

        if not hasattr(self, "classes_rev"):
            self.classes_rev = [{v: k for k, v in classes.items()} for classes in self.classes]
        x = self.transform(x).to(self.device)
        if len(x.shape) == 1: x[None]
        with torch.no_grad():
            output = self.act(self(x))
        self.train()
        return output

    def rebuild(self):
        """
        Internal build method.
        """
        self.loss = [type(x)().to(self.device) for x in self.loss]
        self.optimizer = type(self.optimizer)(
            filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)

    def _epoch(self, train, pbar=None):
        """Combining into training loop"""
        from ignite.metrics import Average
        average = Average()
        for i, b in enumerate(train):
            self.optimizer.zero_grad()
            l, _ = self._step(b)
            l.backward()
            self.optimizer.step()
            average.update(l.item())

            if pbar is not None:
                pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                pbar.update()
        return average.compute().item()

    def _step(self, b):
        """
        This method gets input and output for of one batch and calculates output and predictions
        Args:
            x: input tensor
            y: tensor of truth indices

        Returns:
            loss, output: loss tensor, and the raw prediction output of the network
        """
        x = self.transform(b["text"])
        if isinstance(b["labels"], list):
            y = [y.to(self.device) for y in b["labels"]]
        else:
            y = b["labels"].to(self.device).t()
        output = self(x)
        l = self._loss(output, y)
        l = self._regularize(l)
        return l, output