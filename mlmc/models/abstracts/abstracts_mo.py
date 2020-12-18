import torch
from tqdm import tqdm

from mlmc.metrics.multilabel import MultiLabelReport, AUC_ROC
from mlmc.representation import is_transformer, get
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.thresholds import get as  thresholdget
from ...metrics import MetricsDict

from mlmc.data import MultiOutputMultiLabelDataset, MultiOutputSingleLabelDataset
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

    def build(self):
        """
        Internal build method.
        """
        if isinstance(self.loss, type) and self.loss is not None:
            if self.class_weights is not None:
                self.loss = [self.loss(torch.FloatTensor(w).to(self.device)) for w in self.class_weights]
            else:
                self.loss = [self.loss() for _ in range(self.n_outputs)]
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(
                filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)

    def act(self, x):
        if "softmax" in self.activation.__name__ or "softmin" in self.activation.__name__:
            return [self.activation(o, -1) for o in x]
        else:
            return [self.activation(o) for o in x]

    def _init_metrics(self, metrics=None):
        from copy import deepcopy
        if metrics is None:
            metrics=f"default_{self.target}label"
        metrics = [MetricsDict(metrics) for i in self.classes]
        for m, i in zip(metrics, self.classes):
            m.init({"classes":i, "_threshold_fct":self._threshold_fct, "target":self.target})
        return metrics


    def evaluate(self, data, batch_size=50,  metrics=None):
        """
        Evaluation, return accuracy and loss and some multilabel measure

        Returns p@1, p@3, p@5, AUC, loss, Accuracy@0.5, Accuracy@mcut, ROC Values, class-wise F1, Precision and Recall.
        Args:
            data: A MultilabelDataset with the data for evaluation
            batch_size: The batch size of the evaluation loop. (Larger is often faster, but it should be small enough
            to fit into GPU memory. In general it can be larger than batch_size in training.
            return_roc: If True, the return dictionary contains the ROC values.
            return_report: If True, the return dictionary will contain a class wise report of F1, Precision and Recall.
        Returns:
            A dictionary with the evaluation measurements.
        """

        self.eval()  # set mode to evaluation to disable dropout
        from ignite.metrics import Average

        assert not (type(data) == MultiOutputSingleLabelDataset and self.target == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(data) == MultiOutputMultiLabelDataset and self.target == "single"), \
            "You inserted a MultiLabelDataset but chose single as target."


        initialized_metrics = self._init_metrics(metrics=metrics)

        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                l, output = self._step(x=self.transform(b["text"]).to(self.device), y=y.to(self.device))
                output = self.act(output)
                pred = [self._threshold_fct(o) for o in output]

                average.update(l.item())
                for p, o, t, m in zip(pred, output, y.transpose(0, 1), initialized_metrics):
                    m.update_metrics(( o.cpu(), t.cpu(), p.cpu()))
        self.train()
        return average.compute().item(), initialized_metrics

    def fit(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=1e-2,
            return_roc=False, return_report=False):
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
        import datetime
        id = str(hash(datetime.datetime.now()))[1:7]
        from ...data import SingleLabelDataset
        if isinstance(train, SingleLabelDataset) and self.target != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        validation = []
        train_history = {"loss": []}
        from ...data import MultiOutputSingleLabelDataset, MultiOutputMultiLabelDataset
        assert not (type(train) == MultiOutputSingleLabelDataset and self.target == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(train) == MultiOutputMultiLabelDataset and self.target == "single"), \
            "You inserted a MultiLabelDataset but chose single as target."

        best_loss = 10000000
        last_best_loss_update = 0
        from ignite.metrics import Average
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    l, _ = self._step(x=self.transform(b["text"]).to(self.device), y=b["labels"].to(self.device))

                    l.backward()
                    self.optimizer.step()

                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(), 2 * self.PRECISION_DIGITS)
                    pbar.update()
                # torch.cuda.empty_cache()
                if valid is not None:
                    valid_loss, result_metrics = self.evaluate(
                        data=valid,
                        batch_size=valid_batch_size)

                    valid_loss_dict= {"valid_loss": valid_loss}
                    valid_loss_dict.update({k:v for d in result_metrics for k,v in d.compute().items()})
                    validation.append(valid_loss_dict)

                    printables= {"valid_loss": valid_loss}
                    printables.update({k+f"_{i}":v for i,d in enumerate(result_metrics) for k,v in d.print().items()})
                    pbar.postfix[0].update(printables)
                    pbar.update()

            if patience > -1:
                if valid is None:
                    print("check validation loss")
                    if best_loss - average.compute().item() > tolerance:
                        print("update validation and checkoint")
                        best_loss = average.compute().item()
                        torch.save(self.state_dict(), id + "_checkpoint.pt")
                        # save states
                        last_best_loss_update = 0
                    else:
                        print("increment no epochs")
                        last_best_loss_update += 1

                    if last_best_loss_update >= patience:
                        print("breaking at %i" % (patience,))
                        print("Early Stopping.")
                        break
                elif valid is not None:
                    if best_loss - validation[-1]["valid_loss"] > tolerance:
                        best_loss = validation[-1]["valid_loss"]
                        torch.save(self.state_dict(), id + "_checkpoint.pt")
                        # save states
                        last_best_loss_update = 0
                    else:
                        last_best_loss_update += 1

                    if last_best_loss_update >= patience:
                        print("Early Stopping.")
                        break

            train_history["loss"].append(average.compute().item())
        if patience > -1:
            self.load_state_dict(torch.load(id + "_checkpoint.pt"))
        # Load best
        return {"train": train_history, "valid": validation}

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
        l = torch.stack([l(o, t) for l, o, t in zip(self.loss, x, y.transpose(0, 1))])
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
            A list of the labels

        """
        self.eval()
        if self.target == "single":
            method = "max"

        if not hasattr(self, "classes_rev"):
            self.classes_rev = [{v: k for k, v in classes.items()} for classes in self.classes]
        x = self.transform(x).to(self.device)
        if len(x.shape) == 1: x[None]
        with torch.no_grad():
            output = [self.act(o) for o in self(x)]

        predictions = [self.threshold(o) for o in output]
        self.train()
        if return_scores:
            labels = [[[(d[i.item()], s[i].item())
                        for i in torch.where(p == 1)[0]]
                       for s, p in zip(o1, prediction)]
                      for d, prediction, o1 in zip(self.classes_rev, predictions, output)]
        else:
            labels = [[[d[i.item()] for i in torch.where(p == 1)[0]] for p in prediction] for d, prediction in
                      zip(self.classes_rev, predictions)]
        return list(zip(*labels))

    def predict_dataset(self, data, batch_size=50, tr=0.5, method="hard"):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size
            tr: Threshold
            method: mcut or hard

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        predictions = []
        for b in tqdm(train_loader, ncols=100):
            predictions.extend(self.predict(b["text"]))
        return predictions

    def rebuild(self):
        """
        Internal build method.
        """
        self.loss = [type(x)().to(self.device) for x in self.loss]
        self.optimizer = type(self.optimizer)(
            filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)
