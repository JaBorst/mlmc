import torch
from ignite.metrics import Average
from tqdm import tqdm

from ...data import SingleLabelDataset, MultiLabelDataset
from ...metrics import MetricsDict
from ...representation import is_transformer, get
from ...thresholds import get as thresholdget


class TextClassificationAbstract(torch.nn.Module):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will
        load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input


    """

    def __init__(self, classes, target="multi", representation="google/bert_uncased_L-2_H-128_A-2",
                 activation=None, loss=None, optimizer=torch.optim.Adam, max_len=200, label_len=20,
                 optimizer_params=None, device="cpu", finetune=False, threshold="mcut", n_layers=1, **kwargs):
        """
        Abstract initializer of a Text Classification network.
        Args:
            classes: A dictionary of classes and ther corresponding index. This argument is mandatory.
            representation: The string of the input representation. (Supporting the full transformers list, and glove50, glove100, glove200, glove300)
            max_len: The maximum number of tokens for the input.
            label_len: The maximum number of tokens for labels.
            target: single label oder multilabel mode. defined by keystrings: ("single", "multi").
            Sets some basic options, like loss function, activation and
                    metrics to sensible defaults.
            activation: The activation function applied to the output. Only used for metrics and when you want
            to return scores in predict. (default: torch.softmax for "single", torch.sigmoid for "multi")
            loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss
            for "multi" and torch.nn.CrossEntropyLoss for "single")
            optimizer:  One of toch.optim (default: torch.optim.Adam)
            optimizer_params: A dictionary of optimizer parameters
            device: torch device, destination of training (cpu or cuda:0)
        """

        super(TextClassificationAbstract, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        if optimizer_params is None:
            optimizer_params = {"lr": 5e-5}
        assert target in ("multi", "single"), 'target must be one of "multi" or "single"'

        # Setting default values for learning mode
        self.target = target
        if target == "single":
            self.activation = torch.softmax
            self.loss = torch.nn.CrossEntropyLoss
            if threshold != "max":
                print(f"Using non-max prediction for single label target. ({threshold})")
        elif self.target == "multi":
            self.activation = torch.sigmoid
            self.loss = torch.nn.BCEWithLogitsLoss

        # If there were external arguments we will use them
        if activation is not None:
            self.activation = activation
        if loss is not None:
            self.loss = loss

        self.set_threshold(threshold)

        assert not (self.loss is torch.nn.BCEWithLogitsLoss and target == "single"), \
            "You are using BCE with a single label target. " \
            "Not possible, please use torch.nn.CrossEntropy with a single label target."
        assert not (self.loss is torch.nn.CrossEntropyLoss and target == "multi"), \
            "You are using CrossEntropy with a multi label target. " \
            "Not possible, please use torch.nn.BCELossWithLogits with a multi label target."
        self.use_amp = False
        self.finetune = finetune
        self.device = device
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.PRECISION_DIGITS = 4
        self.representation = representation
        self._init_input_representations()
        self.n_layers = n_layers
        self.max_len = max_len

        self._config = {
            "classes": self.classes,
            "target": self.target, "representation": self.representation,
            "activation":self.activation, "loss": self.loss,
            "optimizer": self.optimizer, "max_len": self.max_len,
            "optimizer_params": self.optimizer_params, "device": self.device,
            "finetune": finetune, "threshold": threshold, "n_layers": self.n_layers,
            "label_len":label_len,
        }
        self._config.update(kwargs)



    def act(self, x):
        """
        Applies activation function to output tensor.

        :param x: An input tensor
        :return: A tensor
        """
        if "softmax" in self.activation.__name__ or "softmin" in self.activation.__name__:
            return self.activation(x, -1)
        else:
            return self.activation(x)

    def set_threshold(self, name):
        """
        Sets the threshold function which will be used to as a decision threshold.

        :param name: Name of the threshold (see mlmc.thresholds.threshold_dict.keys())
        """
        if not (name == "max") and name=="single":
            Warning("You're using a non max threshold in single label mode")
        self.threshold = name
        if isinstance(name, str):
            self._threshold_fct = thresholdget(name)
        elif callable(name):
            self._threshold_fct = name
        else:
            Warning("Threshold is neither callable nor a string")

    def build(self):
        """
        Internal build method.
        """
        if isinstance(self.loss, type) and self.loss is not None:
            self.loss = self.loss().to(self.device)
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()),
                                            **self.optimizer_params)
        self.to(self.device)

    def _init_metrics(self, metrics=None):
        """
        Initializes metrics to be used. If no metrics are specified then depending on the target the default metrics
        for this target will be used. (see mlmc.metrics.metrics_config.items())

        :param metrics: Name of the metrics (see mlmc.metrics.metrics_dict.keys() and mlmc.metrics.metrics_config.keys())
        :return: A dictionary containing the initialized metrics
        """
        if metrics is None:
            metrics=f"default_{self.target}label"
        metrics = MetricsDict(metrics)
        metrics.init(self.__dict__)
        metrics.reset()
        metrics.rename({"multilabel_report": "report", "singlelabel_report": "report"})
        return metrics

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

        assert not (type(data) == SingleLabelDataset and self.target == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(data) == MultiLabelDataset and self.target == "single"), \
            "You inserted a MultiLabelDataset but chose single as target."

        initialized_metrics = self._init_metrics(metrics)
        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                l, output = self._step(x=self.transform(b["text"]), y=y.to(self.device))
                output = self.act(output).cpu()
                pred = self._threshold_fct(output)
                average.update(l.item())
                initialized_metrics.update_metrics((output, y, pred))

        self.train()
        if _fit:
            return average.compute().item(), initialized_metrics
        else:
            return average.compute().item(), initialized_metrics.compute()

    def _epoch(self, train, pbar=None):
        """Combining into training loop"""
        average = Average()
        for i, b in enumerate(train):
            self.optimizer.zero_grad()
            l, _ = self._step(x=self.transform(b["text"]), y=b["labels"].to(self.device))
            l.backward()
            self.optimizer.step()
            average.update(l.item())

            if pbar is not None:
                pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                pbar.update()
        return average.compute().item()

    def _step(self, x, y):
        """
        This method gets input and output for of one batch and calculates output and predictions
        Args:
            x: input tensor
            y: tensor of truth indices

        Returns:
            loss, output: loss tensor, and the raw prediction output of the network
        """
        output = self(x)
        # if x.shape[0] == 1 and output.shape[0] != 1:
        #     output = output[None]
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

    def _callback_epoch_end(self, callbacks):
        # TODO: Documentation
        for cb in callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(self)
    def _callback_train_end(self, callbacks):
        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_epoch_end(self)
    def _callback_epoch_start(self, callbacks):
        # TODO: Documentation
        for cb in callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(self)

    def fit(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=1e-2,
            return_roc=False, return_report=False, callbacks=None, metrics=None, lr_schedule=None, lr_param ={}):
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
        from ...data import SingleLabelDataset
        if isinstance(train, SingleLabelDataset) and self.target != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        validation = []
        train_history = {"loss": []}

        assert not (type(train) == SingleLabelDataset and self.target == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(train) == MultiLabelDataset and self.target == "single"), \
            "You inserted a MultiLabelDataset but chose single as target."

        best_loss = 10000000
        last_best_loss_update = 0
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
                train_history["loss"].append(loss)

                # Validation if available
                if valid is not None:
                    valid_loss, result_metrics = self.evaluate(
                        data=valid,
                        batch_size=valid_batch_size,
                        metrics=metrics,
                        _fit=True)

                    valid_loss_dict= {"valid_loss": valid_loss}
                    valid_loss_dict.update(result_metrics.compute())
                    validation.append(valid_loss_dict)

                    printables= {"valid_loss": valid_loss}
                    printables.update(result_metrics.print())
                    pbar.postfix[0].update(printables)
                    pbar.update()

            # Callbacks
            self._callback_epoch_end(callbacks)


            # Early Stopping
            if patience > -1:
                if valid is None:
                    print("check validation loss")
                    if best_loss - loss > tolerance:
                        print("update validation and checkoint")
                        best_loss = loss
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


        if patience > -1:
            self.load_state_dict(torch.load(id + "_checkpoint.pt"))
        # Load best
        return {"train": train_history, "valid": validation}

    def predict(self, x, return_scores=False):
        """
        Classify sentence string  or a list of strings.

        Args:
            x:  A list of the text instances.
            return_scores:  If True, the labels are returned with their corresponding confidence scores
        Returns:
            A list of the labels

        """
        self.eval()
        assert not (self.target == "single" and self.threshold != "max"), \
            "You are running single target mode and predicting not in max mode."

        if not hasattr(self, "classes_rev") or (list(self.classes_rev.values())[0] not in self.classes.keys()):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
        prediction = self._threshold_fct(output)
        self.train()
        if return_scores:
            return [[(self.classes_rev[i.item()], s[i].item())
                     for i in torch.where(p == 1)[0]] for s, p in zip(output, prediction)]
        return [[self.classes_rev[i.item()] for i in torch.where(p == 1)[0]] for p in prediction]

    def predict_dataset(self, data, batch_size=50, tr=0.5):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        predictions = []
        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        for b in tqdm(train_loader, ncols=100):
            predictions.extend(self.predict(b["text"]))
        del self.classes_rev
        return predictions

    def run(self, x):
        """
        Transforms textual input into network format and applies activation function.

        :param x: A string or a list of strings
        :return: A tensor
        """
        x = [x] if isinstance(x, str) else x
        x = self.transform(x)
        with torch.no_grad(): output = self.act(self(x)).cpu()
        return output

    def scores_dataset(self, data, return_truth=False, batch_size=50):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        scores = []
        truth = []
        if return_truth:
            for b in tqdm(train_loader, ncols=100):
                scores.append(self.run(b["text"]))
                truth.append(b["labels"])
            scores = torch.cat(scores)
            truth = torch.cat(truth)
            return scores, truth
        else:
            for b in tqdm(train_loader, ncols=100):
                scores.append(self.run(b["text"]))
            scores = torch.cat(scores)
            return scores

    def transform(self, x):
        """
        A standard transformation function from text to network input format

        The function looks for the tokenizer attribute. If it doesn't exist the transform function has to
        be implemented in the child class

        Args:
            x: A list of text

        Returns:
            A tensor in the network input format.

        """
        assert hasattr(self, 'tokenizer'), "If the model does not have a tokenizer attribute, please implement the" \
                                           "transform(self, x)  method yourself. TOkenizer can be allocated with " \
                                           "embedder, tokenizer = mlmc.helpers.get_embedding() or " \
                                           "embedder, tokenizer = mlmc.helpers.get_transformer()"
        return self.tokenizer(x, maxlen=self.max_len).to(self.device)

    def _init_input_representations(self):
        # TODO: Documentation
        if is_transformer(self.representation):
            if not hasattr(self, "n_layers"): self.n_layers = 1
            try:
                if self.n_layers == 1:
                    self.embedding, self.tokenizer = get(model=self.representation)
                    self.embeddings_dim = self.embedding(torch.tensor([[0]]))[0].shape[-1]
                else:
                    self.embedding, self.tokenizer = get(model=self.representation, output_hidden_states=True)
                    self.embeddings_dim = \
                        torch.cat(self.embedding(self.embedding.dummy_inputs["input_ids"])[2][-self.n_layers:],
                                  -1).shape[-1]
            except TypeError:
                print("If your using a model that does not support returning hiddenstates, set n_layers=1")
                import sys
                sys.exit()
            for param in self.embedding.parameters(): param.requires_grad = self.finetune
            if self.finetune:
                self.embedding.requires_grad = True
        else:
            self.embedding, self.tokenizer = get(self.representation, freeze=not self.finetune)
            self.embeddings_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]
            for param in self.embedding.parameters(): param.requires_grad = self.finetune

    def num_params(self):
        """
        Count the number of trainable parameters.

        Returns:
            The number of trainable parameters
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Parameters:\n"
              "Trainable:\t%i\n"
              "Fixed:\t%i\n"
              "-----------\n"
              "Total:\t%i" % (trainable, total - trainable, total))

    def embed_input(self, x):
        """
        Using a specified representation (language model or glove vectors) embeds an input tensor.

        :param x: Input tensor
        :return: Embedded tensor
        """
        if is_transformer(self.representation):
            if self.finetune:
                if self.n_layers == 1:
                    embeddings = self.embedding(x)[0]
                else:
                    embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
            else:
                with torch.no_grad():
                    if self.n_layers == 1:
                        embeddings = self.embedding(x)[0]
                    else:
                        embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
        else:
            if self.finetune:
                embeddings = self.embedding(x)
            else:
                with torch.no_grad():
                    embeddings = self.embedding(x)
        return embeddings

    def rebuild(self):
        """
        Internal build method.
        """
        self.loss = [type(x)().to(self.device) for x in self.loss]
        self.optimizer = type(self.optimizer)(filter(lambda p: p.requires_grad, self.parameters()),
                                              **self.optimizer_params)
        self.to(self.device)

    def set_lm(self, finetune):
        self.finetune=finetune
        for param in self.embedding.parameters(): param.requires_grad = self.finetune
        self.rebuild