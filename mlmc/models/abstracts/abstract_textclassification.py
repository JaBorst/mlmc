import torch
import transformers
from ...metrics import Average
from tqdm import tqdm
from copy import copy

import mlmc.loss
from ...data import SingleLabelDataset, MultiLabelDataset
from ...data import PredictionDataset
from ...metrics import MetricsDict
from ...representation import get
from ...thresholds import get as thresholdget
from ...representation.character import makemultilabels
from ...modules.augment import Augment

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
    def __init__(self, classes, target=None, representation="google/bert_uncased_L-2_H-128_A-2",
                 activation=None, loss=None, optimizer=torch.optim.Adam, max_len=450, label_len=20,
                 optimizer_params=None, device="cpu", finetune="fixed", threshold=None,
                 word_cutoff=0.0, feature_cutoff=0.0, span_cutoff=0.0, word_noise=0.0,
                 **kwargs):
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


        if optimizer_params is None:
            optimizer_params = {"lr": 5e-5, "betas": (0.9, 0.99)}

        self.classes = classes
        self.n_classes = len(classes)
        self.use_amp = False
        self.finetune = finetune
        self.device = device
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.PRECISION_DIGITS = 4
        self.representation = representation
        self.max_len = max_len

        self._config = {
            "classes": classes,
            "n_classes": len(classes),
            "target": target,
            "representation": representation,
            "activation": activation, "loss": loss,
            "optimizer": optimizer, "max_len": max_len,
            "optimizer_params": optimizer_params, "device": device,
            "finetune": finetune, "threshold": threshold,
            "label_len": label_len,
            "word_cutoff": word_cutoff,
            "feature_cutoff": feature_cutoff,
            "span_cutoff": span_cutoff,
            "word_noise": word_noise,
        }
        self._augment = Augment( word_cutoff=word_cutoff,
                                 feature_cutoff=feature_cutoff,
                                 span_cutoff=span_cutoff,
                                 word_noise=word_noise)
        self._init_input_representations()
        self._set_entailment_classes()

        # Setting default values for learning mode
        if self._config["target"] is None:
            assert activation is not None, "Did not specify a target type from ('single', 'multi') and activation function is not set"
            assert loss is not None, "Did not specify a target type from ('single', 'multi') and loss function is not set"
        else:
            assert self._config["target"] in ("multi", "single",), 'target must be one of "multi" or "single"'
            if self._config["target"] == "single":
                self.single()
            elif   self._config["target"]=="multi":
                self.multi()
            else:
                Warning(f"Unknown target {target}. Not in ('single', 'multi')")

        # If there were external arguments we will overwrite
        if activation is not None:
            self.set_activation(activation)
        if loss is not None:
            self.set_loss(loss)
        if threshold is not None:
            self.set_threshold(threshold)

        assert not (self._config["loss"] is torch.nn.BCEWithLogitsLoss and self._config["target"] == "single"), \
            "You are using BCE with a single label target. " \
            "Not possible, please use torch.nn.CrossEntropy with a single label target."
        assert not (self._config["loss"] is torch.nn.CrossEntropyLoss and self._config["target"] == "multi"), \
            "You are using CrossEntropy with a multi label target. " \
            "Not possible, please use torch.nn.BCELossWithLogits with a multi label target."

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
        self._config["threshold"] = name
        if isinstance(name, str):
            self._threshold_fct = thresholdget(self._config["threshold"])
        elif callable(name):
            self._threshold_fct = name
        else:
            Warning("Threshold is neither callable nor a string")

    def set_activation(self, name):
        self._config["activation"] = name
        self.activation = name

    def set_loss(self, loss):
        self._config["loss"] = loss
        if isinstance(self._config["loss"], type) and self._config["loss"] is not None:
            self.loss = self._config["loss"]().to(self.device)
        else:
            self.loss = self._config["loss"].to(self.device)

    def set_device(self, device):
        self.device=device
        self = self.to(device)
        if hasattr(self, "label_dict"):
            self.label_dict = {k: v.to(self.device) for k, v in self.label_dict.items()}

    def set_optimizer(self, optimizer, optimizer_params={}):
        self._config["optimizer"] = optimizer
        self.optimizer = self._config["optimizer"](filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)

    def build(self):
        """
        Internal build method.
        """
        self._finetune_mode(self._config["finetune"])
        if isinstance(self._config["loss"], type) and self._config["loss"] is not None:
            self.loss = self._config["loss"]().to(self.device)
        else:
            self.loss = self._config["loss"].to(self.device)
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self._config["optimizer"](filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        else:
            self.optimizer = self._config["optimizer"]
        self.to(self.device)

    def _init_metrics(self, metrics=None):
        """
        Initializes metrics to be used. If no metrics are specified then depending on the target the default metrics
        for this target will be used. (see mlmc.metrics.metrics_config.items())

        :param metrics: Name of the metrics (see mlmc.metrics.metrics_dict.keys() and mlmc.metrics.metrics_config.keys())
        :return: A dictionary containing the initialized metrics
        """
        if metrics is None:
            metrics = f"default_{self._config['target']}label"
        metrics = MetricsDict(metrics)
        metrics.init(self.__dict__)
        metrics.reset()
        metrics.rename({"multilabel_report": "report", "singlelabel_report": "report"})
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

        assert not (type(data) == SingleLabelDataset and self._config["target"] == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(data) == MultiLabelDataset and self._config["target"] == "single"), \
            "You inserted a MultiLabelDataset but chose single as target."

        initialized_metrics = self._init_metrics(metrics)
        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                l, output = self._step(b)
                output = self.act(output).cpu()
                pred = self._threshold_fct(output)

                # Subset evaluation if ...
                if mask is not None:
                    output = output * mask
                    y = y * mask
                    pred = pred * mask

                average.update(l.item())
                initialized_metrics.update_metrics((output, y, pred, b["text"]))

        self.train()
        if _fit:
            return average.compute().item(), initialized_metrics
        else:
            return average.compute().item(), initialized_metrics.compute(model=self)

    def _epoch(self, train, pbar=None):
        """Combining into training loop"""
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
        x = self.transform(b["text"], b.get("hypothesis",None))
        y = b["labels"].to(self.device)
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
                cb.on_train_end(self)

    def _callback_epoch_start(self, callbacks):
        # TODO: Documentation
        for cb in callbacks:
            if hasattr(cb, "on_epoch_start"):
                cb.on_epoch_start(self)

    def fit(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=1e-2,
            return_roc=False, return_report=False, callbacks=None, metrics=None, lr_schedule=None, lr_param={}, log_mlflow=False, valid_prefix="valid"):
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
        if isinstance(train, SingleLabelDataset) and self._config["target"] != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        self.validation = []
        self.train_history = {"loss": []}

        assert not (type(train) == SingleLabelDataset and self._config["target"] == "multi"), \
            "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(train) == MultiLabelDataset and self._config["target"] == "single"), \
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
                self.train_history["loss"].append(loss)
                if log_mlflow:
                    import mlflow
                    mlflow.log_metric("loss", loss, step=e)

                # Validation if available
                if valid is not None:
                    valid_loss, result_metrics = self.evaluate(
                        data=valid,
                        batch_size=valid_batch_size,
                        metrics=metrics,
                        _fit=True)

                    if log_mlflow:
                        import mlflow
                        mlflow.log_metric(f"{valid_prefix}_loss" ,valid_loss, step=e)
                        result_metrics.log_mlflow(step=e, prefix=valid_prefix, model=self)

                    valid_loss_dict = {f"{valid_prefix}_loss": valid_loss}
                    valid_loss_dict.update(result_metrics.compute(model=self))
                    self.validation.append(valid_loss_dict)

                    printables = {f"{valid_prefix}_loss": valid_loss}
                    printables.update(result_metrics.print(model=self))
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
                        print("increment number of epochs")
                        last_best_loss_update += 1

                    if last_best_loss_update >= patience:
                        print("breaking at %i" % (patience,))
                        print("Early Stopping.")
                        break
                elif valid is not None:
                    if best_loss - self.validation[-1]["valid_loss"] > tolerance:
                        best_loss = self.validation[-1]["valid_loss"]
                        torch.save(self.state_dict(), id + "_checkpoint.pt")
                        # save states
                        last_best_loss_update = 0
                    else:
                        last_best_loss_update += 1

                    if last_best_loss_update >= patience:
                        print("Early Stopping.")
                        break

        self._callback_train_end(callbacks)
        if patience > -1:
            self.load_state_dict(torch.load(id + "_checkpoint.pt"))
        # Load best
        from copy import copy
        return_copy = {"train": copy(self.train_history), "valid": copy(self.validation)}
        return return_copy

    def predict(self, x, return_scores=False):
        """
        Classify sentence string  or a list of strings.

        Args:
            x:  A list of the text instances.
            return_scores:  If True, the labels are returned with their corresponding confidence scores and prediction mask
        Returns:
            A list of the labels or a tuple of (labels, scores, mask) if return_scores=True

        """
        self.eval()
        output = self.scores(x)
        prediction = self._threshold_fct(output).cpu()
        self.train()

        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self._config["classes"].items()}
        labels = [[self.classes_rev[i.item()] for i in torch.where(p == 1)[0]] for p in prediction]

        if return_scores:
            return labels, output.cpu(), prediction.cpu()==1
        return labels

    def scores(self, x):
        """
        Returns 2D tensor with length of x and number of labels as shape: (N, L)
        Args:
            x:

        Returns:

        """
        self.eval()
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
        self.train()
        return output

    def predict_dataset(self, data, batch_size=50, return_scores=False):
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
            predictions.extend(self.predict(b["text"], return_scores=return_scores))
        del self.classes_rev
        return predictions

    def predict_batch(self, data, batch_size=50, return_scores=False):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(PredictionDataset(x=data), batch_size=batch_size, shuffle=False)
        predictions = []
        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        for b in tqdm(train_loader, ncols=100):
            predictions.extend(self.predict(b["text"], return_scores=return_scores))
        del self.classes_rev
        if return_scores:
            labels = sum([predictions[x] for x in list(range(0, len(predictions), 3))],[])
            scores = torch.cat([predictions[x] for x in list(range(1, len(predictions) + 1, 3))], dim=0)
            bools = torch.cat([predictions[x] for x in list(range(2, len(predictions), 3))], dim=0)
            return labels, scores, bools
        else:
            labels = sum([predictions[x] for x in list(range(0, len(predictions)))], [])
            return labels

    def run(self, x):
        """
        Transforms textual input into network format and applies activation function.

        :param x: A string or a list of strings
        :return: A tensor
        """
        self.eval()
        x = [x] if isinstance(x, str) else x
        x = self.transform(x)
        with torch.no_grad(): output = self.act(self(x)).cpu()
        self.train()
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
        self.eval()
        if return_truth:
            for b in tqdm(train_loader, ncols=100):
                scores.append(self.run(b["text"]))
                truth.append(b["labels"])
            scores = torch.cat(scores)
            truth = torch.cat(truth)
            self.train()
            return scores, truth
        else:
            for b in tqdm(train_loader, ncols=100):
                scores.append(self.run(b["text"]))
            scores = torch.cat(scores)
            self.train()
            return scores

    def transform(self, x, h=None, max_length=None) -> dict:
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
                                           "embedder, tokenizer = mlmc.representation.get_embedding() or " \
                                           "embedder, tokenizer = mlmc.representation.get_transformer()"
        if max_length is None:
            max_length = self._config["max_len"]
        return {k: v.to(self.device) for k, v in
                self.tokenizer(x, h, padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt').items()}

    def _init_input_representations(self):
        # TODO: Documentation
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
        self._finetune_mode(self._config["finetune"])

    def _finetune_mode(self, mode):
        modes = ("all", "bias", "random", "fixed")
        assert mode in modes
        if mode == "all":
            for x in self.parameters(recurse=True):
                x.requires_grad=True
        elif mode == "bias":
            for name, param in self.named_parameters():
                if "bias" in name.lower():  # isinstance(layer, torch.nn.Linear):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif mode == "random":
            for name, param in self.named_parameters():
                param.requires_grad = bool(torch.rand(1).round())
        elif mode == "fixed":
            self.embedding.requires_grad = False
            for param in self.embedding.parameters(): param.requires_grad = False
        else:
            raise ValueError(f"Finetuning mode [{mode}] unknown. Must be one of {modes}")

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
        if self._config["finetune"] != "fixed":
            if self.n_layers == 1:
                embeddings = self.embedding(**x)[0]
            else:
                embeddings = torch.cat(self.embedding(**x)[2][-self.n_layers:], -1)
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embeddings = self.embedding(**x)[0]
                else:
                    embeddings = torch.cat(self.embedding(**x)[2][-self.n_layers:], -1)
        embeddings = self._augment(embeddings, x["attention_mask"].unsqueeze(-1))
        return embeddings


    def _release_training_memory(self):
        import gc
        self.optimizer.zero_grad(set_to_none=True)
        otype = type(self.optimizer)
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        self.optimizer = otype

    def reset_memory(self):
        self._release_training_memory()
        self.build()

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean Pooling for sequence of embeddings, taking attention mask into account for correct averaging.
        Using the output of the language models
        Args:
            token_embeddings:
            attention_mask:

        Returns:

        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def rebuild(self):
        """
        Internal build method.
        """
        self._finetune_mode(self._config["finetune"])
        self.loss = type(self.loss)().to(self.device)
        self.optimizer = type(self.optimizer)(filter(lambda p: p.requires_grad, self.parameters()),
                                              **self.optimizer_params)
        self.to(self.device)

    def single(self, threshold="max", activation=torch.softmax, loss=torch.nn.CrossEntropyLoss, **kwargs):
        """Setting the default single label mode"""
        self._config["target"] = "single"
        self.set_threshold(threshold)
        self.set_activation(activation)
        self.set_loss(loss)
        self._config.update(**kwargs)

    def multi(self, threshold="mcut", activation=torch.sigmoid, loss=torch.nn.BCEWithLogitsLoss, **kwargs):
        """Setting the defaults for multi label mode"""
        self._config["target"] = "multi"
        self.set_threshold(threshold)
        self.set_activation(activation)
        self.set_loss(loss)
        self._config.update(**kwargs)

    def sts(self, threshold="mcut", activation=lambda x:x, loss=mlmc.loss.RelativeRankingLoss, **kwargs):
        """Helper function to set model into default sts mode"""
        self._config["target"] = "sts"
        self.set_threshold(threshold)
        self.set_activation(activation)
        self.set_loss(loss)
        self._config.update(**kwargs)

    def entailment(self, threshold="max", activation=torch.softmax, loss=torch.nn.CrossEntropyLoss, **kwargs):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "entailment"
        self.set_threshold(threshold)
        self.set_activation(activation)
        self.set_loss(loss)
        self._config.update(**kwargs)

    def _set_entailment_classes(self, classes=None):
        if classes is None:
            classes = {"contradiction": 0, "neutral": 1, "entailment": 2}
        if hasattr(self, "_entailment_classes"):
            if self.embedding.config.label2id != classes:
                raise RuntimeError("Dataset defined entailment classes are different from model set classes")
        if set(self.embedding.config.label2id.keys()) == set(classes.keys()):
            self._entailment_classes =  self.embedding.config.label2id
        else:
            self._entailment_classes =  classes

    def finetune_lm(self, file, epochs=1, batch_size=8 , valid = 0.1):
        import subprocess, pathlib, tempfile, os, sys
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = "" if self.device =="cpu" else str(self.device).split(":")[-1]
        cmd = pathlib.Path(__file__).parents[0] / "pretrain-language-model.py"
        with tempfile.TemporaryDirectory() as f:
            subprocess.call(
                [sys.executable, cmd, "--model", self.representation, "--file", str(file), "--output", str(f),
                 "--epochs", str(epochs), "--batch_size", str(batch_size), "--valid_frac", str(valid)],
                env=my_env
            )
            self.embedding = transformers.AutoModel.from_pretrained(f + "/model")

    def embed_batch(self, data, batch_size=50):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(PredictionDataset(x=data), batch_size=batch_size, shuffle=False)
        l_p, l_e, l_l = [],[],[]
        with torch.no_grad():
            for b in train_loader:
                p, (e,l) = self.embed(b["text"])
                l_p.append(p.detach().cpu())
                l_e.append(e.detach().cpu())
                l_l.append(l.detach().cpu())
        return torch.cat(l_p), (torch.cat(l_e), torch.cat(l_l))

    def embed(self, x):
        """
        Method to return input embeddings.
        ToDo: Modularize the forward to avoid code copying.
        Args:
            x: list of input texts

        Returns: a tuple of:
            A tensor of embeddings shape (b, e), where b is the number of input texts and e the embedding dimension
            A tensor of embeddings shape (l, e), where l is the number of labels and e the embedding dimension

        """
        x = self.transform(x)
        return self.forward(x, emb=True)

    def log_mlflow(self):
        import mlflow
        mlflow.log_params({k:v for k,v in self._config.items() if k not in ["classes"]})
        mlflow.log_param("model", self.__class__.__name__)

    def evaluate_word_importance(self, x, plot=False):
        text = x
        text_split = text.split(" ")
        text = [text] + [" ".join(text_split[:i] + text_split[i + 1:]) for i in range(len(text_split))]
        p, scores, _ = self.predict_batch(text, return_scores=True, batch_size=12)

        # change = (-(scores[1:]-scores[0])).relu()#[:,sbb.classes[p[0][0]]]
        change = scores[0] - scores[1:]

        if plot:
            try:
                import pandas as pd
            except:
                raise ImportError("When trying to plot the word importance, you need pandas installed")
            try:
                import matplotlib.pyplot as plt
            except:
                raise ImportError("When trying to plot the word importance, you need matplotlib installed")
            
            df = pd.DataFrame(change, columns=self.classes.keys(), index=text_split)
            fig = plt.figure(figsize=(16, 8))
            plt.plot(df.values, label=self.classes.keys())
            plt.xticks(range(len(text_split)), labels=text_split, rotation=90)
            plt.grid("True")
            plt.tight_layout()
            plt.legend()
            plt.show()
        return change