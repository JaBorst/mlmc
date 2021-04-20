import torch
from .abstracts import TextClassificationAbstract
from ignite.metrics import Average
from mlmc.metrics.precisionk import Accuracy
from ...data import EntailmentDataset


class EncoderAbstract(TextClassificationAbstract):
    def __init__(self, sformatter = lambda x: f"This is about {x}", label_length=15, *args, **kwargs):
        super(EncoderAbstract, self).__init__(*args, **kwargs)
        self._config["sformatter"] = sformatter
        self._config["label_length"] = label_length
        self._all_compare = True
    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

    def set_sformat(self, c):
        assert callable(c)
        self._config["sformatter"] = c

    def transform(self,x, max_length=400, reshape=False, device=None):
        if device is None:
            device=self.device

        if self._all_compare:
            label = list(self.classes) * len(x)
            text = [s for s in x for _ in range(len(self.classes))]
        else:
            label = list(self.classes)
            text = x
        tok = self.tokenizer.tokenizer(text, label, return_tensors="pt", add_special_tokens=True, padding=True,
                                       truncation=True,
                                       max_length=self.max_len)

        if reshape:
            tok["input_ids"]= tok["input_ids"].reshape((len(x), len(self.classes), -1)).to(device)
            tok["attention_mask"] = tok["attention_mask"].reshape((len(x), len(self.classes), -1)).to(device)
        else:
            tok["input_ids"]= tok["input_ids"].to(device)
            tok["attention_mask"] = tok["attention_mask"].to(device)
        return tok

    def single(self):
        self._config["target"] = "single"
        self.target = "single"
        self.set_threshold("max")
        self.activation = lambda x: x
        self.loss = torch.nn.CrossEntropyLoss()
        # self.build()

    def multi(self):
        self._config["target"] = "multi"
        self.target = "multi"
        self.set_threshold("mcut")
        self.activation = lambda x: x
        # self.loss = torch.nn.MSELoss()
        # self.build()

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

    # def _epoch(self, train, sub_batch_size=32, pbar=None):
    #     """Implementing a subbatching loop"""
    #     average = Average()
    #     for i, b in enumerate(train):
    #         x = self.transform(b["text"], device="cpu")
    #         y = b["labels"].reshape((x["input_ids"].shape[0],))
    #         for ind in range(0, sub_batch_size, y.shape[0], ):
    #             self.optimizer.zero_grad()
    #             self._all_compare = False
    #             l, _ = self._step(
    #                 {k:v[ind:(ind+sub_batch_size)].to(self.device) for k, v in x.items()},
    #                 y[ind:(ind+sub_batch_size)].to(self.device)
    #             )
    #             self._all_compare = True
    #             l.backward()
    #             self.optimizer.step()
    #             average.update(l.item())
    #
    #         if pbar is not None:
    #             pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
    #             pbar.update()
    #     return average.compute().item()

    def pretrain_entailment(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, callbacks=None, lr_schedule=None, lr_param={}):
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

        from tqdm import tqdm
        if callbacks is None:
            callbacks = []
        import datetime
        id = str(hash(datetime.datetime.now()))[1:7]

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
                loss = self._entailment_epoch(train_loader, pbar=pbar)
                if lr_schedule is not None: scheduler.step()
                self.train_history["loss"].append(loss)

                # Validation if available
                if valid is not None:
                    valid_loss, result_metrics = self._entailment_evaluate(
                        data=valid,
                        batch_size=valid_batch_size,
                        metrics=[Accuracy()],
                        _fit=True)

                    valid_loss_dict = {"valid_loss": valid_loss}
                    valid_loss_dict.update(result_metrics.compute())
                    self.validation.append(valid_loss_dict)

                    printables = {"valid_loss": valid_loss}
                    printables.update(result_metrics.print())
                    pbar.postfix[0].update(printables)
                    pbar.update()

            # Callbacks
            self._callback_epoch_end(callbacks)

        self._callback_train_end(callbacks)

        from copy import copy
        return_copy = {"train": copy(self.train_history), "valid": copy(self.validation)}
        return return_copy

    def _entailment_epoch(self, train, pbar=None):
        """Combining into training loop"""
        average = Average()
        self._all_compare = False
        for i, b in enumerate(train):
            self.optimizer.zero_grad()
            self.create_labels(b["x2"])
            l, _ = self._entailment_step(x=self.transform(b["x1"]), y=b["labels"].to(self.device))
            l.backward()
            self.optimizer.step()
            average.update(l.item())

            if pbar is not None:
                pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                pbar.update()
        self._all_compare = True
        return average.compute().item()

    def _entailment_step(self, x, y):
        """
        This method gets input and output for of one batch and calculates output and predictions
        Args:
            x: input tensor
            y: tensor of truth indices

        Returns:
            loss, output: loss tensor, and the raw prediction output of the network
        """
        output = self(x)
        l = self._loss(output, y.float())
        l = self._regularize(l)
        return l, output


    def _entailment_evaluate(self, data, batch_size=50,  metrics=[Accuracy()], _fit=False):
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
        self._all_compare = False
        initialized_metrics = self._init_metrics(metrics)
        average = Average()


        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                self.create_labels(b["x2"])
                l, output = self._entailment_step(x=self.transform(b["x1"]), y=b["labels"].to(self.device))
                output = self.act(output).cpu()
                pred = self._threshold_fct(output)
                average.update(l.item())
                initialized_metrics.update_metrics((output, y, pred))

        self.train()

        if _fit:
            return average.compute().item(), initialized_metrics
        else:
            return average.compute().item(), initialized_metrics.compute()

    def pretrain_mnli(self, *args, **kwargs):
        from mlmc_lab.mlmc_experimental.data.data_loaders import load_mnli
        data, classes = load_mnli()
        classes["contradiction"]=0
        train = EntailmentDataset(x1=data["train_x1"], x2=data["train_x2"], labels=data["train_y"],
                                            classes=classes)
        test = EntailmentDataset(x1=data["test_x1"], x2=data["test_x2"], labels=data["test_y"],
                                           classes=classes)
        history = self.pretrain_entailment(train, valid=test, *args, **kwargs)
        self._all_compare = True

        return history, None

    def pretrain_snli(self, *args, **kwargs):
        from mlmc_lab.mlmc_experimental.data.data_loaders import load_snli
        data, classes = load_snli()
        classes["contradiction"]=0
        train = EntailmentDataset(x1=data["train_x1"],
                                            x2=data["train_x2"],
                                            labels=data["train_y"],
                                            classes=classes)
        valid = EntailmentDataset(x1=data["valid_x1"], x2=data["valid_x2"], labels=data["valid_y"],
                                           classes=classes)
        test = EntailmentDataset(x1=data["test_x1"], x2=data["test_x2"], labels=data["test_y"],
                                           classes=classes)
        history = self.pretrain_entailment(train, valid=valid, *args, **kwargs)
        loss, evaluation = self._entailment_evaluate(test)
        evaluation["loss"] = loss
        return history, evaluation