import torch

import mlmc.data.datasets
from mlmc.models.abstracts.abstracts import TextClassificationAbstract
import mlmc
from ignite.metrics import Average
from mlmc.metrics.precisionk import Accuracy



class SentenceTextClassificationAbstract(TextClassificationAbstract):
    """
    Extending the base class with functionality regarding the sentence-embedding approach and zeroshot capabilities
    """
    def __init__(self, sformatter = lambda x: f"This is about {x}", label_len=45, *args, **kwargs):
        """
        Additional Arguments for sentence Embedder
        Args:
            sformatter: Formatting the label. A callable that takes and returns a string. You can modify the label representation
            label_len:  Setting the maximum token lenght of the label embeddings. This is mainly important for pretraining MNLI or STS
            *args: See mlmc.models.abstract.abstract
            **kwargs:  See mlmc.models.abstract.abstract
        """
        super(SentenceTextClassificationAbstract, self).__init__(*args, **kwargs)
        self._config["sformatter"] = sformatter
        self._config["label_len"] = label_len
        self._all_compare = True

    def set_sformatter(self, c):
        """
        Setter for the label sformatter
        Args:
            c: callable that takes and returns a string

        Returns:

        """
        assert callable(c)
        self._config["sformatter"] = c

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

    def transform(self,x: str, max_length=None) ->dict:
        """Sentence based transformer returns all language model output in a dict
        Args:
            x: string
        """
        if max_length is None:
            max_length = self._config["max_len"]
        return {k:v.to(self.device) for k, v in self.tokenizer.tokenizer(x, padding=True, max_length=max_length, truncation=True,  return_tensors='pt').items()}

    def label_embed(self, x):
        """
        Label embedder in this instance uses the same transformation as the input
        Args:
            x:

        Returns:

        """
        return self.transform([self._config["sformatter"](l) for l in list(x)],max_length=self._config["label_len"] )

    def create_labels(self, classes: dict):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        self.classes = classes
        self.n_classes = len(classes)
        self._config["classes"] = classes
        self._config["n_classes"] = self.n_classes

        if self.n_classes != 0: # To ensure we can initialize the model without specifying classes
            # r = self.label_embed(self.classes)
            # self.label_dict = r["input_ids"]
            # self.label_att =  r["attention_mask"]
            self.label_dict = self.label_embed(self.classes)
            # self.label_dict["token_type_ids"][:] = self.label_dict["attention_mask"]


    def _metric_sim(self, x, y, m=None):
        """
        Helper function for cosine similarity of tensors. Also possible to train a metric tensor
        Args:
            x:
            y:
            m: metric tensor if exists

        Returns:

        """
        m = m if m is not None else torch.eye(x.shape[-1]).to(x.device)
        if self._all_compare:
            x = x / torch.sqrt((x * torch.matmul(x, m.t())).sum(-1, keepdims=True))
            y = y / torch.sqrt((y * torch.matmul(y, m.t())).sum(-1, keepdims=True))
            return (torch.matmul(x, m.t())[:,None] * y).sum(-1)
        else:
            assert x.shape[0] == y.shape[0], "For 1 vs 1 comaprison the tensors must have equal batch size"
            x = x / torch.sqrt((x * torch.matmul(x, m.t())).sum(-1, keepdims=True))
            y = y / torch.sqrt((y * torch.matmul(y, m.t())).sum(-1, keepdims=True))
            return (torch.matmul(x, m.t()) * y).sum(-1)

    def pretrain_entailment(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, callbacks=None, lr_schedule=None, lr_param={}, warmup=None):
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

        self._config["entailment_classes"] = train.classes
        self.validation = []
        self.train_history = {"loss": []}
        import pytorch_warmup as pwarmup
        if lr_schedule is not None:
            scheduler = lr_schedule(self.optimizer, **lr_param)
        warmup_schedule=None
        if warmup is not None:
            warmup_schedule = pwarmup.LinearWarmup(self.optimizer, warmup_period=int(len(train)*warmup / batch_size))

        for e in range(epochs):
            self._callback_epoch_start(callbacks)

            # An epoch
            losses = {"loss": str(0.)}
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                loss = self._entailment_epoch(train_loader, pbar=pbar, schedule=warmup_schedule)
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


    def pretrain_entailment(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, callbacks=None, lr_schedule=None, lr_param={}, warmup=None):
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

        # self._config["entailment_classes"] = train.classes
        self.validation = []
        self.train_history = {"loss": []}
        import pytorch_warmup as pwarmup
        if lr_schedule is not None:
            scheduler = lr_schedule(self.optimizer, **lr_param)
        warmup_schedule=None
        if warmup is not None:
            warmup_schedule = pwarmup.LinearWarmup(self.optimizer, warmup_period=int(len(train)*warmup / batch_size))

        for e in range(epochs):
            self._callback_epoch_start(callbacks)

            # An epoch
            losses = {"loss": str(0.)}
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                loss = self._entailment_epoch(train_loader, pbar=pbar, schedule=warmup_schedule)
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


    def _entailment_epoch(self, train, pbar=None, schedule=None):
        """Combining into training loop"""
        average = Average()
        self._all_compare = False
        for i, b in enumerate(train):

            self.optimizer.zero_grad()
            if schedule is not None:
                schedule.dampen()
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
        self._all_compare=False
        output = self(x)
        l = self._loss(output, y)
        l = self._regularize(l)
        self._all_compare=True
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

    def pretrain_mnli(self, contradiction = -1,binary=True, *args, **kwargs):
        from mlmc_lab.mlmc_experimental.data.data_loaders import load_mnli
        data, classes = load_mnli(binary)
        classes["contradiction"] = contradiction
        train = mlmc.data.datasets.EntailmentDataset(x1=data["train_x1"], x2=data["train_x2"], labels=data["train_y"],
                                                     classes=classes)
        classes["contradiction"] = contradiction
        test = mlmc.data.datasets.EntailmentDataset(x1=data["test_x1"], x2=data["test_x2"], labels=data["test_y"],
                                                    classes=classes)
        history = self.pretrain_entailment(train, valid=test, *args, **kwargs)
        self._all_compare = True

        return history, None

    def pretrain_snli(self, contradiction=-1, binary=True, *args, **kwargs):
        from mlmc_lab.mlmc_experimental.data.data_loaders import load_snli
        data, classes = load_snli(binary)
        classes["contradiction"]=contradiction
        train = mlmc.data.datasets.EntailmentDataset(x1=data["train_x1"],
                                                     x2=data["train_x2"],
                                                     labels=data["train_y"],
                                                     classes=classes)
        valid = mlmc.data.datasets.EntailmentDataset(x1=data["valid_x1"], x2=data["valid_x2"], labels=data["valid_y"],
                                                     classes=classes)
        test = mlmc.data.datasets.EntailmentDataset(x1=data["test_x1"], x2=data["test_x2"], labels=data["test_y"],
                                                    classes=classes)
        history = self.pretrain_entailment(train, valid=valid, *args, **kwargs)
        loss, evaluation = self._entailment_evaluate(test)
        evaluation["loss"] = loss
        return history, evaluation

    def pretrain_sts(self,batch_size=12, datasets=None, steps=600, eval_every=100, log_mlflow=False):
        c = 0
        from ...data.data_loaders_similarity import load_sts
        from tqdm import tqdm
        data = load_sts()
        epochs = int(steps/len(data))+1

        for e in range(epochs):
            train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
            with tqdm(postfix=[{}], desc="STS", ncols=100, total=steps) as pbar:
                average = Average()
                for i, b in enumerate(train_loader):

                    if c % eval_every == 0:
                        if datasets is not None:
                            for d in datasets:
                                test_data = mlmc.data.get(d)
                                if d == "rcv1" or d == "amazonfull":
                                    test_data["test"] = mlmc.data.sampler(test_data["test"], absolute=15000)
                                if mlmc.data.is_multilabel(test_data["train"]):
                                    self.multi()
                                    # self.set_sformatter(SFORMATTER[d])
                                    self.create_labels(test_data["test"].classes)
                                    _, ev = self.evaluate(test_data["test"], _fit=True, batch_size=32)
                                    if log_mlflow: ev.log_mlflow(c, prefix=d)
                                    print(f"{d}:\n", ev.print())
                                else:
                                    self.single()
                                    # self.set_sformatter(SFORMATTER[d])
                                    self.create_labels(test_data["test"].classes)
                                    _, ev = self.evaluate(test_data["test"], _fit=True, batch_size=32)
                                    if log_mlflow: ev.log_mlflow(c, prefix=d)
                                    print(f"{d}:\n", ev.print())

                        # reset
                        self.sts()

                    self.optimizer.zero_grad()
                    self.create_labels(b["x2"])
                    o = self(x=self.transform(b["x1"])).diag()  # ,
                    y = b["labels"].to(self.device)# / 5 * 2 - 1
                    l = self._loss(o, y=y)
                    l.backward()
                    self.optimizer.step()
                    average.update(l.item())

                    c += 1
                    if pbar is not None:
                        pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                        pbar.update()
                    if c==steps:
                        break

                if c == steps:
                    break
    def single(self):
        """Helper function to set model into default single label mode"""
        self._config["target"] = "single"
        self.set_threshold("max")
        self.set_activation( lambda x: x)

    def multi(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)

    def sts(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = None
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)
        from ...loss import RelativeRankingLoss
        # self.set_loss(RelativeRankingLoss(0.5))
        self.set_loss(RelativeRankingLoss(0.5))