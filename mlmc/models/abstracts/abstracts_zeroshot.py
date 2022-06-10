import torch

from ...metrics import Average
from ...metrics.precisionk import Accuracy


class TextClassificationAbstractZeroShot(torch.nn.Module):
    """
    This class contains  methods to train on tasks other than classification.
    This class (currently) supports training for STS, Entailment, CausalEntailment.
    """

    def _zeroshot_printable(self, GZSL, ZSL, NSL, zeroshot_classes):
        printable = {
            "gzsl": {
                "overall": {"micro": GZSL["report"]["micro avg"],
                            "macro": GZSL["report"]["macro avg"]},
                "accuracy": GZSL["accuracy"] if "accuracy" in GZSL else None,
                "labels": {
                    x: GZSL["report"][x] for x in zeroshot_classes
                },
            },
            "zsl": {
                "overall": {"micro": ZSL["report"]["micro avg"],
                            "macro": ZSL["report"]["macro avg"]},
                "accuracy": GZSL["accuracy"] if "accuracy" in GZSL else None,
                "labels": {x: ZSL["report"][x] for x in zeroshot_classes}
            },
            "nsl": {
                "overall": {"micro": NSL["report"]["micro avg"],
                            "macro": NSL["report"]["macro avg"]},
                "accuracy": GZSL["accuracy"] if "accuracy" in GZSL else None,
            }
        }
        return printable


    def _eval_data_list(self, datasets, formatters=None,  batch_size=50, log_mlflow=False, c=0, s=-1):
        """
        Evaluate a list of datasets (either single or multilabel)
        :param datasets: List of dataset names
        :param log_mlflow: If True this method calls the appropriate mlflow logging functions and logs to an activae run.
        :param c: The index of the step to log the results with (usually the epoch)
        :param s: The sample size  for each dataset. If s=-1 the whole test set will be used.
        :return:
        """
        from mlmc_lab.mlmc_experimental.data import get, SFORMATTER
        from mlmc.data import is_multilabel, sampler
        if formatters is not None:
            formatters = formatters if isinstance(formatters,list) else [formatters]
            formatters = formatters if len(formatters) == len(datasets) else formatters * len(datasets)
        for i,d in  enumerate(datasets):
            if isinstance(d, str):
                test_data = get(d)["test"]
                if d == "rcv1" or d == "amazonfull":
                    test_data = sampler(test_data, absolute=15000)
                if s > 0:
                    test_data = sampler(test_data, absolute=s)
                prefix = d
            else:
                test_data = d
                prefix = f"set_{i}"
            if formatters is not None:
                self.set_sformatter(formatters[i])
            else:
                self.set_sformatter(SFORMATTER[d])
            self.create_labels(test_data.classes)

            if is_multilabel(test_data):
                self.multi()
                _, ev = self.evaluate(test_data, _fit=True, batch_size=batch_size)
                if log_mlflow: ev.log_mlflow(c, prefix=prefix)
                print(f"\n{d}:\n", ev.print())
            else:
                self.single()
                _, ev = self.evaluate(test_data, _fit=True, batch_size=batch_size)
                if log_mlflow: ev.log_mlflow(c, prefix=prefix)
                print(f"\n{d}:\n", ev.print())

    def pretrain_entailment(self, train,
            valid=None, steps=1000, eval_every=100, datasets = None, formatters=None,
                            batch_size=16, valid_batch_size=32, callbacks=None, lr_schedule=None, lr_param={}, log_mlflow=False,
                            sample_size=-1):
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

        self.validation = []
        self.train_history = {"loss": []}

        if lr_schedule is not None:
            scheduler = lr_schedule(self.optimizer, **lr_param)
        epochs = int(steps * batch_size / len(train)) + 1
        c = 0
        with tqdm(postfix=[{}], desc="Entailment", ncols=100, total=steps) as pbar:
            average = Average()
            for e in range(epochs):
                train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
                for i, b in enumerate(train_loader):
                    if c % eval_every == 0:
                        # Validation if available
                        if valid is not None:
                            average = Average()
                            self.entailment()
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
                            print("valid: ", printables)

                        if datasets is not None:
                            self._eval_data_list(datasets, formatters, log_mlflow=log_mlflow, c=c, s=sample_size)
                            self.set_sformatter(lambda x:x)
                    self.entailment()

                    self.optimizer.zero_grad()
                    l, _ = self._step(b)
                    l.backward()
                    self.optimizer.step()
                    average.update(l.item())

                    if pbar is not None:
                        pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                        pbar.update()
                    c+=1
                    if c == steps:
                        break
                if c == steps:
                    break

        from copy import copy
        return_copy = {"train": copy(self.train_history), "valid": copy(self.validation)}
        return return_copy

    def _entailment_step(self, b):
        """
        This method gets input and output for of one batch and calculates output and predictions
        Args:
            x: input tensor
            y: tensor of truth indices

        Returns:
            loss, output: loss tensor, and the raw prediction output of the network
        """
        x = self.transform(b["x1"])
        y = b["labels"].to(self.device)
        output = self(x)
        l = self._loss(output, y)
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
        initialized_metrics = self._init_metrics(metrics)
        average = Average()


        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                l, output = self._step(b)
                output = self.act(output).cpu()
                pred = self._threshold_fct(output)
                average.update(l.item())
                initialized_metrics.update_metrics((output, y, pred))

        self.train()

        if _fit:
            return average.compute().item(), initialized_metrics
        else:
            return average.compute().item(), initialized_metrics.compute()


    def pretrain_sts(self,batch_size=12, datasets=None, steps=600, eval_every=100, log_mlflow=False):
        c = 0
        from mlmc_lab.mlmc_experimental.data.data import load_sts
        from tqdm import tqdm
        data = load_sts()
        epochs = int(steps*batch_size/len(data))+1

        with tqdm(postfix=[{}], desc="STS", ncols=100, total=steps) as pbar:
            average = Average()
            for e in range(epochs):
                train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
                for i, b in enumerate(train_loader):
                    if c % eval_every == 0:
                        if datasets is not None:
                            self._eval_data_list(datasets, log_mlflow=log_mlflow, c=c)
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
