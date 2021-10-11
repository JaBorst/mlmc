import torch
from tqdm import tqdm
from abc import abstractmethod
from ...data import MultiLabelDataset, SingleLabelDataset
from ...data.datasets import EntailmentDataset

from copy import deepcopy
from tqdm import tqdm
from ignite.metrics import Average
from mlmc.metrics.precisionk import Accuracy
from ...data.datasets import EntailmentDataset

try:
    from apex import amp
except:
    pass
from ...data import is_multilabel

class TextClassificationAbstractZeroShot(torch.nn.Module):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

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


    def _zeroshot_fit(self,*args, **kwargs):
        # TODO: Documentation
        return self.zeroshot_fit_sacred(_run=None, *args,**kwargs)

    def zeroshot_fit_sacred(self, data, epochs=10, batch_size=16, _run=None, metrics=None, callbacks=None, log=False):
        histories = {"train": [], "gzsl": [], "zsl": [], "nsl": []}
        if "trained_classes" not in self._config:
            self._config["trained_classes"] = []
        self._config["trained_classes"].extend(list(data["train"].classes.keys()))
        self._config["trained_classes"] = list(set(self._config["trained_classes"]))
        for i in range(epochs):
            self.create_labels(data["train"].classes)
            if is_multilabel(data["train"]):
                self.multi()
            else:
                self.single()
            history = self.fit(data["train"],
                batch_size=batch_size, epochs=1, metrics=metrics, callbacks=callbacks)
            if _run is not None: _run.log_scalar("train_loss", history["train"]["loss"][0], i)

            self.create_labels(data["valid_gzsl"].classes)
            if is_multilabel(data["valid_gzsl"]):
                self.multi()
            else:
                self.single()
            gzsl_loss, GZSL = self.evaluate(data["valid_gzsl"], batch_size=batch_size, metrics=metrics,_fit=True)
            if _run is not None: GZSL.log_sacred(_run, i, "gzsl")
            GZSL.rename({"multilabel_report":"report", "singlelabel_report":"report"})
            GZSL_comp = GZSL.compute()
            histories["gzsl"].append({"gzsl_loss": gzsl_loss})
            histories["gzsl"][-1].update(GZSL_comp)

            self.create_labels(data["valid_zsl"].classes)
            if is_multilabel(data["valid_zsl"]):
                self.multi()
            else:
                self.single()
            zsl_loss, ZSL = self.evaluate(data["valid_zsl"], batch_size=batch_size, metrics=metrics,_fit=True)
            ZSL.rename({"multilabel_report":"report", "singlelabel_report":"report"})
            if _run is not None: ZSL.log_sacred(_run, i, "zsl")
            ZSL_comp = ZSL.compute()
            histories["zsl"].append({"zsl_loss": zsl_loss})
            histories["zsl"][-1].update(ZSL_comp)

            self.create_labels(data["valid_nsl"].classes)
            if is_multilabel(data["valid_nsl"]):
                self.multi()
            else:
                self.single()
            nsl_loss, NSL = self.evaluate(data["valid_nsl"], batch_size=batch_size, metrics=metrics,_fit=True)
            NSL.rename({"multilabel_report":"report", "singlelabel_report":"report"})
            if _run is not None: NSL.log_sacred(_run, i, "nsl")
            NSL_comp = NSL.compute()
            histories["nsl"].append({"nsl_loss": zsl_loss})
            histories["nsl"][-1].update(NSL_comp)

            histories["train"].append(history["train"]["loss"][0])

            print("Validation: ")
            print(self._zeroshot_printable(GZSL_comp, ZSL_comp, NSL_comp, data.zeroshot_classes))
            print({"gzsl_valid_loss": gzsl_loss, "zsl_valid_loss": zsl_loss, "nsl_valid_loss": nsl_loss})
            print("========================================================================================\n")

        self.create_labels(data["test_gzsl"].classes)
        if is_multilabel(data["test_gzsl"]):
            self.multi()
        else:
            self.single()
        gzsl_loss, GZSL = self.evaluate(data["test_gzsl"], batch_size=batch_size,_fit=True)
        GZSL.rename({"multilabel_report": "report", "singlelabel_report": "report"})
        if _run is not None: GZSL.log_sacred(_run, epochs, "gzsl")

        self.create_labels(data["test_zsl"].classes)
        if is_multilabel(data["test_zsl"]):
            self.multi()
        else:
            self.single()
        zsl_loss, ZSL = self.evaluate(data["test_zsl"], batch_size=batch_size,_fit=True)
        ZSL.rename({"multilabel_report": "report", "singlelabel_report": "report"})
        if _run is not None: ZSL.log_sacred(_run, epochs, "zsl")

        self.create_labels(data["test_nsl"].classes)
        if is_multilabel(data["test_nsl"]):
            self.multi()
        else:
            self.single()
        nsl_loss, NSL = self.evaluate(data["test_nsl"], batch_size=batch_size,_fit=True)
        NSL.rename({"multilabel_report": "report", "singlelabel_report": "report"})
        if _run is not None: NSL.log_sacred(_run, epochs, "nsl")

        histories["test"] = {
            "gzsl": {"loss": gzsl_loss},
            "zsl": {"loss": zsl_loss},
            "nsl": {"loss": nsl_loss}
        }
        histories["test"]["gzsl"].update(GZSL.compute())
        histories["test"]["zsl"].update(ZSL.compute())
        histories["test"]["nsl"].update(NSL.compute())
        return histories

    @abstractmethod
    def label_embed(self, classes):
        return

    def create_label_dict(self):
        self.label_dict = {k:v for k,v in zip(self.classes.keys(), self.label_embed(self.classes))}

    def create_labels(self, classes, mode="glove"):
        self.classes = classes
        self.n_classes = len(classes)
        self._config["classes"] = classes
        self._config["n_classes"] = self.n_classes
        self.classes_rev = {v: k for k, v in self.classes.items()}

        if not hasattr(self, "label_dict"):
            self.create_label_dict()
        try:
            self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        except:
            self.create_label_dict()
            self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        if not hasattr(self, "_trained_classes"):
            self._trained_classes = []

        self.label_embedding = self.label_embedding.to(self.device)
        self.label_embedding_dim = self.label_embedding.shape[-1]
        l = list(classes.items())
        l.sort(key=lambda x: x[1])

        #Auxiliary values
        self._config["zeroshot_ind"] = torch.LongTensor([1 if x[0] in self._trained_classes else 0 for x in l])
        self._config["mixed_shot"] = not (self._config["zeroshot_ind"].sum() == 0 or  self._config["zeroshot_ind"].sum() == self._config["zeroshot_ind"].shape[
            0]).item()  # maybe obsolete?




    def _eval_data_list(self, datasets, log_mlflow=False, c=0, s=-1):
        from ...data import get, is_multilabel, sampler, SFORMATTER
        for d in datasets:
            test_data = get(d)
            if d == "rcv1" or d == "amazonfull":
                test_data["test"] = sampler(test_data["test"], absolute=15000)
            if s>0:
                test_data["test"] = sampler(test_data["test"], absolute=s)
            if is_multilabel(test_data["train"]):
                self.multi()
                self.set_sformatter(SFORMATTER[d])
                self.create_labels(test_data["test"].classes)
                _, ev = self.evaluate(test_data["test"], _fit=True, batch_size=32)
                if log_mlflow: ev.log_mlflow(c, prefix=d)
                print(f"\n{d}:\n", ev.print())
            else:
                self.single()
                self.set_sformatter(SFORMATTER[d])
                self.create_labels(test_data["test"].classes)
                _, ev = self.evaluate(test_data["test"], _fit=True, batch_size=32)
                if log_mlflow: ev.log_mlflow(c, prefix=d)
                print(f"\n{d}:\n", ev.print())

    def pretrain_entailment(self, train,
            valid=None, steps=1000, eval_every=100, datasets = None, epochs=1,
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
                            self._eval_data_list(datasets, log_mlflow=log_mlflow, c=c, s=sample_size)
                            self.set_sformatter(lambda x:x)
                    self.entailment()

                    self.optimizer.zero_grad()
                    self.create_labels(b["x2"])
                    l, _ = self._step(x=self.transform(b["x1"]), y=b["labels"].to(self.device))
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
                self.create_labels(b["x2"])
                l, output = self._step(x=self.transform(b["x1"]), y=b["labels"].to(self.device))
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
        from ...data.data_loaders_similarity import load_sts
        from ...data import SFORMATTER
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
