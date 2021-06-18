import torch
from tqdm import tqdm
from abc import abstractmethod
from ...data import MultiLabelDataset, SingleLabelDataset
from ...data.datasets import EntailmentDataset

from copy import deepcopy
from tqdm import tqdm

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



    def _entail_forward(self, x1, x2):
        self.label_embedding = x2
        return self.forward(x1).diag()

    def entailment_pretrain(self, data, valid = None, epochs=10, batch_size=16):
        train_history = {"loss": []}
        for e in range(epochs):
            # An epoch
            losses = {"loss": str(0.)}
            dl = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
            with tqdm(dl,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:


                from ignite.metrics import Average
                average = Average()
                for b in dl:
                    self.zero_grad()
                    scores = self._entail_forward(self.transform(b["x1"]).to(self.device),
                                                  self.transform(b["x2"]).to(self.device))
                    l = self.loss(scores, b["labels"].to(self.device).float())
                    l.backward()
                    self.optimizer.step()
                    average.update(l.detach().item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(), 8)
                    pbar.update()
                if valid is not None:
                    validation_result = self.entailment_eval(valid,batch_size=batch_size*2)
                    pbar.postfix[0]["valid_loss"] = round(validation_result["loss"], 8)
                    pbar.postfix[0]["valid_accuracy"] = round(validation_result["accuracy"], 8)
                    pbar.update()

                train_history["loss"].append(average.compute().item())
        return {"train": train_history, "valid": validation_result}

    def entailment_eval(self, data, batch_size=16):
        self.eval()
        dl = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        from ignite.metrics import Average
        average = Average()
        accuracy = Average()
        for b in dl:
            with torch.no_grad():
                scores = self._entail_forward(self.transform(b["x1"]).to(self.device),
                                              self.transform(b["x2"]).to(self.device))
                l = self.loss(scores, b["labels"].to(self.device).float())

                for i in (b["labels"].to(self.device)==(scores>0.5)): accuracy.update(i.item())
                average.update(l.detach().item())
        return { "loss": average.compute().item(), "accuracy": accuracy.compute().item()}