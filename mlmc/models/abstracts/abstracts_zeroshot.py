import torch
from tqdm import tqdm
from abc import abstractmethod
from ...data import MultiLabelDataset, SingleLabelDataset

from copy import deepcopy

try:
    from apex import amp
except:
    pass

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
        if self.target == "multi":
            printable = {
                "gzsl": {
                    "overall": {"micro": GZSL["multilabel_report"]["micro avg"],
                                "macro": GZSL["multilabel_report"]["macro avg"]},
                    "labels": {
                        x: GZSL["multilabel_report"][x] for x in zeroshot_classes
                    },
                },
                "zsl": {
                    "overall": {"micro": ZSL["multilabel_report"]["micro avg"],
                                "macro": ZSL["multilabel_report"]["macro avg"]},
                    "labels": {x: ZSL["multilabel_report"][x] for x in zeroshot_classes}
                },
                "nsl": {
                    "overall": {"micro": NSL["multilabel_report"]["micro avg"],
                                "macro": NSL["multilabel_report"]["macro avg"]}
                }
            }
        else:
            printable = {
                "gzsl": {
                    "overall": {"accuracy": GZSL["accuracy"],
                                "macro": GZSL["report"]["macro avg"]},
                    "labels": {
                        x: GZSL["report"][x] for x in zeroshot_classes
                    },
                },
                "zsl": {
                    "overall": {"accuracy": ZSL["accuracy"],
                                "macro": ZSL["singlelabel_report"]["macro avg"]},
                    "labels": {x: ZSL["report"][x] for x in zeroshot_classes}
                },
                "nsl": {
                    "overall": {"accuracy": NSL["accuracy"],
                                "macro": NSL["singlelabel_report"]["macro avg"]}
                }
            }
        return printable

    def _zeroshot_fit(self,*args, **kwargs):
        return self.zeroshot_fit_sacred(_run=None, *args,**kwargs)

    def zeroshot_fit_sacred(self, data, epochs=10, batch_size=16, _run=None, metrics=None, callbacks=None):
        histories = {"train": [], "gzsl": [], "zsl": [], "nsl": []}
        self._config["trained_classes"].extend(list(data["train"].classes.keys()))
        self._config["trained_classes"] = list(set(self._config["trained_classes"]))
        for i in range(epochs):
            history = self.fit(data["train"],
                batch_size=batch_size, epochs=1, metrics=metrics, callbacks=callbacks)
            if _run is not None: _run.log_scalar("train_loss", history["train"]["loss"][0], i)

            self.create_labels(data["valid_gzsl"].classes)
            gzsl_loss, GZSL = self.evaluate(data["valid_gzsl"], batch_size=batch_size, metrics=metrics,_fit=True)
            if _run is not None: GZSL.log_sacred(_run, i, "gzsl")
            GZSL_comp = GZSL.compute()
            histories["gzsl"].append({"gzsl_loss": gzsl_loss})
            histories["gzsl"][-1].update(GZSL_comp)

            self.create_labels(data["valid_zsl"].classes)
            zsl_loss, ZSL = self.evaluate(data["valid_zsl"], batch_size=batch_size, metrics=metrics,_fit=True)
            if _run is not None: ZSL.log_sacred(_run, i, "zsl")
            ZSL_comp = ZSL.compute()
            histories["zsl"].append({"zsl_loss": zsl_loss})
            histories["zsl"][-1].update(ZSL_comp)

            self.create_labels(data["valid_nsl"].classes)
            nsl_loss, NSL = self.evaluate(data["valid_nsl"], batch_size=batch_size, metrics=metrics,_fit=True)
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
        gzsl_loss, GZSL = self.evaluate(data["test_gzsl"], batch_size=batch_size)
        if _run is not None: GZSL.log_sacred(_run, epochs, "gzsl")

        self.create_labels(data["test_zsl"].classes)
        zsl_loss, ZSL = self.evaluate(data["test_zsl"], batch_size=batch_size)
        if _run is not None: ZSL.log_sacred(_run, epochs, "zsl")

        self.create_labels(data["test_nsl"].classes)
        nsl_loss, NSL = self.evaluate(data["test_nsl"], batch_size=batch_size)
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
        self._zeroshot_ind = torch.LongTensor([1 if x[0] in self._trained_classes else 0 for x in l])
        self._mixed_shot = not (self._zeroshot_ind.sum() == 0 or self._zeroshot_ind.sum() == self._zeroshot_ind.shape[
            0]).item()  # maybe obsolete?

