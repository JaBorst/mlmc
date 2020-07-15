import torch
from pkg_resources import _by_version_descending
from tqdm import tqdm

from ..metrics.multilabel import MultiLabelReport, AUC_ROC
from ..representation import is_transformer, get
from ..representation.labels import makemultilabels

from ..representation import threshold_mcut, threshold_hard,threshold_max

from ..data import SingleLabelDataset,MultiLabelDataset,MultiOutputMultiLabelDataset,MultiOutputSingleLabelDataset
import re

import abc
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
    def zeroshot_fit(self, train, valid, epochs=1, batch_size=16, valid_batch_size=50, classes_subset=None, patience=-1, tolerance=1e-2,
            return_roc=False):
        history = []
        evaluation = []
        zeroshot_classes=list(set(valid.classes.keys()) - set(train.classes.keys()))
        print("Found Zero-shot Classes: ", str(zeroshot_classes))

        import datetime
        id = str(hash(datetime.datetime.now()))[1:7]
        from ..data import SingleLabelDataset
        if isinstance(train, SingleLabelDataset) and self.target != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        validation = []
        train_history = {"loss": []}

        assert not (type(
            train) == SingleLabelDataset and self.target == "multi"), "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(
            train) == MultiLabelDataset and self.target == "single"), "You inserted a MultiLabelDataset but chose single as target."

        best_loss = 10000000
        last_best_loss_update = 0
        from ignite.metrics import Average
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            self.create_labels(train.classes)
            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                for i, b in enumerate(train_loader):

                    self.optimizer.zero_grad()
                    y = b["labels"].to(self.device)

                    x = self.transform(b["text"])
                    output = self(x)
                    if hasattr(self, "regularize"):
                        l = self.loss(output, y) + self.regularize()
                    else:
                        l = self.loss(output, y)
                    if self.use_amp:
                        with amp.scale_loss(l, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        l.backward()

                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(), 2 * self.PRECISION_DIGITS)
                    pbar.update()
                # EVALUATION
                self.create_labels(valid.classes)
                GZSL = self.evaluate(data=valid,
                                     batch_size=valid_batch_size,
                                     return_report=True,
                                     return_roc=return_roc)
                from copy import deepcopy
                zsl_data = deepcopy(valid)
                zsl_data.reduce(dict(zip(zeroshot_classes,range(len(zeroshot_classes)))))
                self.create_labels(zsl_data.classes)
                ZSL = self.evaluate(data=zsl_data,
                                    batch_size=valid_batch_size,
                                    return_report=True,
                                    return_roc=return_roc)

                nsl_data = deepcopy(valid)
                nsl_data.reduce(train.classes)
                self.create_labels(train.classes)
                NSL = self.evaluate(data=nsl_data,
                                    batch_size=valid_batch_size,
                                    return_report=True,
                                    return_roc=return_roc)

                printable = {
                    "gzsl": {
                        "overall": {"micro": GZSL["report"]["micro avg"],
                                    "macro": GZSL["report"]["macro avg"]},
                        "labels": {
                            x: GZSL["report"][x] for x in zeroshot_classes
                        },
                    },
                    "zsl": {
                        "overall": {"micro": ZSL["report"]["micro avg"],
                                    "macro": ZSL["report"]["macro avg"]},
                        "labels": {x: ZSL["report"][x] for x in zeroshot_classes}
                    },
                    "nsl":{
                        "overall": {"micro": NSL["report"]["micro avg"],
                                    "macro": NSL["report"]["macro avg"]}
                    }
                }

                pbar.postfix[0].update({"gzsl_valid_loss": GZSL["valid_loss"],
                                        "zsl_valid_loss":ZSL["valid_loss"],
                                        "nsl_valid_loss": NSL["valid_loss"]})
                pbar.update()
                print("")
                print("\nGZSL:"+"\n\t"+"\n\t".join([b + ": "+str(s) for b,s in printable["gzsl"].items()]) )
                print("ZSL:"+"\n\t"+"\n\t".join([b + ": "+str(s) for b,s in printable["zsl"].items()]) )
                print("NSL:"+"\n\t"+"\n\t".join([b + ": "+str(s) for b,s in printable["nsl"].items()]) )


                validation.append({"gzsl":GZSL, "zsl":ZSL, "nsl":NSL})

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

        return {"train":train_history, "valid":validation}

    @abc.abstractmethod
    def create_labels(self, classes):
        """This method has to be implemented by the model to ensure that the representation of the labels are created in the desired manner."""
        return