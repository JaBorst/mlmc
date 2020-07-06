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


class TextClassificationAbstractZeroShot(torch.nn.Module):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def fit(self, train,
            valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=1e-2,
            return_roc=False, return_report=False, reg=[".*"]):
        """
        Training function

        Args:
            train: MultilabelDataset used as training data
            valid: MultilabelDataset to keep track of generalization
            epochs: Number of epochs (times to iterate the train data)
            batch_size: Number of instances in one batch.
            valid_batch_size: Number of instances in one batch  of validation.
            patience: (default -1) Early Stopping Arguments. Number of epochs to wait for performance improvements before exiting the training loop.
            tolerance: (default 1e-2) Early Stopping Arguments. Minimum improvement of an epoch over the best validation loss so far.

        Returns:
            A history dictionary with the loss and the validation evaluation measurements.

        """
        import datetime
        id = str(hash(datetime.datetime.now()))[1:7]
        from ..data import SingleLabelDataset
        if isinstance(train, SingleLabelDataset) and self.target != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        validation=[]
        train_history = {"loss": []}

        assert not (type(train)== MultiOutputSingleLabelDataset and self.target=="multi"), "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(train) == MultiOutputMultiLabelDataset and self.target=="single"), "You inserted a MultiLabelDataset but chose single as target."


        best_loss = 10000000
        last_best_loss_update=0
        from ignite.metrics import Precision, Accuracy, Average
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs), ncols=100) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    y = b["labels"].to(self.device)

                    x = self.transform(b["text"]).to(self.device)
                    output = self(x)
                    l = torch.stack([l(o, t) for l,o,t in zip(self.loss,output,y.transpose(0,1))])

                    if self.aggregation == "sum":
                        l = l.sum()
                    if self.aggregation == "mean":
                        l = l.mean()
                    if hasattr(self, "regularize"):
                        l = l + self.regularize()
                    l.backward()

                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),2*self.PRECISION_DIGITS)
                    pbar.update()
                # torch.cuda.empty_cache()
                if valid is not None:
                    validation.append(self.evaluate(data=valid,
                                                    batch_size=valid_batch_size,
                                                    return_report= return_report,
                                                    return_roc=return_roc)
                                      )
                    printable = validation[-1].copy()
                    reg = [reg] if isinstance(reg, str) else reg
                    combined = "(" + ")|(".join(reg) + ")" if len(reg) > 1 else reg[0]
                    printable = {k:x for k,x in printable.items()  if re.match(combined, k) or k=="valid_loss"}
                    if return_roc==True:
                        printable["auc_roc"] = (printable["auc_roc"][0], "...")
                    if return_report==True:
                        for k in list(printable.keys()):
                            if "report" in k:
                                printable[k+"_weighted"] = printable[k]["weighted avg"]
                                printable[k+"_macro"] = printable[k]["macro avg"]
                                del printable[k]
                    else:
                        for k in list(printable.keys()):
                            if "report" in k:
                                del printable[k]

                    pbar.postfix[0].update(printable)
                    pbar.update()
            if patience > -1:
                if valid is None :
                    print("check validation loss")
                    if best_loss - average.compute().item() > tolerance:
                        print("update validation and checkoint")
                        best_loss = average.compute().item()
                        torch.save(self.state_dict(), id+"_checkpoint.pt")
                        #save states
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
                        torch.save(self.state_dict(), id+"_checkpoint.pt")
                        # save states
                        last_best_loss_update = 0
                    else:
                        last_best_loss_update += 1

                    if last_best_loss_update >= patience:
                        print("Early Stopping.")
                        break

            train_history["loss"].append(average.compute().item())
        if patience > -1:
            self.load_state_dict(torch.load(id+"_checkpoint.pt"))
        #Load best
        return{"train":train_history, "valid": validation }

    def zeroshot_fit(self,zeroshot_classes, data, **kwargs):
        history = []
        evaluation = []

        data["train"].remove(zeroshot_classes)
        data["train"].classes = {j: i for i, j in enumerate(
            [k for k in data["classes"].keys() if k not in zeroshot_classes])}

        tmp_embeddings = self.label_embeddings.clone().detach().cpu()
        tmp_adjacency = self.adjacency_param.clone().detach().cpu()

        zeroshot_adjacency = torch.stack(
            [tmp_adjacency[ind] for k, ind in data["classes"].items() if k not in zeroshot_classes])
        zeroshot_adjacency = torch.stack([zeroshot_adjacency[:, ind] for k, ind in data["classes"].items() if
                                       k not in zeroshot_classes], -1)

        zeroshot_embeddings = torch.stack(
            [tmp_embeddings[ind] for k, ind in data["classes"].items() if k not in zeroshot_classes])

        for i in range(10):
            # label remove
            self.adjacency_param = torch.nn.Parameter(torch.tensor(zeroshot_adjacency).float().to(self.device))
            self.adjacency_param.requires_grad = False
            self.label_embeddings = torch.nn.Parameter(torch.tensor(zeroshot_embeddings).to(self.device))

            history.append(self.fit(data=data["train"], **kwargs))

            # reset for evaluation
            self.adjacency_param = torch.nn.Parameter(torch.tensor(tmp_adjacency).float().to(self.device))
            self.adjacency_param.requires_grad = False
            self.label_embeddings = torch.nn.Parameter(torch.tensor(tmp_embeddings).to(self.device))

            evaluation.append(self.evaluate(data=data["valid"], return_report=True, batch_size=256))
            print("P@1:", evaluation[-1]["p@1"])
            for l in zeroshot_classes:
                print(l + ":", evaluation[-1]["report"][l])
        return {"train":history, "valid":evaluation}

    @abc.abstractmethod
    def create_labels(self, classes):
        return