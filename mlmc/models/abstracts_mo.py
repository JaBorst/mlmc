import torch
from pkg_resources import _by_version_descending
from tqdm import tqdm

from ..metrics.multilabel import MultiLabelReport, AUC_ROC
from ..representation import is_transformer, get
from ..representation.labels import makemultilabels

from ..representation import threshold_mcut, threshold_hard,threshold_max

from ..data import SingleLabelDataset,MultiLabelDataset,MultiOutputMultiLabelDataset,MultiOutputSingleLabelDataset
import re



class TextClassificationAbstractMultiOutput(torch.nn.Module):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def __init__(self, n_outputs, aggregation="mean", weights = None, target="single", finetune=False, activation=None, loss=None, optimizer=torch.optim.Adam, optimizer_params={"lr": 5e-5}, device="cpu", **kwargs):
        """
        Abstract initializer of a Text Classification network.
        Args:
            target: single label oder multilabel mode. defined by keystrings: ("single", "multi"). Sets some basic options, like loss function, activation and
                    metrics to sensible defaults.
            activation: The activation function applied to the output. Only used for metrics and when you want to return scores in predict. (default: torch.softmax for "single", torch.sigmoid for "multi")
            loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss for "multi" and torch.nn.CrossEntropyLoss for "single")
            optimizer:  One of toch.optim (default: torch.optim.Adam)
            optimizer_params: A dictionary of optimizer parameters
            device: torch device, destination of training (cpu or cuda:0)
        """
        super(TextClassificationAbstractMultiOutput,self).__init__(**kwargs)

        assert target in ("multi", "single"), 'target must be one of "multi" or "single"'

        # Setting default values for learning mode
        self.target = target
        self.class_weights = weights
        self.n_outputs = n_outputs
        self.use_amp=False
        self.finetune = finetune

        if target == "single":
            self.activation = torch.softmax
            self.loss = torch.nn.CrossEntropyLoss
        elif self.target == "multi":
            self.activation = torch.sigmoid
            self.loss = torch.nn.BCEWithLogitsLoss

        self.aggregation = aggregation

        # If there were external arguments we will use them
        if activation is not None:
            print("This has no effect as of yet")
            # self.activation = activation
        if loss is not None:
            print("This has no effect as of yet")
            # self.loss = loss

        assert not(self.loss is torch.nn.BCEWithLogitsLoss and target=="single"),\
            "You are using BCE with a single label target. Not possible, please use torch.nn.CrossEntropy with a single label target."
        assert not(self.loss is torch.nn.CrossEntropyLoss and target=="multi"),\
            "You are using CrossEntropy with a multi label target. Not possible, please use torch.nn.BCELossWithLogits with a multi label target."

        self.device = device
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.PRECISION_DIGITS = 4

    def act(self, x):
        if "softmax" in self.activation.__name__ or "softmin" in self.activation.__name__:
            return self.activation(x,-1)
        else:
            return self.activation(x)


    def build(self):
        """
        Internal build method.
        """
        if isinstance(self.loss, type) and self.loss is not None:
            if self.class_weights is not None:
                self.loss = [self.loss(torch.FloatTensor(w).to(self.device)) for w in self.class_weights]
            else:
                self.loss = [self.loss() for _ in range(self.n_outputs)]
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)

    def evaluate(self, data, batch_size=50, return_roc=False, return_report=False):
        """
        Evaluation, return accuracy and loss and some multilabel measure

        Returns p@1, p@3, p@5, AUC, loss, Accuracy@0.5, Accuracy@mcut, ROC Values, class-wise F1, Precision and Recall.
        Args:
            data: A MultilabelDataset with the data for evaluation
            batch_size: The batch size of the evaluation loop. (Larger is often faster, but it should be small enough to fit into GPU memory. In general it can be larger than batch_size in training.
            return_roc: If True, the return dictionary contains the ROC values.
            return_report: If True, the return dictionary will contain a class wise report of F1, Precision and Recall.
        Returns:
            A dictionary with the evaluation measurements.
        """
        self.eval()  # set mode to evaluation to disable dropout
        from ignite.metrics import Average
        from ..metrics import PrecisionK, AccuracyTreshold


        assert not (type(data)== MultiOutputSingleLabelDataset and self.target=="multi"), "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(data) == MultiOutputMultiLabelDataset and self.target=="single"), "You inserted a MultiLabelDataset but chose single as target."


        multilabel_metrics = [{
            f"p@1_{i}": PrecisionK(k=1, is_multilabel=True, average=True),
            f"p@3_{i}": PrecisionK(k=3, is_multilabel=True, average=True),
            f"p@5_{i}": PrecisionK(k=5, is_multilabel=True, average=True),
            f"tr@0.5_{i}": AccuracyTreshold(trf=threshold_hard, args_dict={"tr": 0.5}, is_multilabel=True),
            f"mcut_{i}": AccuracyTreshold(trf=threshold_mcut, is_multilabel=True),
            f"auc_roc_{i}": AUC_ROC(len(self.classes), return_roc=return_roc),
        } for i in range(len(self.n_classes))]

        if return_report:
            for i, k in enumerate(multilabel_metrics):
                k[f"report_{i}"] = MultiLabelReport(self.classes, trf=threshold_hard, tr=0.5)

        if len(self.classes) <= 5:
            for i, d in enumerate(multilabel_metrics): del d[f"p@5_{i}"]
        if len(self.classes) <= 3:
            for i, d in enumerate(multilabel_metrics): del d[f"p@3_{i}"]

        singlelabel_metrics = [{
            f"accuracy_{i}":  AccuracyTreshold(threshold_max, is_multilabel=False),
            f"report_{i}": MultiLabelReport(self.classes[i],threshold_max)
        } for i in range(len(self.n_classes))]

        metrics = multilabel_metrics
        if self.target == "single":
            metrics = singlelabel_metrics

        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                x = self.transform(b["text"])
                output = self(x.to(self.device))
                l = torch.stack([l(o , t) for l, o, t in zip(self.loss,output, y.to(self.device).transpose(0, 1))])
                if self.aggregation == "sum":
                    l = l.sum()
                if self.aggregation == "mean":
                    l = l.mean()

                if hasattr(self, "regularize"):
                    l = l + self.regularize()

                average.update(l.item())
                output = [o.cpu() for o in output]
                for o, t, m in zip(output,y.transpose(0, 1), metrics):
                    for k,v in m.items():
                        if self.target == "single" and "report" in k:
                            v.update((o, torch.nn.functional.one_hot(t,o.shape[-1])))
                        else:
                            v.update((o, t))
        self.train()

        results = {"valid_loss": round(average.compute().item(), 2*self.PRECISION_DIGITS)}
        for val in metrics: results.update({k: round(v.compute(), self.PRECISION_DIGITS) if isinstance(v.compute(), float) else v.compute() for k, v in val.items()})

        return results

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


    def predict(self, x, return_scores=False, tr=0.5, method="hard"):
        """
        Classify sentence string  or a list of strings.

        Args:
            x:  A list of the text instances.
            return_scores:  If True, the labels are returned with their corresponding confidence scores
            tr: The threshold at which the labels are returned.
            method: Method of thresholding
                    (hard will cutoff at ``tr``, mcut will look for the largest distance in
                    confidence between two labels following each other and will return everything above)

        Returns:
            A list of the labels

        """
        self.eval()
        if self.target =="single":
            method="max"

        if not hasattr(self, "classes_rev"):
            self.classes_rev = [{v: k for k, v in classes.items()} for classes in self.classes]
        x = self.transform(x).to(self.device)
        with torch.no_grad(): output = [self.act(o) for o in  self(x)]
        predictions = [self.threshold(o, tr=tr, method=method) for o in output]
        self.train()
        if return_scores:
            labels = [[[(d[i.item()], s[i].item())  for i in torch.where(p == 1)[0]] for s, p in zip(o1,prediction)] for d, prediction, o1 in zip(self.classes_rev, predictions, output)]
        else:
            labels = [[[d[i.item()] for i in torch.where(p == 1)[0]] for p in prediction] for d, prediction in
                      zip(self.classes_rev, predictions)]
        return list(zip(*labels))

    def predict_dataset(self, data, batch_size=50, tr=0.5, method="hard"):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size
            tr: Threshold
            method: mcut or hard

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        predictions = []
        for b in tqdm(train_loader, ncols=100):
            predictions.extend(self.predict(b["text"], tr=tr, method=method))
        return predictions

    def threshold(self, x, tr=0.5, method="hard"):
        """
        Thresholding function for outputs of the neural network.
        So far a hard threshold ( tr=0.5, method="hard")  is supported and
        dynamic cutting (method="mcut")

        This is wrapper for functions defined in py:mod:`mlmc.representations.output_transformations`

        Args:
            x: A tensor
            tr: Threshold
            method: mcut or hard

        Returns:

        """
        if method=="hard":
            return threshold_hard(x=x, tr=tr)
        if method=="mcut":
            return threshold_mcut(x)
        if method=="max":
            return threshold_max(x)

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
        return self.tokenizer(x,self.max_len)

    def _init_input_representations(self):
        if is_transformer(self.representation):
            if not hasattr(self, "n_layers"): self.n_layers=4
            try:
                if self.n_layers == 1:
                    self.embedding, self.tokenizer = get(model=self.representation)
                    self.embeddings_dim = self.embedding(input_ids=torch.tensor([[0]]))[0].shape[-1]
                else:
                    self.embedding, self.tokenizer = get(model=self.representation, output_hidden_states=True)
                    self.embeddings_dim = \
                        torch.cat(self.embedding(input_ids=self.embedding.dummy_inputs["input_ids"])[2][-self.n_layers:], -1).shape[-1]
            except TypeError:
                print("If your using a model that does not support returning hiddenstates, set n_layers=1")
                import sys
                sys.exit()
            for param in self.embedding.parameters(): param.requires_grad = self.finetune
            if self.finetune:
                # self.embedding = torch.quantization.quantize(self.embedding, dtype=torch.float16)
                self.embedding.requires_grad=True
        else:
            self.embedding, self.tokenizer = get(self.representation, freeze=True)
            self.embeddings_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]
            for param in self.embedding.parameters(): param.requires_grad = False

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
              "Total:\t%i" % (trainable, total-trainable,total))

    def predict_restricted(self, x, restrictions):
        self.eval()
        if not hasattr(self, "classes_rev"):
            self.classes_rev = [{v: k for k, v in classes.items()} for classes in self.classes]
        x = self.transform(x).to(self.device)
        with torch.no_grad():
            output = [self.act(o) for o in self(x)]

        triple_indices = [[self.classes[i][x] for i,x in enumerate(triple)] for triple in restrictions]
        scores = [[(triple, sum([o[0][i] for o, i in zip(output, triple)]).item() ) for triple in triple_indices] for n in range(output[0].shape[0])]
        for i in range(len(scores)):
            scores[i].sort(key=lambda x: x[1], reverse=True)
        prediction = [tuple([[self.classes_rev[i][x]] for i, x in enumerate(x[0][0])]) for x in scores]
        self.train()

        return prediction