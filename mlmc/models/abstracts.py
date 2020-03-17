from tqdm import tqdm
import torch
from ignite.metrics import Precision, Accuracy, Average
from ..metrics.multilabel import MultiLabelReport,AUC_ROC
from ..representation import is_transformer,get
from ..representation.labels import makemultilabels



class TextClassificationAbstract(torch.nn.Module):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def __init__(self, loss=torch.nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam, optimizer_params = {"lr": 5e-5}, device="cpu",**kwargs):
        """
        Abstract initializer of a Text Classification network.
        Args:
            loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss)
            optimizer:  One of toch.optim (default: torch.optim.Adam)
            optimizer_params: A dictionary of optimizer parameters
            device: torch device, destination of training (cpu or cuda:0)
        """
        super(TextClassificationAbstract,self).__init__(**kwargs)

        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.PRECISION_DIGITS = 4

    def build(self):
        """
        Internal build method.
        """
        if isinstance(self.loss, type) and self.loss is not None:
            self.loss = self.loss().to(self.device)
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)

    def evaluate_classes(self, classes_subset=None, **kwargs):
        """wrapper for evaluation function if you just want to evaluate on subsets of the classes."""
        if classes_subset is None:
            return self.evaluate(**kwargs)
        else:
            mask = makemultilabels([list(classes_subset.values())], maxlen=len(self.classes))
            return self.evaluate(**kwargs, mask=mask)

    def evaluate(self, data, batch_size=50, return_roc=False, return_report=False, mask=None):
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
        p_1 = Precision(is_multilabel=True,average=True)
        p_3 = Precision(is_multilabel=True,average=True)
        if len(self.classes)>5: p_5 = Precision(is_multilabel=True,average=True)
        subset_65 = Accuracy(is_multilabel=True)
        subset_mcut = Accuracy(is_multilabel=True)
        report = MultiLabelReport(self.classes) if mask is None else MultiLabelReport(self.classes, check_zeros=True)
        auc_roc = AUC_ROC(len(self.classes))
        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                y[y!=0] = 1
                x = self.transform(b["text"])
                output = self(x.to(self.device)).cpu()
                if hasattr(self, "regularize"):
                    l = self.loss(output, y) + self.regularize()
                else:
                    l = self.loss(output, y)
                output = torch.sigmoid(output)

                # Subset evaluation if ...
                if mask is not None:
                    output = output * mask
                    y = y * mask

                average.update(l.item())
                p_1.update((torch.zeros_like(output).scatter(1, torch.topk(output, k=1)[1],1), y))
                p_3.update((torch.zeros_like(output).scatter(1, torch.topk(output, k=3)[1],1), y))
                if len(self.classes)>5:
                    p_5.update((torch.zeros_like(output).scatter(1, torch.topk(output, k=5)[1],1), y))
                subset_65.update((self.threshold(output, tr=0.5, method="hard"), y))
                subset_mcut.update((self.threshold(output, tr=0.5, method="mcut"), y))
                if return_report: report.update((self.threshold(output, tr=0.5, method="mcut"), y))
                auc_roc.update((torch.sigmoid(output).detach(),y.detach()))
        self.train()
        return {
            # "accuracy": accuracy.compute(),
            "valid_loss": round(average.compute().item(), 2*self.PRECISION_DIGITS),
            "p@1": round(p_1.compute(),self.PRECISION_DIGITS),
            "p@3": round(p_3.compute(),self.PRECISION_DIGITS),
            "p@5": round(p_5.compute(),self.PRECISION_DIGITS) if len(self.classes)>5 else None,
            "auc":  auc_roc.compute() if return_roc else round(auc_roc.compute()[0],self.PRECISION_DIGITS),
            "a@0.5": round(subset_65.compute(),self.PRECISION_DIGITS),
            "a@mcut": round(subset_mcut.compute(),self.PRECISION_DIGITS),
            "report": report.compute() if return_report else None,
        }

    def fit(self, train, valid = None, epochs=1, batch_size=16, valid_batch_size=50, classes_subset=None, patience=-1, tolerance=1e-2):
        """
        Training function

        Args:
            train: MultilabelDataset used as training data
            valid: MultilabelDataset to keep track of generalization
            epochs: Number of epochs (times to iterate the train data)
            batch_size: Number of instances in one batch.
            valid_batch_size: Number of instances in one batch  of validation.
        Returns:
            A history dictionary with the loss and the validation evaluation measurements.

        """
        validation=[]
        train_history = {"loss": []}

        best_loss = 10000000
        last_best_loss_update=0

        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs)) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    y = b["labels"].to(self.device)
                    y[y!=0] = 1
                    x = self.transform(b["text"]).to(self.device)
                    output = self(x)
                    if hasattr(self, "regularize"):
                        l = self.loss(output, y) + self.regularize()
                    else:
                        l = self.loss(output, y)
                    l.backward()

                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),2*self.PRECISION_DIGITS)
                    pbar.update()
                # torch.cuda.empty_cache()
                if valid is not None:
                    validation.append(self.evaluate_classes(classes_subset=classes_subset,
                                                           data=valid,
                                                           batch_size=valid_batch_size,
                                                           return_report=False,
                                                           return_roc=False))
                    pbar.postfix[0].update(validation[-1])
                    pbar.update()

            train_history["loss"].append(average.compute().item())
            if patience > -1:
                if  best_loss - average.compute().item() > tolerance:
                    best_loss = average.compute().item()
                    torch.save(self.state_dict(), "checkpoint.pt")
                    #save states
                    last_best_loss_update = 0
                else:
                    last_best_loss_update += 1

                if last_best_loss_update >= patience:
                    break
                # torch.cuda.empty_cache()
        if patience > -1:
            self.load_state_dict(torch.load("checkpoint.pt"))
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
        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x).to(self.device)
        with torch.no_grad(): output = torch.sigmoid(self(x))
        prediction = self.threshold(output, tr=tr, method=method)
        self.train()
        if return_scores:
            return [[(self.classes_rev[i.item()], s[i].item()) for i in torch.where(p==1)[0]] for s, p in zip(output,prediction)]
        return [[self.classes_rev[i.item()] for i in torch.where(p==1)[0]] for p in prediction]

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
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        predictions = []
        for b in tqdm(train_loader):
            predictions.extend(self.predict(b["text"], tr=tr, method=method))
        return predictions

    def threshold(self, x, tr=0.5, method="hard"):
        """
        Thresholding function for outputs of the neural network.
        So far a hard threshold ( tr=0.5, method="hard")  is supported and
        dynamic cutting (method="mcut")

        Args:
            x: A tensor
            tr: Threshold
            method: mcut or hard

        Returns:

        """
        if method=="hard":
            return (x>tr).int()
        if method=="mcut":
            x_sorted = torch.sort(x,-1)[0]
            thresholds = (x_sorted[:,1:] - x_sorted[:,:-1]).max(-1)[0]
            return (x > thresholds[:, None]).float()

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
                    self.embeddings_dim = self.embedding(torch.tensor([[0]]))[0].shape[-1]
                else:
                    self.embedding, self.tokenizer = get(model=self.representation, output_hidden_states=True)
                    self.embeddings_dim = \
                        torch.cat(self.embedding(self.embedding.dummy_inputs["input_ids"])[2][self.n_layers:], -1).shape[-1]
            except TypeError:
                print("If your using a model that does not support returning hiddenstates, set n_layers=1")
                import sys
                sys.exit()
            for param in self.embedding.parameters(): param.requires_grad = False

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