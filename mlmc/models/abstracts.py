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
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def __init__(self, loss=torch.nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam, optimizer_params = {"lr": 5e-5}, device="cpu",**kwargs):
        super(TextClassificationAbstract,self).__init__(**kwargs)

        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.PRECISION_DIGITS = 4

    def build(self):
        if isinstance(self.loss, type) and self.loss is not None:
            self.loss = self.loss().to(self.device)
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)

    def evaluate_classes(self, classes_subset=None, **kwargs):
        if classes_subset is None:
            return self.evaluate(**kwargs)
        else:
            mask = makemultilabels([list(classes_subset.values())], maxlen=len(self.classes))
            return self.evaluate(**kwargs, mask=mask)

    def evaluate(self, data, batch_size=50, return_roc=False, return_report=False, mask=None):
        """
        Evaluation, return accuracy and loss
        """
        self.eval()  # set mode to evaluation to disable dropout
        p_1 = Precision(is_multilabel=True,average=True)
        p_3 = Precision(is_multilabel=True,average=True)
        p_5 = Precision(is_multilabel=True,average=True)
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
                    l = self.loss(output, torch._cast_Float(y)) + self.regularize()
                else:
                    l = self.loss(output, torch._cast_Float(y))
                output = torch.sigmoid(output)

                # Subset evaluation if ...
                if mask is not None:
                    output = output * mask
                    y = y * mask

                average.update(l.item())
                p_1.update((torch.zeros_like(output).scatter(1, torch.topk(output, k=1)[1],1), y))
                p_3.update((torch.zeros_like(output).scatter(1, torch.topk(output, k=3)[1],1), y))
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
            "p@5": round(p_5.compute(),self.PRECISION_DIGITS),
            "auc":  auc_roc.compute() if return_roc else round(auc_roc.compute()[0],self.PRECISION_DIGITS),
            "a@0.65": round(subset_65.compute(),self.PRECISION_DIGITS),
            "a@mcut": round(subset_mcut.compute(),self.PRECISION_DIGITS),
            "report": report.compute() if return_report else None,
        }

    def fit(self, train, valid = None, epochs=1, batch_size=16, valid_batch_size=50, classes_subset=None):
        validation=[]
        train_history = {"loss": []}
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
                        l = self.loss(output, torch._cast_Float(y)) + self.regularize()
                    else:
                        l = self.loss(output, torch._cast_Float(y))
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
                # torch.cuda.empty_cache()
            train_history["loss"].append(average.compute().item())
        return{"train":train_history, "valid": validation }


    def predict(self, x, tr=0.65, method="hard"):
        self.eval()
        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x).to(self.device)
        with torch.no_grad(): output = self(x)
        prediction = self.threshold(torch.sigmoid(output), tr=tr, method=method)
        self.train()
        return [[self.classes_rev[i.item()] for i in torch.where(p==1)[0]] for p in prediction]

    def predict_dataset(self, data, batch_size=50, tr=0.65, method="hard"):
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        predictions = []
        for b in tqdm(train_loader):
            predictions.extend(self.predict(b["text"], tr=tr, method=method))
        return predictions

    def threshold(self, x, tr=0.5, method="hard"):
        if method=="hard":
            return (x>tr).int()
        if method=="mcut":
            x_sorted = torch.sort(x,-1)[0]
            thresholds = (x_sorted[:,1:] - x_sorted[:,:-1]).max(-1)[0]
            return (x > thresholds[:, None]).float()

    def transform(self, x):
        assert hasattr(self, 'tokenizer'), "If the model does not have a tokenizer attribute, please implement the" \
                                           "transform(self, x)  method yourself. TOkenizer can be allocated with " \
                                           "embedder, tokenizer = mlmc.helpers.get_embedding() or " \
                                           "embedder, tokenizer = mlmc.helpers.get_transformer()"
        return self.tokenizer(x,self.max_len)

    def _init_input_representations(self):
        if is_transformer(self.representation):
            self.n_layers=4
            self.embedding, self.tokenizer = get(self.representation, output_hidden_states=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers
        else:
            self.embedding, self.tokenizer = get(self.representation, freeze=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]
