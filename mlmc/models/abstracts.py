from tqdm import tqdm
import torch
import ignite
from sklearn import metrics as skmetrics
from ..metrics.multilabel import MultiLabelReport



class TextClassificationAbstract(torch.nn.Module):
    def __init__(self, loss, optimizer, optimizer_params = {"lr": 1.0}, device="cpu", **kwargs):
        super(TextClassificationAbstract,self).__init__(**kwargs)

        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    def build(self):
        self.to(self.device)
        self.loss = self.loss().to(self.device)
        self.optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

    def evaluate(self, data, batch_size=50, return_report=False):
        """
        Evaluation, return accuracy and loss
        """
        self.eval()  # set mode to evaluation to disable dropout
        p_1 = ignite.metrics.Precision(is_multilabel=True,average=True)
        p_3 = ignite.metrics.Precision(is_multilabel=True,average=True)
        p_5 = ignite.metrics.Precision(is_multilabel=True,average=True)
        subset_65 = ignite.metrics.Accuracy(is_multilabel=True)
        subset_mcut = ignite.metrics.Accuracy(is_multilabel=True)
        report = MultiLabelReport(self.classes)
        average = ignite.metrics.Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                y = b["labels"]
                y[y!=0] = 1
                x = self.transform(b["text"])
                output = self(x.to(self.device)).cpu()
                l = self.loss(output, torch._cast_Float(y))

                output = torch.sigmoid(output)
                average.update(l.item())
                # accuracy.update((prediction, y))
                p_1.update((torch.zeros_like(output).scatter(1,torch.topk(output, k=1)[1],1), y))
                p_3.update((torch.zeros_like(output).scatter(1,torch.topk(output, k=3)[1],1), y))
                p_5.update((torch.zeros_like(output).scatter(1,torch.topk(output, k=5)[1],1), y))
                subset_65.update((self.threshold(output,tr=0.65,method="hard"), y))
                subset_mcut.update((self.threshold(output,tr=0.65,method="mcut"), y))
                report.update((self.threshold(output,tr=0.65,method="mcut"), y))
                # auc_roc.update((torch.sigmoid(output),y))
        self.train()
        return {
            # "accuracy": accuracy.compute(),
            "valid_loss": round(average.compute().item(),6),
            "p@1": round(p_1.compute(),4),
            "p@3": round(p_3.compute(),4),
            "p@5": round(p_5.compute(),4),
            "a@0.65": round(subset_65.compute(),4),
            "a@mcut": round(subset_mcut.compute(),4),
            "report": report.compute() if return_report else None
            # "auc": round(auc_roc.compute(),4),
        }

    def fit(self, train, valid = None, epochs=1, batch_size=16):
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = ignite.metrics.Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs)) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    y = b["labels"].to(self.device)
                    x = self.transform(b["text"]).to(self.device)
                    output = self(x)
                    l = self.loss(output, y)
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),6)
                    l.backward()
                    self.optimizer.step()
                    pbar.update()
                if valid is not None:

                    pbar.postfix[0].update(self.evaluate(valid))
                    pbar.update()

    def threshold(self, x, tr=0.5, method="hard"):
        if method=="hard":
            return (torch.sigmoid(x)>tr).int()
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