from tqdm import tqdm
import torch
from ignite.metrics import Precision, Accuracy, Average
from ..metrics.multilabel import MultiLabelReport,AUC_ROC
from ..representation import is_transformer,get
from ..representation.labels import makemultilabels



class LanguageModelAbstract(torch.nn.Module):
    """
    Abstract class for Language Models. Defines fit, evaluate, predict and threshold methods for virtually any
    language model.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def __init__(self, loss=torch.nn.CrossEntropyLoss, optimizer=torch.optim.Adam, optimizer_params={"lr": 5e-5}, device="cpu", **kwargs):
        super(LanguageModelAbstract, self).__init__(**kwargs)

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

    def evaluate(self, data, batch_size=50):
        """
        Evaluation, return accuracy and loss
        """
        self.eval()  # set mode to evaluation to disable dropout
        p_1 = Accuracy()#Precision(is_multilabel=True,average=True)
        average = Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

        with torch.no_grad():
            for i, b in enumerate(data_loader):
                x = self.transform(b["input"]).to(self.device)
                y = self.transform(b["forward"]).to(self.device)
                output = self(x)
                if hasattr(self, "regularize"):
                    l = self.loss(output.permute(0, 2, 1), y) + self.regularize()
                else:
                    l = self.loss(output.permute(0, 2, 1), y)

                output = torch.softmax(output, -1)
                average.update(l.item())
                for i in range(output.shape[1]):
                    p_1.update((torch.nn.functional.one_hot(torch.max(output[:, i], -1)[1], output.shape[-1]),
                                torch.nn.functional.one_hot(y[:, i], output.shape[-1])))

        self.train()
        return {
            # "accuracy": accuracy.compute(),
            "valid_loss": round(average.compute().item(), 2*self.PRECISION_DIGITS),
            "precision": round(p_1.compute(),self.PRECISION_DIGITS),
            "perplexity": round(torch.exp(average.compute()).item(), 2*self.PRECISION_DIGITS),
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
                    x = self.transform(b["input"]).to(self.device)
                    y = self.transform(b["forward"]).to(self.device)
                    output = self(x)

                    if hasattr(self, "regularize"):
                        l = self.loss(output.permute(0,2,1), y) + self.regularize()
                    else:
                        l = self.loss(output.permute(0, 2, 1), y)
                    l.backward()
                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),2*self.PRECISION_DIGITS)
                    pbar.update()
                # torch.cuda.empty_cache()
                if valid is not None:
                    validation.append(self.evaluate(data=valid,
                                                    batch_size=valid_batch_size))
                    pbar.postfix[0].update(validation[-1])
                    pbar.update()
                # torch.cuda.empty_cache()
            train_history["loss"].append(average.compute().item())
        return{"train":train_history, "valid": validation }

    def generate(self, prompt="", steps=100, sample=True):
        with torch.no_grad():
            answer = self.tokenizer.encode(prompt).ids[:-1]

            for _ in range(steps):
                cl = answer[(-min(len(answer), self.max_len)):]
                output  = self(torch.LongTensor([cl]).to(self.device))
                probabilities = torch.softmax(output[0,-1],-1).detach().cpu()
                if sample:
                    import numpy as np
                    next_token = np.random.choice(range(probabilities.shape[-1]),1, p=(probabilities/probabilities.sum()).numpy())[0]
                if not sample:
                    next_token=torch.argmax(probabilities).item()
                answer.append(next_token)

        return self.tokenizer.decode(answer).replace(' ##', '')
