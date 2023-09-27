import torch
from tqdm import tqdm

from mlmc.data.dataset_classes import PredictionDataset

class BayesNetwork:
    def __init__(self, model):
        self.model = model

    # def __getattr__(self, attr):
    #     return getattr(self.model, attr)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def bayesian_prediction(self, x, return_scores=False, n=10, p=0.3):
        self.model = self.model.eval()
        for param in self.model.modules():
            if isinstance(param, torch.nn.Dropout):
                param.p=p
                param.training = True
        xt = self.model.transform(x)
        with torch.no_grad():
            outputs = []
            for _ in range(n):
                outputs.append(self.model.act(self(xt)).cpu())

        o = torch.stack(outputs)
        prediction = sum([self.model._threshold_fct(x).cpu() for x in o]) / n
        pmax = self.model._threshold_fct(prediction)
        scores, variance =  o.mean(0), o.var(0)
        if not hasattr(self, "classes_rev"):
            self.model.classes_rev = {v: k for k, v in self.model._config["classes"].items()}
        labels = [[self.model.classes_rev[i.item()] for i in torch.where(p == 1)[0]] for p in pmax]
        self.model = self.model.train()
        if return_scores:
            return labels, scores.cpu(), variance.cpu(), prediction.cpu()
        return labels


    def bayesian_predict_batch(self, data, batch_size=50, return_scores=False, **kwargs):
        """
        Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.

        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`

        Args:
            data: A MultilabelDataset
            batch_size: Batch size

        Returns:
            A list of labels

        """
        train_loader = torch.utils.data.DataLoader(PredictionDataset(x=data), batch_size=batch_size, shuffle=False)
        predictions = []
        if not hasattr(self, "classes_rev"):
            self.model.classes_rev = {v: k for k, v in self.model.classes.items()}
        for b in tqdm(train_loader, ncols=100):
            predictions.extend(self.bayesian_prediction(b["text"], return_scores=return_scores,**kwargs))
        del self.model.classes_rev
        if return_scores:
            labels = sum([predictions[x] for x in list(range(0, len(predictions), 4))],[])
            scores = torch.cat([predictions[x] for x in list(range(1, len(predictions) + 1, 4))], dim=0)
            variances = torch.cat([predictions[x] for x in list(range(2, len(predictions) + 1, 4))], dim=0)
            bools = torch.cat([predictions[x] for x in list(range(3, len(predictions), 4))], dim=0)
            return labels, scores ,variances , bools
        else:
            labels = sum([predictions[x] for x in list(range(0, len(predictions)))], [])
            return labels
