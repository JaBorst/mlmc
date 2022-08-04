import torch
from tqdm import tqdm

from ...data import SingleLabelDataset, PredictionDataset

class AspectBasedSentimentAbstract(torch.nn.Module):
    def _batch_entailment_predict(self, x, hypotheses, batch_size=15):
        data = PredictionDataset(x=x, hypothesis=hypotheses)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        outputs = []
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                x = self.transform(b["text"], b.get("hypothesis", None))
                outputs.append(self(x).cpu())
        return torch.cat(outputs, 0)

    def _predict(self, o):
        p = o[..., self.entailment_id].softmax(-1).argmax(-1)
        return p

    def predict_aspect_based_sentiment(self, x, aspects, sformatter, mode="commonaspects"):
        if mode == "commonaspects":
            _x = sum([[s]*len(aspects)*self._config["n_classes"] for s in x],[])
            hypotheses = sum([[sformatter(cls, asp) for cls in self.classes.keys()] for asp in aspects*len(x)],[])
            self.entailment()
            o = self._batch_entailment_predict(x=_x, hypotheses=hypotheses).reshape(
                (-1, len(aspects), self._config["n_classes"], 3))
            p = self._predict(o)
            classes_rev = {v: k for k, v in self.classes.items()}
            return [[(a, classes_rev[l.item()]) for a, l in zip(aspects, x)] for x in p]

        if mode == "aspectlist":
            _x = sum([[s] * self._config["n_classes"] for s in x], [])
            hypotheses = sum([[sformatter(cls, asp) for cls in self.classes.keys()] for asp in aspects],[])
            self.entailment()
            o = self._batch_entailment_predict(x=_x, hypotheses=hypotheses).reshape((-1, self._config["n_classes"], 3))
            p = self._predict(o)
            classes_rev = {v: k for k, v in self.classes.items()}
            return [(a, classes_rev[l.item()]) for a, l in zip(aspects, p)]
