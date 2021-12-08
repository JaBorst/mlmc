from ignite.metrics import Precision, Accuracy
import torch
import itertools


class TFIDF:
    def keywords(self, text, n=100):
        self.create_dtm(text)
        return self.tfidf_keywords(n=n)

    def preprocess(self, txt):
        import nltk
        txt = txt if len(txt[0]) > 1 else nltk.word_tokenize(txt)
        txt = [x.replace("ß", "ss").lower() for x in txt if x.isalpha() and len(x) > 1]
        return txt

    def create_dtm(self, texts, cut_first=200, min_freq=3):
        # Clean texts
        import nltk
        self.texts = {k: self.preprocess(" ".join(v)) for k, v in texts.items()}
        # Count texts
        self.vocab = nltk.FreqDist(sum(self.texts.values(), []))
        # Prune overall vocabulary
        self.vocab = sorted(list(self.vocab.items()), key=lambda x: -x[1])
        self.vocab = [x[0] for x in self.vocab[200:] if x[1] >= min_freq]
        # count and restrict domain level text to the vocabulary
        self.term_frequencies = {genre: nltk.FreqDist(text) for genre, text in self.texts.items()}
        self.dtm = torch.tensor([[v.get(w, 0) for w in self.vocab] for k, v in self.term_frequencies.items()])

    def tfidf(self):
        tf = torch.log(self.dtm + 1e-25)
        # tf = 0.5 + 0.5 * self.dtm / self.dtm.max(-1, keepdims=True) # alternative normalization.
        idf = torch.log(self.dtm.shape[0] / ((self.dtm > 0).sum(0) + 1e-25))
        return tf * idf

    def tfidf_keywords(self, n=10):
        """Iterate all copora for printing"""
        tfidf = self.tfidf()
        return {k: [self.vocab[k] for k in torch.argsort(tfidf[i], dim=0, descending=True)[:n]] for i, k in enumerate(self.texts.keys())}

class TopicCoherence():
    """Multilabel iterative F1/Precision/Recall. Ignite API like"""
    def __init__(self, n=150, weighted=False, **kwargs):
        self.reset()
        self.n = n
        self.weighted=weighted

    def init(self, classes, **kwargs):
        "an extra function for model specific parameters of the metric"
        self.classes = classes

    def reset(self):
        """Clears previously added truth and pred instance attributes."""
        self.scores = []
        self.pred = []
        self.truth = []
        self.text = []

    def update(self, batch):
        """
        Adds classification output to class for computation of metric.

        :param batch: Output of classification task in form (scores, truth, pred)
        """
        assert isinstance(batch, tuple), "batch needs to be a tuple"
        self.text.extend(batch[3])
        self.pred.append(batch[2])
        self.scores.append(batch[0])

    def _get_values(self, pred, texts):
        tfidf = TFIDF()
        text = [tfidf.preprocess(t) for t in texts]
        corpus = [[] for _ in self.classes.keys()]
        for t, b in zip(text, pred):
            for i in torch.where(b == 1)[0].tolist():
                corpus[i].append(" ".join(t))
        kw = tfidf.keywords({k: v for k, v in zip(self.classes.keys(), corpus) if len(v) > 0}, n=self.n)

        # [len((set(kw[cls]).intersection(set(sum([v for k,v in kw.items() if k != cls],[])))))/self.n  for cls in self.classes.keys()]

        counts = torch.FloatTensor([[[t.count(kw) for kw in k] for k in kw.values()] for t in text])
        counts = (counts != 0).float()

        inter_class = 0
        intra_class = 0
        for cls in range(len(kw)):
            cls_idx = list(range(len(kw)))
            cls_idx.remove(cls)
            for i in range(self.n):
                docs = counts[counts[:, cls, i] != 0]
                if docs.shape[0] == 0:
                    continue
                docs[:, :, i] = 0
                if self.weighted:
                    inter_class =inter_class + 1/(i+1) * (docs.sum(-1)[:, cls_idx].sum(0) > 0).float() / docs.shape[0] / self.n / len(kw)
                else:
                    inter_class = inter_class + (docs.sum(-1)[:, cls_idx].sum(0) > 0).float() / docs.shape[0] / self.n / len(kw)

            if docs.shape[0] == 0:
                continue
            intra_class = intra_class + (docs[:, cls, :].sum(-1) > 0).float().mean() / self.n

        return {"intra_class": intra_class.mean(), "inter_class": 1 - inter_class.mean(), "cls": inter_class}
    def compute(self):
        pred = torch.cat(self.pred)
        if not hasattr(self,"baseline"):
            rp = torch.nn.functional.one_hot(torch.randn_like(pred).argmax(-1),num_classes=len(self.classes))
            self.baseline = self._get_values(rp, self.text)
        r = self._get_values(pred, self.text)
        r = {k:torch.relu((v-self.baseline[k])/(1-self.baseline[k])) for k,v in r.items()}
        return r

    def print(self):
        """
        Computes metric.

        :return: Classification report
        """
        r = self.compute()
        return r

