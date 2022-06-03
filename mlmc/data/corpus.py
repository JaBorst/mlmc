from .sampler import get as sampler_get
from .dataset_classes import MultiLabelDataset, SingleLabelDataset, PredictionDataset

class Corpus():
    def __init__(self, corpus, classes=None, strategy=None):
        self.set_classes(classes)
        self.set_corpus(corpus)
        self.set_strategy(strategy)

    def set_corpus(self, c):
        if self._dataset_type == PredictionDataset:
            self._corpus = self._dataset_type(x=c)
        else:
            self._corpus = self._dataset_type(x = c, y=self._labels)

    def set_classes(self, c):
        self._classes = c

    def set_strategy(self, s):
        self._strategy =  sampler_get(s) if s is not None else sampler_get("random")

    def set_labels(self, l: list):
        if l is not None:
            assert isinstance(l, list)
            assert isinstance(l[0], list)
            assert isinstance(l[0][0], str)
        self._labels = l
        self._dataset_type = (SingleLabelDataset if len(l[0]) == 1 else MultiLabelDataset) if l is not None else PredictionDataset

    def get_sample(self, n=100):
        self._strategy(dataset=self._corpus, k=n)
