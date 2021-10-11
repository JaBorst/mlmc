import torch

import mlmc.data.dataset_classes
from .abstract_label import LabelEmbeddingAbstract


class SentenceTextClassificationAbstract(LabelEmbeddingAbstract):
    """
    Extending the base class with functionality regarding the sentence-embedding approach and zeroshot capabilities
    """
    def __init__(self,_all_compare=True, *args, **kwargs):
        """
        Additional Arguments for sentence Embedder
        Args:
            sformatter: Formatting the label. A callable that takes and returns a string. You can modify the label representation
            label_len:  Setting the maximum token lenght of the label embeddings. This is mainly important for pretraining MNLI or STS
            *args: See mlmc.models.abstract.abstract
            **kwargs:  See mlmc.models.abstract.abstract
        """
        super(SentenceTextClassificationAbstract, self).__init__(*args, **kwargs)
        self._all_compare = _all_compare

    def _metric_sim(self, x, y, m=None):
        """
        Helper function for cosine similarity of tensors. Also possible to train a metric tensor
        Args:
            x:
            y:
            m: metric tensor if exists

        Returns:

        """
        m = m if m is not None else torch.eye(x.shape[-1]).to(x.device)
        if self._all_compare:
            x = x / torch.sqrt((x * torch.matmul(x, m.t())).sum(-1, keepdims=True))
            y = y / torch.sqrt((y * torch.matmul(y, m.t())).sum(-1, keepdims=True))
            return (torch.matmul(x, m.t())[:,None] * y).sum(-1)
        else:
            assert x.shape[0] == y.shape[0], "For 1 vs 1 comaprison the tensors must have equal batch size"
            x = x / torch.sqrt((x * torch.matmul(x, m.t())).sum(-1, keepdims=True))
            y = y / torch.sqrt((y * torch.matmul(y, m.t())).sum(-1, keepdims=True))
            return (torch.matmul(x, m.t()) * y).sum(-1)

    def single(self, all_compare=True, **kwargs):
        """Helper function to set the model into single label mode"""
        super().single(**kwargs, all_compare=all_compare)

    def multi(self, all_compare=True, **kwargs):
        """Helper function to set the model into multi label mode"""
        super().multi(all_compare=all_compare, **kwargs)

    def sts(self, all_compare=True, **kwargs):
        """Helper function to set the model into sts mode"""
        super().sts(all_compare=all_compare, **kwargs)

    def entailment(self, all_compare=True, **kwargs):
        """Helper function to set the model into entailment mode"""
        super().entailment(all_compare=all_compare, **kwargs)