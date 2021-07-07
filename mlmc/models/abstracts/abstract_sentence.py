import torch

import mlmc.data.datasets
from mlmc.models.abstracts.abstracts import TextClassificationAbstract
import mlmc
from ignite.metrics import Average
from mlmc.metrics.precisionk import Accuracy
from ...data.dataset_formatter import SFORMATTER


class SentenceTextClassificationAbstract(TextClassificationAbstract):
    """
    Extending the base class with functionality regarding the sentence-embedding approach and zeroshot capabilities
    """
    def __init__(self, sformatter = lambda x: f"This is about {x}", label_len=45, *args, **kwargs):
        """
        Additional Arguments for sentence Embedder
        Args:
            sformatter: Formatting the label. A callable that takes and returns a string. You can modify the label representation
            label_len:  Setting the maximum token lenght of the label embeddings. This is mainly important for pretraining MNLI or STS
            *args: See mlmc.models.abstract.abstract
            **kwargs:  See mlmc.models.abstract.abstract
        """
        super(SentenceTextClassificationAbstract, self).__init__(*args, **kwargs)
        self._config["sformatter"] = sformatter
        self._config["label_len"] = label_len
        self._all_compare = True

    def set_sformatter(self, c):
        """
        Setter for the label sformatter
        Args:
            c: callable that takes and returns a string

        Returns:

        """
        assert callable(c)
        self._config["sformatter"] = c

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean Pooling for sequence of embeddings, taking attention mask into account for correct averaging.
        Using the output of the language models
        Args:
            token_embeddings:
            attention_mask:

        Returns:

        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def transform(self,x: str, max_length=None) ->dict:
        """Sentence based transformer returns all language model output in a dict
        Args:
            x: string
        """
        if max_length is None:
            max_length = self._config["max_len"]
        return {k:v.to(self.device) for k, v in self.tokenizer.tokenizer(x, padding=True, max_length=max_length, truncation=True,  return_tensors='pt').items()}

    def label_embed(self, x):
        """
        Label embedder in this instance uses the same transformation as the input
        Args:
            x:

        Returns:

        """
        return self.transform([self._config["sformatter"](l) for l in list(x)],max_length=self._config["label_len"] )

    def create_labels(self, classes: dict):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        # if isinstance(classes, list):
        #     classes = {w:i for i,w in enumerate(classes)}
        self.classes = classes
        self.n_classes = len(classes)
        self._config["classes"] = classes
        self._config["n_classes"] = self.n_classes

        if isinstance(classes, dict):
            self.classes_rev = {v: k for k, v in self.classes.items()}

        if self.n_classes != 0: # To ensure we can initialize the model without specifying classes
            # r = self.label_embed(self.classes)
            # self.label_dict = r["input_ids"]
            # self.label_att =  r["attention_mask"]
            self.label_dict = self.label_embed(self.classes)
            # self.label_dict["token_type_ids"][:] = self.label_dict["attention_mask"]


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

    def single(self):
        """Helper function to set model into default single label mode"""
        self._config["target"] = "single"
        self.set_threshold("max")
        self.set_activation( lambda x: x)

    def multi(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)

    def sts(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "sts"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)
        # from ...loss import RelativeRankingLoss
        # self.set_loss(RelativeRankingLoss(0.5))

    def entailment(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "entailment"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)
        self.loss = torch.nn.CrossEntropyLoss()
        # from ...loss import RelativeRankingLoss
        # self.set_loss(RelativeRankingLoss(0.5))