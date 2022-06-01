"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import re

from ..abstracts.abstracts_multi_output import TextClassificationAbstractMultiOutput
from ..abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...modules import *


class MoLSANNC(TextClassificationAbstractMultiOutput, TextClassificationAbstractZeroShot):

    def __init__(self, scale="mean", share_weighting=False, weight_norm="norm", branch_noise=0., dropout=0.3,
                 hidden_representations=400, label_model="glove300", d_a=200, **kwargs):
        """
        Class constructor and initialization of every hyperparameter.

        :param dropout: Dropout rate
        :param hidden_representations: Hidden state dimension of the LSTM used to create the word embeddings
        :param d_a: Arbitrarily set hyperparameter
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstract`
        """
        super(MoLSANNC, self).__init__(**kwargs)
        # My Stuff
        self._config["scale"] = scale
        self._config["share_weighting"] = share_weighting
        self._config["weight_norm"] = weight_norm
        self._config["branch_noise"] = branch_noise
        self._config["dropout"] = dropout
        self._config["hidden_representations"] = hidden_representations
        self._config["d_a"] = d_a
        self._config["label_model"] = label_model

        self.log_bw = False
        # Original
        self.create_labels(self.classes)

        self.projection_input = torch.nn.Linear(self.embeddings_dim,
                                                self._config["hidden_representations"] * 2)

        # self.projection_labels = torch.nn.Linear(self.label_embedding_dim, self.hidden_representations)
        from ...modules import LSANNCModule
        self.lsannc = LSANNCModule(self._config["hidden_representations"]*2,
                                   self.label_embedding_dim,
                                   hidden_features=self._config["d_a"])

        self.output_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in range(self.n_outputs)])
        self.dropout_layer = torch.nn.Dropout(0.3)
        self.build()

    def forward(self, x, return_weights=False):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :param return_weights: If true, returns the learnable weights of the module as well
        :return: Output tensor
        """
        outputs = self.projection_input(self.embed_input(x) / self.embeddings_dim)
        label_embed = torch.cat([x for x in self.label_embedding],0)

        outputs = self.dropout_layer(outputs)
        doc, weights = self.lsannc(outputs,label_embed,return_weights=True)
        # if self.log_bw:
        #     self.bw.append(weights.cpu())
        doc = self.dropout_layer(doc)
        n = 0
        labels = []
        for i in self.n_classes:
            labels.append(doc[:,n:(n+i)])
            n = n+i

        pred = [l(o / self.label_embedding_dim).squeeze(-1) for o,l in zip(labels, self.output_layer)]
        if return_weights:
            return pred, weights
        return pred

    def log_branch_weights(self, s=True):
        """Deprecated"""
        self.log_bw = s

    def reset_branch_weights(self):
        """Deprecated"""
        self.bw = []

    def get_branch_weights(self):
        """Deprecated"""
        return torch.cat(self.bw).cpu()

    def create_label_dict(self):
        """
        Embeds the labels of each class.

        :return: Dictionary containing the original label with its corresponding embedding.
        """
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        from ...representation import get_word_embedding_mean
        with torch.no_grad():
            l = [get_word_embedding_mean(
                [(" ".join(re.split("[/ _-]", re.sub("[0-9]", "", x.lower())))).strip() for x in classes.keys()],
                self._config["label_model"] ) for classes in self.classes]

        self.label_embedding_dim = l[0].shape[-1]
        return [{w: e for w, e in zip(classes.keys(), emb)} for classes, emb in zip(self.classes, l)]

    def create_labels(self, classes):
        """
        Creates label embeddings and adds them to the model in form of a ParameterList.

        :param classes: The classes mapping
        """
        self.classes = classes
        self._config["classes"]=classes
        self.n_classes = [len(x) for x in classes]

        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict()
        self.label_embedding = [torch.stack([dic[cls] for cls in cls.keys()]) for cls, dic in
                                zip(self.classes, self.label_dict)]
        self.label_embedding = torch.nn.ParameterList([torch.nn.Parameter(x) for x in self.label_embedding]).to(
            self.device)
        for x in self.label_embedding: x.requires_grad = True
