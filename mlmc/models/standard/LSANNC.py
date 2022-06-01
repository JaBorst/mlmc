"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
from ..abstracts.abstract_label import LabelEmbeddingAbstract
from ..abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
import re
from mlmc.modules import *

class LSANNC(LabelEmbeddingAbstract,TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, scale="mean", share_weighting=False, weight_norm ="norm", branch_noise = 0., dropout=0.3,
                 hidden_representations= 400,  d_a=200, label_model="glove300", **kwargs):
        """
        Class constructor and initialization of every hyperparameter.

        :param dropout: Dropout rate
        :param hidden_representations: Hidden state dimension of the LSTM used to create the word embeddings
        :param d_a: Arbitrarily set hyperparameter
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstract`
        """
        super(LSANNC, self).__init__(**kwargs)
        self.log_bw=False

        self._config["scale"] = scale
        self._config["share_weighting"] = share_weighting
        self._config["weight_norm"] = weight_norm
        self._config["branch_noise"] = branch_noise
        self._config["dropout"] = dropout
        self._config["hidden_representations"] = hidden_representations
        self._config["d_a"] = d_a
        self._config["label_model"] = label_model


        # Original
        self.create_labels(self.classes)

        self.projection_input = torch.nn.Linear(self.embeddings_dim,
                                                self._config["hidden_representations"] * 2)

        from ...modules import LSANNCModule

        self.lsannc = LSANNCModule(self._config["hidden_representations"]*2,
                                   self.label_embedding_dim ,
                                   hidden_features=self._config["d_a"] )
        self.dropout_layer = torch.nn.Dropout(self._config["dropout"])
        self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.build()

    def forward(self, x, return_weights=False):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :param return_weights: If true, returns the learnable weights of the module as well
        :return: Output tensor
        """
        outputs = self.projection_input(self.embed_input(x) / self.embeddings_dim)
        # outputs = self.dropout_layer(outputs)
        # label_embed = self.dropout_layer(self.label_embedding)
        doc, weights = self.lsannc(outputs, self.label_dict, return_weights=True)

        pred = self.output_layer(doc).squeeze(-1)
        if return_weights:
            return pred, weights
        return pred
    def log_branch_weights(self, s=True):
        """Deprecated"""
        self.log_bw=s
    def reset_branch_weights(self):
        """Deprecated"""
        self.bw=[]
    def get_branch_weights(self):
        """Deprecated"""
        return torch.cat(self.bw).cpu()

    def label_embed(self, classes):
        """
        Embeds the labels of the classes mapping.

        :param classes: The classes mapping
        :return: Tensor containing the embedded labels
        """
        from ...representation import get_word_embedding_mean
        import re
        with torch.no_grad():
            l = get_word_embedding_mean(
                list(self.classes.keys()),
                self._config["label_model"])
            self.label_embedding_dim = l.shape[-1]
        return l
