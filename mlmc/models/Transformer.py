import torch
from mlmc.models.abstracts.abstracts import TextClassificationAbstract


##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class Transformer(TextClassificationAbstract):
    """
    Implementation of Yoon Kim 2014 KimCNN Classification network for Multilabel Application (added support for Language Models).
    """
    def __init__(self, dropout=0.5, **kwargs):
        """Class constructor and intialization of every hyperparameters
        :param dropout: Droupout rate
         """
        super(Transformer, self).__init__(**kwargs)

        self._config["dropout"] = dropout

        self.projection = torch.nn.Linear(in_features=self.embeddings_dim, out_features=self.n_classes)
        self.dropout_layer = torch.nn.Dropout(self._config["dropout"] )
        self.build()


    def forward(self, x):
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        output = self.projection(self.dropout_layer(embedded.mean(-2)))
        return output

