import torch
from mlmc.models.abstracts.abstract_textclassification import TextClassificationAbstract


##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class Transformer(TextClassificationAbstract):
    """
    Implementation of a simple transformer model.
    """
    def __init__(self, dropout=0.5, **kwargs):
        """
        Class constructor and initialization of every hyperparameter.

        :param dropout: Dropout rate
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstract`
        """
        super(Transformer, self).__init__(**kwargs)

        self._config["dropout"] = dropout

        self.projection = torch.nn.Linear(in_features=self.embeddings_dim, out_features=self.n_classes)
        self.dropout_layer = torch.nn.Dropout(self._config["dropout"] )
        self.build()


    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        output = self.projection(self.dropout_layer(embedded.mean(-2)))
        return output

