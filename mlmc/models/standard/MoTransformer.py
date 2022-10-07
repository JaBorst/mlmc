import torch
from ..abstracts.abstracts_multi_output import TextClassificationAbstractMultiOutput


class MoTransformer(TextClassificationAbstractMultiOutput):
    """
    Implementation of a simple transformer model for multioutput
    """
    def __init__(self, dropout=0.5, **kwargs):
        """
        Class constructor and initialization of every hyperparameter.

        :param dropout: Dropout rate
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstractMultiOutput`
        """
        super(MoTransformer, self).__init__(**kwargs)

        self._config["dropout"] = dropout
        self.projection = torch.nn.ModuleList([torch.nn.Linear(in_features=self.embeddings_dim, out_features=x) for x in self.n_classes])
        self.dropout_layer = torch.nn.Dropout(self._config["dropout"])
        self.build()


    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        output = [x(self.dropout_layer(embedded.mean(-2))) for x in self.projection]
        return output

