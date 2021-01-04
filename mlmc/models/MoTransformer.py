import torch
from mlmc.models.abstracts.abstracts_mo import TextClassificationAbstractMultiOutput


class MoTransformer(TextClassificationAbstractMultiOutput):
    """
    Implementation of a simple transofmrer model for multioutput
    """
    def __init__(self, dropout=0.5, **kwargs):
        """Class constructor and intialization of every hyperparameters
        :param dropout: Droupout rate
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.TextclassificationAbstract`
         """
        super(MoTransformer, self).__init__(**kwargs)

        self._config["dropout"] = dropout
        self.projection = torch.nn.ModuleList([torch.nn.Linear(in_features=self.embeddings_dim, out_features=x) for x in self.n_classes])
        self.dropout_layer = torch.nn.Dropout(self._config["dropout"])
        self.build()


    def forward(self, x):
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        output = [x(self.dropout_layer(embedded.mean(-2))) for x in self.projection]
        return output

