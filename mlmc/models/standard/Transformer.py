import torch
from ..abstracts.abstract_textclassification import TextClassificationAbstract


##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class Transformer(TextClassificationAbstract):
    """
    Implementation of a simple transformer model.
    """
    def __init__(self, dropout=0.3, **kwargs):
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


    def forward(self, x, emb=False):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        embedded = self._mean_pooling(embedded, x["attention_mask"])
        output = self.projection(self.dropout_layer(embedded))
        if emb:
            return output, (embedded, torch.tensor([0.]))
        return output

    def _contrastive_embedding(self, x, y):
        x = self.transform(x)
        input_embedding = self.embed_input(x)
        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        y = self.transform(y)
        y_embedding = self.embed_input(y)
        y_embedding = self._mean_pooling(y_embedding, y["attention_mask"])
        return input_embedding, y_embedding