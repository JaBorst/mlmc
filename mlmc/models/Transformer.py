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
    def __init__(self, classes, representation="roberta", dropout=0.5, n_layers=1, max_len=200, **kwargs):
        """Class constructor and intialization of every hyperparameters

        :param classes:  A list of dictionary of the class label and the corresponding index
        :param representation: The name of the representation to use. glove* or one of the hugginface transformers models.
        :param dropout: Droupout rate
        :param max_len: Maximum length input sequences. Longer sequences will be cut.
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.TextclassificationAbstract`
         """
        super(Transformer, self).__init__(classes=classes,**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.l = 1
        self.representation = representation
        self.dropout = dropout
        self.n_layers = n_layers
        self._init_input_representations()


        self.projection = torch.nn.Linear(in_features=self.embeddings_dim, out_features=self.n_classes)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.build()


    def forward(self, x):
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        output = self.projection(self.dropout_layer(embedded.mean(-2)))
        return output

