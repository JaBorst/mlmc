import torch
from mlmc.models.abstracts.abstracts_mo import TextClassificationAbstractMultiOutput


class MoTransformer(TextClassificationAbstractMultiOutput):
    """
    Implementation of a simple transofmrer model for multioutput
    """
    def __init__(self, classes, representation="roberta", dropout=0.5, n_layers=1, max_len=200, **kwargs):
        """Class constructor and intialization of every hyperparameters

        :param classes:  A list of dictionary of the class label and the corresponding index
        :param representation: The name of the representation to use. glove* or one of the hugginface transformers models.
        :param dropout: Droupout rate
        :param max_len: Maximum length input sequences. Longer sequences will be cut.
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.TextclassificationAbstract`
         """
        super(MoTransformer, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = [len(x) for x in classes]
        self.n_outputs = len(self.n_classes)
        self.max_len = max_len
        self.l = 1
        self.representation = representation
        self.dropout = dropout
        self.n_layers = n_layers
        self._init_input_representations()


        self.projection = torch.nn.ModuleList([torch.nn.Linear(in_features=self.embeddings_dim, out_features=x) for x in self.n_classes])
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.build()


    def forward(self, x):
        embedded = self.embed_input(x)
        embedded = self.dropout_layer(embedded)
        output = [x(self.dropout_layer(embedded.mean(-2))) for x in self.projection]
        return output

