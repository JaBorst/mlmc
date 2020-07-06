import torch
from .abstracts_mo import TextClassificationAbstractMultiOutput
from ..representation import get
##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class MoTransformer(TextClassificationAbstractMultiOutput):
    """
    Implementation of Yoon Kim 2014 KimCNN Classification network for Multilabel Application (added support for Language Models).
    """
    def __init__(self, n_outputs, classes, mode="transformer", representation="roberta", kernel_sizes=[3,4,5,6], filters=100, dropout=0.5, n_layers=1, max_len=200, **kwargs):
        """Class constructor and intialization of every hyperparameters

        :param classes:  A dictionary of the class label and the corresponding index
        :param mode:  One of (trainable, untrainable, multichannel, transformer).
                    Transformer has to be used in combination with representation being a transformer model name (see: https://huggingface.co/transformers/pretrained_models.html).
                    In combination with the other three one of the glove embeddings can be used (glove50, glove100, glove200, glove300)
                    'trainable' will finetune the wordembedding used, whereas "untrainable" will freeze the embedding layer.
                    'multichannel' will combine two embedding layers, one for finetuning on the task, one frozen.
        :param representation: The name of the representation to use. glove* or one of the hugginface transformers models.
        :param kernel_sizes: Sizes of the kernel used for the convolution
        :param filters: Number of filters used in the convolution
        :param dropout: Droupout rate
        :param max_len: Maximum length input sequences. Longer sequences will be cut.
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.TextclassificationAbstract`
         """
        super(MoTransformer, self).__init__(n_outputs = n_outputs, **kwargs)

        self.classes = classes
        self.n_classes = [len(x) for x in classes]
        self.max_len = max_len
        self.modes = ("trainable","untrainable","multichannel","transformer")
        self.mode = mode
        self.l = 1
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.representation = representation
        self.dropout = dropout
        self.n_layers = n_layers

        assert self.mode in self.modes, "%s not in (%s, %s, %s, %s)" % (self.mode, *self.modes)
        self._init_input_representations()


        self.projection = torch.nn.ModuleList([torch.nn.Linear(in_features=self.embeddings_dim, out_features=x) for x in self.n_classes])
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.build()


    def forward(self, x):

        if self.finetune:
            if self.n_layers == 1:
                embedded = self.embedding(input_ids=x)[0].permute(0, 2, 1)
            else:
                embedded = torch.cat(self.embedding(input_ids=x)[1][(self.n_layers):], -1).permute(0, 2, 1)
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embedded = self.embedding(input_ids=x)[0].permute(0, 2, 1)
                else:
                    embedded = torch.cat(self.embedding(input_ids=x)[1][(self.n_layers):], -1).permute(0, 2, 1)
        output = [x(self.dropout_layer(embedded.mean(-1))) for x in self.projection]
        return output

