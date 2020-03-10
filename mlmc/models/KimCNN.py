import torch
from .abstracts import TextClassificationAbstract
from ..representation import get
##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class KimCNN(TextClassificationAbstract):
    """
    Implementation of Yoon Kim 2014 KimCNN Classification network for Multilabel Application.
    Added support for Language Model as a feature.
    """
    def __init__(self, classes, mode="transformer", representation="roberta", kernel_sizes=[3,4,5,6], filters=100, dropout=0.5, max_len=500, **kwargs):
        """
        Class constructor and intialization of every hyperparameters
        :param classes:  A dictionary of the class label and the corresponding index
        :param mode:  One of (trainable, untrainable, multichannel, transformer). Transformer will use a language model. for the other three refer to the paper.
        :param representation: The name of the representation to use. glove* or one of the hugginface transformers models.
        :param kernel_sizes: Sizes of the kernel used for the convolution
        :param filters: Number of filters used in the convolution
        :param dropout: Droupout rate
        :param max_len: Maximum length input sequences. Longer sequences will be cut.
        :param kwargs:
        """
        super(KimCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.modes = ("trainable","untrainable","multichannel","transformer")
        self.mode = mode
        self.l = 1
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.representation = representation
        self.dropout = dropout

        assert self.mode in self.modes, "%s not in (%s, %s, %s, %s)" % (self.mode, *self.modes)
        self._init_input_representations()

        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embeddings_dim, self.filters, k) for k in self.kernel_sizes])
        self.projection = torch.nn.Linear(in_features=self.l*len(self.kernel_sizes)*self.filters, out_features=self.n_classes)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.build()

    def _init_input_representations(self):
        if self.mode == "trainable":
            self.embedding, self.tokenizer = get(model=self.representation, freeze=False)
            self.embeddings_dim = self.embedding.weight.shape[-1]
        elif self.mode == "untrainable":
            self.embedding, self.tokenizer = get(model=self.representation, freeze=True)
            self.embeddings_dim= self.embedding.weight.shape[-1]
        elif self.mode =="multichannel":
            self.l = 2
            self.embedding, self.tokenizer = get(model=self.representation, freeze=True)
            self.embeddings_dim = self.embedding.weight.shape[-1]

            self.embedding_untrainable = torch.nn.Embedding(*self.embedding.weight.shape)
            self.embedding_untrainable = self.embedding.from_pretrained(self.embedding.weight.clone(), freeze=False)
            self.embedding = torch.nn.ModuleList([self.embedding, self.embedding_untrainable])

        elif self.mode == "transformer":
            self.embedding, self.tokenizer = get(model=self.representation, output_hidden_states=True)
            self.embeddings_dim = torch.cat(self.embedding(self.embedding.dummy_inputs["input_ids"])[2][-5:-1], -1).shape[-1]

    def forward(self, x):

        if self.mode == "trainable":
            embedded = self.embedding(x).permute(0, 2, 1)
            embedded = self.dropout_layer(embedded)
            c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        elif self.mode == "untrainable":
            with torch.no_grad():
                embedded = self.embedding(x).permute(0, 2, 1)
            embedded = self.dropout_layer(embedded)
            c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        elif self.mode =="multichannel":
            embedded_1 = self.embedding[0](x).permute(0, 2, 1)
            with torch.no_grad():
                embedded_2 = self.embedding[1](x).permute(0, 2, 1)
            embedded_1 = self.dropout_layer(embedded_1)
            embedded_2 = self.dropout_layer(embedded_2)
            c = [torch.nn.functional.relu(conv(embedded_1).permute(0, 2, 1).max(1)[0]) for conv in self.convs]+\
                [torch.nn.functional.relu(conv(embedded_2).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        elif self.mode == "transformer":
            with torch.no_grad():
                embedded = torch.cat(self.embedding(x)[2][-5:-1], -1).permute(0, 2, 1)
            c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]

        c = torch.cat(c, 1)
        output = self.projection(self.dropout_layer(c))
        return output
