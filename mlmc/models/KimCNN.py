import torch
from .abstracts import TextClassificationAbstract
from ..representation import get
##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class KimCNN(TextClassificationAbstract):
    def __init__(self, classes, mode="transformer", representation="roberta", max_len=500, **kwargs):
        super(KimCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.modes = ("trainable","untrainable","multichannel","transformer")
        self.mode = mode
        self.l = 1
        self.kernel_sizes = [3,4,5,6]
        self.filters = 100
        self.representation = representation
        assert self.mode in self.modes, "%s not in (%s, %s, %s, %s)" % (self.mode, *self.modes)
        self._init_input_representations()
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embeddings_dim, self.filters, k) for k in self.kernel_sizes])
        self.projection = torch.nn.Linear(in_features=self.l*len(self.kernel_sizes)*self.filters, out_features=self.n_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.build()

    def _init_input_representations(self):
        if self.mode == "trainable":
            self.embedding_trainable, self.tokenizer = get(model=self.representation, freeze=False)
            self.embeddings_dim = self.embedding_trainable.weight.shape[-1]
        elif self.mode == "untrainable":
            self.embedding_untrainable, self.tokenizer = get(model=self.representation, freeze=True)
            self.embeddings_dim= self.embedding_untrainable.weight.shape[-1]
        elif self.mode =="multichannel":
            self.l = 2
            self.embedding_untrainable, self.tokenizer = get(model=self.representation, freeze=True)
            self.embedding_trainable = torch.nn.Embedding(*self.embedding_untrainable.weight.shape)
            self.embedding_trainable = self.embedding_trainable.from_pretrained(self.embedding_untrainable.weight.clone(),freeze=False)
            self.embeddings_dim = self.embedding_untrainable.weight.shape[-1]
        elif self.mode == "transformer":
            self.transformer, self.tokenizer = get(model=self.representation, output_hidden_states=True)
            self.embeddings_dim = torch.cat(self.transformer(self.transformer.dummy_inputs["input_ids"])[2][-5:-1], -1).shape[-1]

    def forward(self, x):
        if self.mode == "trainable":
            embedded = self.embedding_trainable(x).permute(0, 2, 1)
            embedded = self.dropout(embedded)
            c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]

        elif self.mode == "untrainable":
            with torch.no_grad():
                embedded = self.embedding_untrainable(x).permute(0, 2, 1)
            embedded = self.dropout(embedded)
            c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]

        elif self.mode =="multichannel":
            embedded_1 = self.embedding_trainable(x).permute(0, 2, 1)
            with torch.no_grad():
                embedded_2 = self.embedding_untrainable(x).permute(0, 2, 1)
            embedded_1 = self.dropout(embedded_1)
            embedded_2 = self.dropout(embedded_2)
            c = [torch.nn.functional.relu(conv(embedded_1).permute(0, 2, 1).max(1)[0]) for conv in self.convs]+\
                [torch.nn.functional.relu(conv(embedded_2).permute(0, 2, 1).max(1)[0]) for conv in self.convs]


        elif self.mode == "transformer":
            with torch.no_grad():
                embedded = torch.cat(self.transformer(x)[2][-5:-1], -1).permute(0, 2, 1)
            c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]

        c = torch.cat(c, 1)
        output = self.projection(self.dropout(c))
        return output
