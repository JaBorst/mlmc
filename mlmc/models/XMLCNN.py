import torch
from .abstracts import TextClassificationAbstract
from ..representation import get
##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class XMLCNN(TextClassificationAbstract):
    def __init__(self, classes, mode="trainable", representation="roberta", max_len=500, **kwargs):
        super(XMLCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.modes = ("trainable", "untrainable", "multichannel", "transformer")
        self.mode = mode
        self.l = 1
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        assert self.mode in self.modes, "%s not in (%s, %s, %s, %s)" % (self.mode, *self.modes)

        self._init_input_representations()
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embeddings_dim, 100, k) for k in self.kernels ])

        self.dynpool = torch.nn.MaxPool2d(( 1, self.window))
        self.bottleneck = torch.nn.Linear(
            in_features=100*( 2*(sum([(self.maxlen-s+1)// self.window  for s in self.kernels]))),
            out_features=30)
        self.projection = torch.nn.Linear(in_features=30, out_features=self.n_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.sf = torch.nn.Sigmoid()
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
        elif self.mode == "untrainable":
            with torch.no_grad():
                embedded = self.embedding_untrainable(x).permute(0, 2, 1)
            embedded = self.dropout(embedded)
        elif self.mode == "multichannel":
            embedded_1 = self.embedding_trainable(x).permute(0, 2, 1)
            with torch.no_grad():
                embedded_2 = self.embedding_untrainable(x).permute(0, 2, 1)
            embedded_1 = self.dropout(embedded_1)
            embedded_2 = self.dropout(embedded_2)
            embedded = torch.cat([embedded_1, embedded_2], dim=-1)
        elif self.mode == "transformer":
            with torch.no_grad():
                embedded = torch.cat(self.transformer(x)[2][-5:-1], -1).permute(0, 2, 1)
        c = [torch.nn.functional.relu(self.dynpool(conv(embedded))) for conv in self.convs]
        concat=self.dropout(torch.cat(c,1).view(x.shape[0],-1))
        bn = torch.relu(self.bottleneck(concat))
        output = self.projection(bn)
        return output


