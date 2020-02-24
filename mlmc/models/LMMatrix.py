"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from ..models.abstracts import TextClassificationAbstract
from ..representation import get, is_transformer

class LMMatrix(TextClassificationAbstract):
    def __init__(self, classes,   adjacency, representation="roberta", max_len=200, norm=False, dropout = 0.5, **kwargs):
        super(LMMatrix, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 300
        self.hidden_dim=512
        self.kernel_sizes = [3,10]
        self.dropout = dropout
        self.adjacency = adjacency
        self.adjacency_param = torch.nn.Parameter(torch.from_numpy(adjacency).float())
        self.adjacency_param.requires_grad = False
        self.n_layers=2
        self.representation = representation
        self._init_input_representations()
        for param in self.embedding.parameters(): param.requires_grad = True
        self.norm=norm
        import string
        label_names = ["".join([c if c in string.ascii_letters else " " for c in k]) for k in self.classes.keys()]
        self.label_tokens = self.tokenizer(label_names, 6)

        self.label_mask = self.tokenizer(label_names, 6)
        self.label_mask[self.label_mask==1]=0
        self.label_mask[self.label_mask!=0]=1
        self.label_mask = torch.nn.Parameter(self.label_mask.float())
        self.label_mask.requires_grad = False

        # self.document_projection = torch.nn.Linear(self.embedding_dim, self.label_embeddings_dim)


        # self.dropout_layer= torch.nn.Dropout(self.dropout)
        # import torch_geometric as torchg
        # self.gcn1 = torchg.nn.GCNConv(in_channels=self.label_embeddings_cls_dim, out_channels=self.hidden_dim)
        # self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.projection = torch.nn.Linear(in_features=768, out_features=self.embedding_dim).to(self.device)
        # self.projection2 = torch.nn.Linear(in_features=self.max_len, out_features=1).to(self.device)

        self.build()
        self.label_tokens = self.label_tokens.to(self.device)

    def forward(self, x, y=None, return_scores=False):
        # embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
        embeddings = self.embedding(x)[1]
        label_embeddings = self.embedding(self.label_tokens)[1] #[2][(-1 - self.n_layers):-1], -1)

        embeddings = self.projection(embeddings)
        # embeddings = embeddings - embeddings.mean(-1,keepdim=True)
        label_embeddings = self.projection(label_embeddings)

        if self.norm:
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            label_embeddings = label_embeddings / label_embeddings.norm(p=2, dim=-1, keepdim=True)
        a = torch.einsum("ik,mk->im",embeddings, label_embeddings)
        # a = torch.relu(a - a.mean([-1], keepdim=True))
        # a = a * self.label_mask.unsqueeze(0).unsqueeze(0)
        # a = (a*self.label_mask.unsqueeze(0).unsqueeze(0).to(self.device)).permute(0, 2, 1, 3)
        # label_scores = a.sum(-2)

        return a
