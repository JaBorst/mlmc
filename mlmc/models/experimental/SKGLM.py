import torch
from ..abstracts_graph import TextClassificationAbstractGraph
from ...representation import get, is_transformer

class SKGLMConv(TextClassificationAbstractGraph):
    def __init__(self, classes, representation="roberta", topk=10, depth=2, max_len=200,
                 graph = ["wordnet"],channels=100,
                 dropout = 0.5, norm=False, n_layers=1, **kwargs):
        super(SKGLMConv, self).__init__(**kwargs)
        # Attributes
        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 100
        self.hidden_dim=512
        self.topk=topk
        self.depth = depth
        self.kernel_sizes = [3,10]
        self.dropout = dropout
        self.n_layers=n_layers
        self.norm = norm
        self.representation = representation
        self.graph = graph
        self.channels = channels

        # Initializations
        self.kb = mlmc.graph.get(graph)
        self._init_input_representations()
        self.create_labels(classes)

        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.kernel_sizes = [3,4,5,6]
        # self.convs = torch.nn.ModuleList([torch.nn.Conv1d(10, self.filters, k) for k in self.kernel_sizes])
        self.embedding_to_embedding1 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim).to(self.device)
        self.embedding_to_embedding2 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)
        self.embedding_to_embedding3 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)
        import torch_geometric as  torchg

        self.label_embedding_gcn = torchg.nn.GCNConv(in_channels=self.embeddings_dim, out_channels=self.embeddings_dim, node_dim=1).to(self.device)


        self.gcn1 = torchg.nn.GCNConv(in_channels=self.max_len, out_channels=self.channels*2, node_dim=1).to(self.device)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.channels*2, out_channels=1, node_dim=1).to(self.device)
        self.act = torch.nn.LeakyReLU().to(self.device)
        self.eye = torch.eye(self.max_len)[None].to(self.device)

        self.build()


    def forward(self, x, return_graph_scores=False):
        self.label_embeddings = self.label_embeddings.to(self.device)
        with torch.no_grad():
            if is_transformer(self.representation):
                embeddings = self.embedding(x)[0]
                label_embeddings = self.embedding(self.label_embeddings)[0]
            else:
                embeddings = self.embedding(x)
                label_embeddings = self.embedding(self.label_embeddings)
            label_embedding_mask = self.label_embeddings!=0
            label_embeddings = label_embeddings * label_embedding_mask[:,:,None]

            label_embeddings = label_embeddings/label_embeddings.norm(dim=-1, p=2, keepdim=True)
            embeddings = embeddings/embeddings.norm(dim=-1, p=2, keepdim=True)

        tem = self.embedding_to_embedding1(embeddings)
        score_matrix = torch.einsum("ijk,lnk->ijln", tem, self.embedding_to_embedding2(label_embeddings)).max(-1)[0]

        selfsim = torch.bmm(embeddings, self.embedding_to_embedding3(embeddings).transpose(2,1)) - self.eye
        torch.cuda.synchronize(self.device)

        selfsim=torch.softmax(selfsim.sum(1), -1)
        score_matrix = self.dropout_layer((score_matrix) * selfsim[:,:,None]).transpose(1,2)
        labelgcn = self.gcn1(score_matrix, self.adj)
        labelgcn = self.dropout_layer(self.act(labelgcn))
        labelgcn = self.gcn2(labelgcn, self.adj)[:,:,0]
        if return_graph_scores:
            return labelgcn
        return labelgcn[:,:self.n_classes]

    def transform(self, x):
        def clean(x):
            import string
            import re
            x = re.sub("[" + string.punctuation + "]+", " ", x)
            x = re.sub("\s+", " ", x)
            x = "".join([c for c in x.lower() if c not in string.punctuation])
            return x
        return self.tokenizer([clean(sent) for sent in x],self.max_len).to(self.device)
