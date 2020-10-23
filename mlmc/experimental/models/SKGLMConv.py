import torch
from ...models.abstracts_graph import TextClassificationAbstractGraph
from ...representation import get, is_transformer
from ...graph import get as gget

class SKGLMConv(TextClassificationAbstractGraph):
    def __init__(self, classes, representation="roberta", max_len=200,channels=100, propagation_layers=2,dropout = 0.5,  n_layers=1, **kwargs):
        super(SKGLMConv, self).__init__(**kwargs)
        # Attributes
        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 100
        self.hidden_dim=512
        self.dropout = dropout
        self.n_layers=n_layers
        self.representation = representation
        self.channels = channels
        self.propagation_layers = propagation_layers

        # Initializations
        self._init_input_representations()
        self.create_labels(classes)

        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.embedding_to_embedding1 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim).to(self.device)
        self.embedding_to_embedding2 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)
        self.embedding_to_embedding3 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)
        import torch_geometric as  torchg

        self.label_embedding_gcn = torchg.nn.GCNConv(in_channels=self.embeddings_dim, out_channels=self.embeddings_dim, node_dim=1).to(self.device)

        self.belief_projection = torch.nn.Linear(in_features=self.max_len, out_features=self.channels).to(self.device)
        self.gcn1 = torch.nn.ModuleList([torchg.nn.GCNConv(in_channels=self.channels, out_channels=self.channels, node_dim=1).to(self.device) for _ in range(propagation_layers)])
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.channels, out_channels=1, node_dim=1).to(self.device)

        self.gcc = torchg.nn.GatedGraphConv(out_channels=self.channels, num_layers=2, node_dim=0).to(self.device)

        self.eye = torch.eye(self.max_len)[None].to(self.device)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.build()


    def forward(self, x, return_graph_scores=False):
        if self.finetune:
            if is_transformer(self.representation):
                embeddings = self.embedding(x)[0]
                label_embeddings = self.embedding(self.label_embeddings)[0]
            else:
                embeddings = self.embedding(x)
                label_embeddings = self.embedding(self.label_embeddings)
        else:
            with torch.no_grad():
                if is_transformer(self.representation):
                    embeddings = self.embedding(x)[0]
                    label_embeddings = self.embedding(self.label_embeddings)[0]
                else:
                    embeddings = self.embedding(x)
                    label_embeddings = self.embedding(self.label_embeddings)
                label_embedding_mask = self.label_embeddings!=0
                label_embeddings = label_embeddings * label_embedding_mask[:,:,None]


            embeddings = self.embedding_to_embedding1(embeddings)
            label_embeddings = self.embedding_to_embedding2(label_embeddings)


        score_matrix = torch.einsum("ijk,lnk->ijln", embeddings, label_embeddings).max(-1)[0]
        torch.cuda.synchronize(self.device)
        beliefs = self.belief_projection(self.dropout_layer(score_matrix.transpose(1,2)))
        # beliefs[:self.n_classes,:self.n_classes]=0

        for m in self.gcn1:
            beliefs = m(beliefs, self.adj)
            beliefs = self.dropout_layer(self.leaky_relu(beliefs))
        beliefs=self.gcn2(beliefs, self.adj)[:,:,0]
        if return_graph_scores:
            return beliefs
        return beliefs[:,:self.n_classes]

    def transform(self, x):
        def clean(x):
            import string
            import re
            x = re.sub("[" + string.punctuation + "]+", " ", x)
            x = re.sub("\s+", " ", x)
            x = "".join([c for c in x.lower() if c not in string.punctuation])
            return x
        return self.tokenizer([clean(sent) for sent in x],self.max_len).to(self.device)
