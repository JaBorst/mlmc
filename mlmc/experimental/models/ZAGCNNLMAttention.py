"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from mlmc.models.abstracts.abstracts_graph import TextClassificationAbstractGraph
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.representation import Embedder

class ZAGCNNAttention(TextClassificationAbstractGraph,TextClassificationAbstractZeroShot):
    def __init__(self, classes, representation="roberta", max_len=200,channels=100,
                 dropout = 0.5, norm=False, n_layers=1, **kwargs):
        super(ZAGCNNAttention, self).__init__(**kwargs)
        # Attributes
        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 100
        self.dropout = dropout
        self.n_layers=n_layers
        self.norm = norm
        self.representation = representation
        self.channels = channels

        # Initializations
        self._init_input_representations()
        self.create_labels(classes)

        self.e = Embedder("glove300")
        self.label_gloves = self.e.embed(list(self.label_subgraph.nodes()), pad=20).mean(1)

        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.projection = torch.nn.Linear(in_features=self.max_len, out_features=len(self.label_subgraph)).to(self.device)
        self.projection2 = torch.nn.Linear(in_features=self.embeddings_dim, out_features=self.channels).to(self.device)

        self.embedding_to_embedding1 = torch.nn.MultiheadAttention(embed_dim=self.embeddings_dim,
                                                       num_heads=1).to(self.device)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=self.channels, out_channels=self.channels, node_dim=0).to(self.device)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.channels, out_channels=self.channels, node_dim=0).to(self.device)
        self.act = torch.nn.LeakyReLU()
        # self.eye = torch.eye(self.max_len)[None]

        self.build()


    def forward(self, x, return_graph_scores=False):
        if self.finetune:
            embeddings = self.embedding(x[0])[0]
            label_embeddings = self.embedding(self.label_embeddings)[0].mean(1)

        else:
            with torch.no_grad():
                embeddings = self.embedding(x[0])[0]
                label_embeddings = self.embedding(self.label_embeddings)[0].mean(1)

        emb, att = self.embedding_to_embedding1(embeddings.transpose(0, 1), embeddings.transpose(0, 1), embeddings.transpose(0, 1))
        emb = emb.transpose(0,1)

        labelgcn = self.gcn1(self.projection2(label_embeddings), self.adj)
        labelgcn = self.dropout_layer(self.act(labelgcn))
        labelgcn = self.gcn2(labelgcn, self.adj)

        graph_scores = torch.einsum("jk,ink-> inj", labelgcn, self.projection2(emb)).mean(1)

        if return_graph_scores:
            return graph_scores
        return graph_scores[:, :self.n_classes]


    def transform(self, x):
        """
        A standard transformation function from text to network input format

        The function looks for the tokenizer attribute. If it doesn't exist the transform function has to
        be implemented in the child class

        Args:
            x: A list of text

        Returns:
            A tensor in the network input format.

        """
        assert hasattr(self, 'tokenizer'), "If the model does not have a tokenizer attribute, please implement the" \
                                           "transform(self, x)  method yourself. Tokenizer can be allocated with " \
                                           "embedder, tokenizer = mlmc.helpers.get_embedding() or " \
                                           "embedder, tokenizer = mlmc.helpers.get_transformer()"
        return (self.tokenizer(x, maxlen=self.max_len).to(self.device), self.glove_scores(x).to(self.device))

    def glove_scores(self, x):
        ie = self.e.embed(x, pad=self.max_len)
        return torch.einsum("ijk,lk-> ijl", ie, self.label_gloves)




