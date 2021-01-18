import torch
from mlmc.models.abstracts.abstracts_graph import TextClassificationAbstractGraph
import networkx as nx
from ..modules import SKGModule


class SKGLM(TextClassificationAbstractGraph):
    def __init__(self, propagation_layers=3, graph_type="gcn", dropout=0.5, **kwargs):
        """
        Class constructor and initialization of every hyperparameter.

        :param propagation_layers: Number of GCNConv layers. Only used when graph_type="gcn"
        :param graph_type: Determines the convolutional layer used. GCNConv if graph_type="gcn", else GatedGraphConv
        :param dropout: Dropout rate
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstractGraph`
        """
        super(SKGLM, self).__init__(**kwargs)
        # Attributes
        self._config["dropout"] = dropout
        self._config["propagation_layers"] = propagation_layers
        self._config["graph_type"] = graph_type
        # assert channels > max_len, "Channels cannot be smaller than the maximum sequence length"

        # Initializations
        self.create_labels(self.classes)

        self.dropout_layer = torch.nn.Dropout(self._config["dropout"])
        self.embedding_to_embedding1 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)
        self.embedding_to_embedding2 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)
        self.embedding_to_embedding3 = torch.nn.Linear(in_features=self.embeddings_dim,
                                                       out_features=self.embeddings_dim)

        self.skg_module = SKGModule(self.embeddings_dim, self.embeddings_dim, sequence_length=self._config["max_len"], graph_type=graph_type,
                                    propagation_layers=propagation_layers)

        self.build()

    def forward(self, x, return_graph_scores=False):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        embeddings = self.embed_input(x)
        label_embeddings = self.embed_input(self.label_embeddings)
        if not self.finetune:
            embeddings = self.embedding_to_embedding1(embeddings)
            label_embeddings = self.embedding_to_embedding2(label_embeddings)

        beliefs = self.skg_module(embeddings, self.embedding_to_embedding3(label_embeddings), self.adj.to(x.device))
        return beliefs

    def transform(self, x):
        def clean(x):
            import string
            import re
            x = re.sub("[" + string.punctuation + "]+", " ", x)
            x = re.sub("\s+", " ", x)
            x = "".join([c for c in x.lower() if c not in string.punctuation])
            return x

        return self.tokenizer([clean(sent) for sent in x], self.max_len).to(self.device)

    def create_labels(self, classes):
        self.classes = classes
        self._config["classes"] = classes
        self.n_classes = len(classes)

        ind = [list(self.kb.nodes).index(k) for k in classes.keys()]
        self.adjacency = torch.from_numpy(nx.adjacency_matrix(self.kb).toarray()).float()

        self.adjacency = torch.stack([self.adjacency[i] for i in ind])
        self.adjacency = torch.stack([self.adjacency[:, i] for i in ind], 1)
        self.adjacency = self.adjacency + torch.eye(self.adjacency.shape[0])
        self.adj = torch.stack(torch.where(self.adjacency.t() == 1), dim=0).to(self.device)

        if not hasattr(self, "label_dict"):
            self.label_embeddings = self.tokenizer(list(self.classes.keys()), 10)
        self.label_embeddings = self.label_embeddings.to(self.device)
