import torch

from .abstracts import TextClassificationAbstract
from .abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..graph import get as graph_get
import networkx as nx
from ..graph import subgraphs


class TextClassificationAbstractGraph(TextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def __init__(self, graph, topk=20, depth=2, **kwargs):
        """
        Abstract initializer of a Text Classification network.
        Args:
            target: single label oder multilabel mode. defined by keystrings: ("single", "multi"). Sets some basic options, like loss function, activation and
                    metrics to sensible defaults.
            activation: The activation function applied to the output. Only used for metrics and when you want to return scores in predict. (default: torch.softmax for "single", torch.sigmoid for "multi")
            loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss for "multi" and torch.nn.CrossEntropyLoss for "single")
            optimizer:  One of toch.optim (default: torch.optim.Adam)
            optimizer_params: A dictionary of optimizer parameters
            device: torch device, destination of training (cpu or cuda:0)
        """
        super(TextClassificationAbstractGraph,self).__init__(**kwargs)


        if isinstance(graph, str) or isinstance(graph, list):
            self.topk = topk
            self.depth = depth
            self.graph = graph
            self.kb = graph_get(graph)
        elif isinstance(graph, nx.Graph):
            self.kb = graph


    def create_labels(self, classes):
        # ToDo still needs to be checked for sort order
        self.label_subgraph = subgraphs(self.classes, self.kb, model="glove300", topk=self.topk,
                                           depth=self.depth, device=self.device)
        self.classes = classes
        self.n_classes = len(classes)
        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.label_subgraph).toarray()).float()
        tmp_adj = tmp_adj + torch.eye(tmp_adj.shape[0])

        self.adj = torch.stack(torch.where(tmp_adj.t() == 1), dim=0).to(self.device)
        self.label_embeddings = self.tokenizer(list(self.label_subgraph.nodes), pad=True, maxlen=10).to(self.device)
        self.label_embeddings_dim = self.embeddings_dim
        self.no_nodes = len(self.label_subgraph)


    def complete_score_graph(self,x, n=20, method="hard"):
        x = [x] if isinstance(x, str) else x
        scores = self.act(self(self.transform(x), return_graph_scores=True))
        from copy import deepcopy
        new_graph = deepcopy(self.label_subgraph)
        for sentence, sentence_scores, ones in zip(x, scores, self.threshold(scores, method=method)):
            keywords = str([list(self.label_subgraph.nodes)[i] for score, i in zip(*sentence_scores.topk(n))])

            new_graph.add_node(sentence,
                               prediction={list(self.classes)[i]: sentence_scores[i].cpu().item() for i in torch.where(ones[:self.n_classes]==1)[0]},
                               type="text",
                               keywords=keywords)


            for score, i in zip(*sentence_scores.topk(n)):
                new_graph.add_edge(sentence,list(self.label_subgraph.nodes)[i], weight=score.cpu().item(), type="label")
        return new_graph


    def score_graph(self,x, n=20, method="mcut"):
        x = [x] if isinstance(x, str) else x
        scores = self.act(self(self.transform(x), return_graph_scores=True))
        new_graph = nx.DiGraph()
        for sentence, sentence_scores, ones in zip(x, scores, self.threshold(scores, method=method)):
            keywords = str([list(self.label_subgraph.nodes)[i] for score, i in zip(*sentence_scores.topk(n)) if i > self.n_classes])

            new_graph.add_node(sentence,
                               prediction={list(self.classes)[i]: sentence_scores[i].cpu().item() for i in
                                           torch.where(ones[:self.n_classes] == 1)[0]},
                               type="text",
                               keywords=keywords)
            for score, i in zip(*sentence_scores.topk(n)):
                new_graph.add_edge(sentence, list(self.label_subgraph.nodes)[i], weight=score.cpu().item(),
                                   type="label")
        return new_graph

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
        return self.tokenizer(x, maxlen=self.max_len).to(self.device)
