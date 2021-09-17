import torch
import networkx as nx
import mlmc.modules
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract
from ...abstracts.abstracts import TextClassificationAbstract
import random
import re


class GraphModule(torch.nn.Module):
    def __init__(self, graph="wordnet", representation="google/bert_uncased_L-2_H-128_A-2", sformatter = lambda x: f"{x}",device="cpu", *args, **kwargs):
        super(GraphModule, self).__init__(*args, **kwargs)
        self._config = {}
        self._config["graph"] = graph
        self._config["representation"] = representation
        self._config["sformatter"] = sformatter
        self._config["label_len"] = 25
        self._config["device"] = device
        self.graph = mlmc.graph.get(self._config["graph"])
        self.hops = 2
        self.dropout = 0.3
        self.cache = {}

    def build_subgraph(self, classes):
        if set(list(classes)) == set(self.cache):
            pass
        else:
            self.cache = classes
            nodes = [x for x in self.graph for cls in classes.keys() if any([c in x for c in re.split("[-_ /]", cls)])]
            for _ in range(self.hops):
                nodes += sum([list(nx.neighbors(self.graph, n)) for n in nodes],[])
            nodes = list(set(nodes))
            if self.training:
                nodes = random.sample(nodes, int((1-self.dropout) * len(nodes)))
            self.subgraph = nx.subgraph(self.graph, nodes)
        return list(self.subgraph.nodes), torch.tensor(nx.adj_matrix(self.subgraph).toarray()).to(self._config["device"]).float()


class GraphBased(SentenceTextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self,  *args, **kwargs):
        """
         Zeroshot model based on cosine distance of embedding vectors.
        This changes the default activation to identity function (lambda x:x)
        Args:
            mode: one of ("vanilla", "max", "mean", "max_mean", "attention", "attention_max_mean"). determines how the sequence are weighted to build the input representation
            entailment_output: the format of the entailment output if NLI pretraining is used. (experimental)
            *args:
            **kwargs:
        """
        if "act" not in kwargs:
            kwargs["activation"] = lambda x: x
        super(GraphBased, self).__init__(*args, **kwargs)

        self.create_labels(self.classes)
        self.bottle_neck = 1024

        self.parameter = torch.nn.Linear(self.embeddings_dim,self.bottle_neck)
        self.parameter2 = torch.nn.Linear(self.bottle_neck,self.embeddings_dim)


        self.gm = GraphModule(sformatter=self._config["sformatter"], device=self._config["device"])
        import torch_geometric as torchg
        depth = 3
        self.gcn = torch.nn.ModuleList([torchg.nn.GCNConv(1,1, improved=True, normalize=False) for _ in range(depth)])
        self.bottle_neck = 384
        self.entailment_projection = torch.nn.Linear(6 * self.embeddings_dim, self.bottle_neck)
        self.entailment_projection2 = torch.nn.Linear(self.bottle_neck, 3)
        self.dropout = torch.nn.Dropout(0.1)
        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, *args, **kwargs):
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]

        nodes, adj = self.gm.build_subgraph(self.classes)
        nodes_tok = self.transform(nodes)
        nodes_embedding = self.embedding(**nodes_tok)[0]

        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
        nodes_embedding = self._mean_pooling(nodes_embedding, nodes_tok["attention_mask"])
        nodes_embedding = nodes_embedding - nodes_embedding.mean(0, keepdim=True)

        input_scores = torch.mm(input_embedding, nodes_embedding.t())#.softmax(-1)
        label_scores = torch.mm(label_embedding, nodes_embedding.t())#.softmax(-1)

        for gcn in self.gcn:
            input_scores = self.dropout(gcn(input_scores[...,None], torch.stack(torch.where(adj == 1), dim=0))[...,0].relu())#.softmax(-1))
            label_scores = self.dropout(gcn(label_scores[...,None], torch.stack(torch.where(adj == 1), dim=0))[...,0].relu())#.softmax(-1)

        input_knowledge = torch.mm(input_scores.softmax(-1), nodes_embedding)
        label_knowledge = torch.mm(label_scores.softmax(-1), nodes_embedding)

        input_embedding = torch.cat([input_embedding, input_knowledge],-1)
        label_embedding = torch.cat([label_embedding, label_knowledge],-1)

        scores = torch.mm(mlmc.modules.norm(input_embedding), mlmc.modules.norm(label_embedding).t())
        # return r
        r = torch.cat([
            input_embedding[:,None].repeat((1,label_embedding.shape[0],1)),
            label_embedding[None].repeat((input_embedding.shape[0],1,1)),
            torch.abs(input_embedding[:,None] - label_embedding[None])
        ],-1)
        logits = self.entailment_projection2(torch.tanh(self.entailment_projection(r)))
        logits = torch.cat([logits[...,:2],(logits[..., -1] * (1 - scores))[...,None]],-1)

        if self._config["target"] == "entailment":
            pass
        elif self._config["target"] == "single":
            logits = torch.log(logits[..., -1].softmax(-1))
        elif self._config["target"] == "multi":
            logits = torch.log(logits[..., [0, 2]].softmax(-1)[..., -1])
        else:
            assert not self._config["target"], f"Target {self._config['target']} not defined"
        return logits

    def scores(self, x):
        """
        Returns 2D tensor with length of x and number of labels as shape: (N, L)
        Args:
            x:

        Returns:

        """
        self.eval()
        assert not (self._config["target"] == "single" and   self._config["threshold"] != "max"), \
            "You are running single target mode and predicting not in max mode."

        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
            if self._loss_name == "ranking":
                output = 0.5*(output+1)
        self.train()
        return output

    def single(self, loss="ranking"):
        """Helper function to set the model into single label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "single"
        self.set_threshold("max")
        self.set_activation(lambda x: x)
        self._loss_name = loss
        if loss == "ranking":
            self.set_loss(RelativeRankingLoss(0.5))
        else:
            self.set_loss(torch.nn.CrossEntropyLoss())
        self._all_compare=True

    def multi(self, loss="ranking"):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)
        self._loss_name = loss
        if loss == "ranking":
            self.set_loss(RelativeRankingLoss(0.5))
        else:
            self.set_loss(torch.nn.BCELoss)
        self._all_compare=True

    def sts(self):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self._loss_name="ranking"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)
        self.set_loss(RelativeRankingLoss(0.5))

    def entailment(self):
        self._config["target"] = "entailment"
        self.target = "entailment"
        self.set_sformatter(lambda x: x)
        self.set_threshold("max")
        self.set_activation(torch.softmax)
        self.set_loss = torch.nn.CrossEntropyLoss()
        self._all_compare = False
