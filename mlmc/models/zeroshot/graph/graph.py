import networkx as nx

from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
import torch
from mlmc.modules.dropout import VerticalDropout
from mlmc.modules.module_tfidf import TFIDFAggregation
from mlmc.graph import Graph

class GraphBased(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    def __init__(self, similarity="cosine", dropout=0.5,
                 measures=None,depth=1,fallback_classifier = False,
                 graph="wordnet", *args, **kwargs):
        super(GraphBased, self).__init__(*args, **kwargs)

        self._config["fallback_classifier"] = fallback_classifier
        self.dropout = torch.nn.Dropout(dropout)
        from ....graph.helpers import keywordmap
        self.graph = Graph(graph, depth = depth, map=keywordmap)

        self._config["dropout"] = dropout
        self._config["similarity"] = similarity
        self.agg = TFIDFAggregation()

        self.create_labels(self.classes)
        self.vdropout = VerticalDropout(0.5)
        self._classifier_weight = torch.nn.Parameter(torch.tensor([0.01]))
        self.build()


    def create_labels(self, classes: dict):
        super().create_labels(classes)
        self.graph(classes)
        self._node_list, self.class_nodes_index, self.adjacency, self.class_adjacency  = self.graph.get(self.device)
        self.nodes = self.transform([x.replace("_", " ") for x in self._node_list])

    def _sim(self, x, y):
        if self._config["similarity"] == "cosine":
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            y = y / (y.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            r = torch.matmul(x,y.t())#(x.unsqueeze(-2) * y.unsqueeze(-3) ).sum(-1)
            r = torch.log(0.5 * (r + 1))
        elif self._config["similarity"] == "scalar":
            r = (x[:, None] * y[None]).sum(-1)
        elif self._config["similarity"] == "manhattan":
            r = - (x[:, None] * y[None]).abs().sum(-1)
        return r

    def transform(self, x, max_length=None) -> dict:
        if max_length is None:
            max_length = self._config["max_len"]
        r = {k: v.to(self.device) for k, v in
                self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt').items()}
        r["text"] = [self.tokenizer.tokenize(s) for s in x]
        return r

    def forward(self, x, kw=False):
        input_embedding_t = self.dropout(self.embedding(**{k:x[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if k in x})[0])
        label_embedding = self.dropout(self.embedding(**{k:self.label_dict[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if k in self.label_dict})[0])
        nodes_embedding = self.dropout(self.embedding(**{k:self.nodes[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if k in self.nodes})[0])


        if self.training:
            input_embedding_t = input_embedding_t + 0.01*torch.rand_like(input_embedding_t)[:,0,None,0,None].round()*torch.rand_like(input_embedding_t) #
            input_embedding_t = input_embedding_t * ((torch.rand_like(input_embedding_t[:,:,0])>0.05).float()*2 -1)[...,None]

        input_embedding=input_embedding_t


        nodes_embedding = self._mean_pooling(nodes_embedding, self.nodes["attention_mask"])
        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])

        pooled_similarity = self._sim(input_embedding, label_embedding).squeeze(-1)  # pooled-similarity
        nodes_embedding_norm = nodes_embedding / nodes_embedding.norm(2, dim = -1, keepdim=True)
        input_embedding_t = input_embedding_t / input_embedding_t.norm(2, dim = -1, keepdim=True)
        label_embedding = label_embedding / label_embedding.norm(2, dim = -1, keepdim=True)



        in_distr = torch.matmul(input_embedding_t, nodes_embedding_norm.t())#.sum(1)#max(1)[0]
        with torch.no_grad():
            mean = (in_distr.mean((-1), keepdim=True)).float()
            std = (in_distr.std((-1), keepdim=True)).float()
            mask = (in_distr > (mean + 5*std))#.to_sparse()
            mask = (mask * x["attention_mask"].unsqueeze(-1)).float()#.to_sparse()
        in_distr = (in_distr*mask.detach()).sum(1)

        # node_rep = torch.mm(in_distr/in_distr.sum(-1,keepdim=True), nodes_embedding)
        # node_sim = torch.matmul(node_rep, label_embedding.t()).log_softmax(-1)


        sim=( in_distr[:, None] * self.class_adjacency.to_dense()[None]).sum(-1) #(mask.sum([1,2]).unsqueeze(-1))*

        # t_distr = torch.matmul(input_embedding_t, label_embedding.t())#.sum(1)#max(1)[0]
        # with torch.no_grad():
        #     # t_distr = torch.matmul(input_embedding_t, label_embedding.t())  # .sum(1)#max(1)[0]
        #     t_mean = (t_distr.mean((1,2), keepdim=True)).float()
        #     t_std = (t_distr.std((1,2), keepdim=True)).float()
        #     t_mask = (t_distr > (t_mean +  3*t_std))  # .to_sparse()
        #     t_topmask = torch.nn.functional.one_hot(t_distr.argmax(-1), len(self.classes))
        #     t_mask = t_mask* t_topmask * x["attention_mask"].unsqueeze(-1)
        #     t_mask.sum(1)
        # sim3 = (t_distr*t_mask.detach()).sum(1) #/ (t_mask.sum(1) + 1e-6)

        # l = [(0.5*(1+sim)).log(), (0.5*(1+sim3)).log(), pooled_similarity]#.log_softmax(-1)
        l = [(0.5*(1+sim)).log(), pooled_similarity]#, (0.5*(1+sim)).log(),(0.5*(1+sim3)).log(),]#.log_softmax(-1)
        scores=torch.stack(l, -1).mean(-1)#.log_softmax(-1)
        if kw:
            return scores, in_distr, mask
        return scores

    def words(self, x, n = 10):
        self.eval()
        with torch.no_grad():
            tok = self.transform(x)
            scores, graphkw, t_mask = self.forward(tok, kw=True)
        for b, text in enumerate(tok["text"]):
            print(" ".join(text))
            cls = scores.argmax(-1)[b].item()
            print([self._node_list[i.item()] for i in (graphkw).topk(n, dim=-1)[1][b]])
            print(list(self.classes.keys())[cls])
            print("\n\n")
        for b, (text, tkw) in enumerate(zip(tok["text"],t_mask)):
            for i, (token, tk) in enumerate(zip(text,tkw[1:])):
                print(token, [self._node_list[x.item()] for x in torch.where(tk)[0]])

            print(" ".join(text))
            cls = scores.argmax(-1)[b].item()
            print([self._node_list[i.item()] for i in (graphkw).topk(10, dim=-1)[1][b]])
            print(list(self.classes.keys())[cls])
            print("\n\n")

    def keywords(self, x, n = 10):
        self.eval()
        with torch.no_grad():
            tok = self.transform(x)
            scores, graphkw, text_kw = self.forward(tok, kw=True)

        gkw = [[list(self._node_list)[i] for i in x] for x in graphkw.softmax(-1).topk(n, dim=-1, )[1]]
        labels = [[self.classes_rev[x.item()]] for x in scores.argmax(-1).cpu()]
        keywords = [list(zip(t, l[1:(1 + len(t))].tolist())) for t, l in zip(tok["text"], text_kw)]
        keywords_new = []
        for l in keywords:
            new_list = []
            new_tuple = [[], 0]
            for i in range(1, 1 + len(l)):
                new_tuple[0].append(l[-i][0])
                new_tuple[1] += l[-i][1]

                if not l[-i][0].startswith("##"):
                    new_tuple[0] = "".join(new_tuple[0][::-1]).replace("##", "")
                    new_list.append(tuple(new_tuple))
                    new_tuple = [[], 0]
            keywords_new.append(new_list)
        tkw = [sorted(x, key=lambda x: -x[1])[:n] for x in keywords_new]
        return labels, gkw, tkw