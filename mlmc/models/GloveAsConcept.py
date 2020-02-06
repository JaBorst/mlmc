"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
import re
from ..representation.labels import makemultilabels
import torch_geometric as torch_g


class GloveConcepts(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, concepts, label_vocabulary, representation="roberta", label_freeze=True, max_len=400, **kwargs):
        super(GloveConcepts, self).__init__(**kwargs)
        #My Stuff
        assert is_transformer(representation), "This model only works with transformers"


        self.classes = classes
        self.max_len = max_len
        self.n_layers = 2
        self.concept_embedding_dim = concepts.shape[-1]
        self.n_concepts = concepts.shape[0]
        self.representation = representation
        self._init_input_representations()
        # Original
        self.n_classes = len(classes)
        self.label_freeze = label_freeze
        self.label_vocabulary = label_vocabulary
        self.att_dim = 256

        self.concepts=torch.nn.Parameter(torch.from_numpy(concepts).float())
        self.concepts.requires_grad=False

        label_embed = [[self.label_vocabulary.get(w,0) if w != "comdedy" else self.label_vocabulary["comedy"] for w in re.split("[ ,'-]",x.lower())] for x in self.classes.keys()]
        # label_embed = [torch.stack([self.concepts[x] for x in label ],0) for label in label_embed]
        # label_embed = [x.max(0)[0] for x in label_embed]


        self.label_concept_onehot = makemultilabels(label_embed, len(self.label_vocabulary)+1).float()
        self.labels = torch.matmul(self.label_concept_onehot, self.concepts)
        self.labels = self.labels / self.labels.norm(p=2, dim=-1)[:, None]
        self.labels[torch.isnan(self.labels.norm(dim=-1))] = 0

        self.labels = torch.nn.Parameter(self.labels)
        self.labels.requires_grad = False


        # self.query_projection = torch.nn.Linear(self.embedding_dim,200)
        # self.key_projection = torch.nn.Linear(self.embedding_dim,200)

        self.input_projection = torch.nn.Linear(self.max_len*self.embedding_dim, 600)
        self.input_projection2 = torch.nn.Linear(300, 600)
        # self.concept_projection = torch.nn.Linear(2048, self.n_classes)
        # self.output_projection = torch.nn.Linear(300, self.n_classes,bias=False)
        # self.output_projection.weight.requires_grad =False
        # with torch.no_grad():
        #     self.output_projection.weight.copy_(self.concept_aggregation_labels)

        # self.importance = torch.nn.Linear(self.embedding_dim,1).to(self.device)
        #
        # self.query_projection = torch.nn.Linear(self.embedding_dim, self.att_dim)
        # self.key_projection = torch.nn.Linear(self.embedding_dim, self.att_dim)
        # self.scaling = torch.nn.Parameter(torch.sqrt(torch.Tensor([1./400.])))

        # self.comparing_space_doc = torch.nn.Linear(self.n_concepts, 256)
        # self.comparing_space_label = torch.nn.Linear(self.n_concepts, 256)

        # self.ggc = torch_g.nn.GraphConv(self.n_concepts, self.n_concepts).to(self.device)
        # self.metric_tensor = torch.nn.Parameter(
        #     torch.triu(torch.rand((self.concept_embedding_dim, self.concept_embedding_dim)))#+  self.concept_embedding_dim*torch.eye(self.concept_embedding_dim)
        # )
        # self.metric_tensor.requires_grad=True
        # torch.nn.init.kaiming_normal_(self.metric_tensor)
        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
            embeddings = embeddings.flatten(1,2)
        p = self.input_projection(embeddings)
        p2= self.input_projection2(self.labels)
        return torch.matmul(p,p2.t())

    # def regularize(self):
    #     lower_tri = torch.triu(self.metric_tensor).t()
    #     dev = torch.relu((lower_tri.sum(-1) - 2 * lower_tri.diag()))
    #     return  dev.sum() + torch.relu(-lower_tri.diag()).sum()
