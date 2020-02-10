"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
import re
from ..representation.labels import makemultilabels
from ..layers import Bilinear

class GloveConcepts(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, concepts, label_vocabulary, representation="roberta", label_freeze=True, max_len=300, **kwargs):
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
        self.compare_space_dim = 256


        self.concepts=torch.nn.Parameter(torch.from_numpy(concepts).float())
        self.concepts.requires_grad=False

        label_embed = [[self.label_vocabulary.get(w, 0) if w != "comdedy" else self.label_vocabulary["comedy"] for w in
                        re.split("[ ,'-]", x.lower())] for x in self.classes.keys()]
        # label_embed = [torch.stack([self.concepts[x] for x in label ],0) for label in label_embed]
        # label_embed = [x.max(0)[0] for x in label_embed]

        self.label_concept_onehot = makemultilabels(label_embed, len(self.label_vocabulary)).float()
        self.label_concept_onehot = self.label_concept_onehot / self.label_concept_onehot.norm(p=2,dim=-1,keepdim=True)
        self.labels = torch.matmul(self.label_concept_onehot, self.concepts)
        # self.labels = self.labels / self.labels.norm(p=2, dim=-1)[:, None]
        self.labels[torch.isnan(self.labels.norm(dim=-1))] = 0

        self.labels = torch.nn.Parameter(self.labels)
        self.labels.requires_grad = False

        self.scaling = torch.nn.Parameter(torch.sqrt(torch.Tensor([max_len])))
        self.scaling.requires_grad=False

        self.input_projection = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.input_projection2 = torch.nn.Linear(self.concept_embedding_dim, self.embedding_dim)

        self.beta = torch.nn.Parameter(torch.sqrt(torch.Tensor([self.n_concepts])))
        self.beta.requires_grad = False
        # self.projection_importance_key = torch.nn.Linear(self.embedding_dim, self.compare_space_dim)
        # self.projection_importance_query = torch.nn.Linear(self.embedding_dim, self.compare_space_dim)
        # self.projection_importance = torch.nn.Linear(self.max_len*self.max_len, self.max_len)


        self.metric = Bilinear(self.embedding_dim)
        self.metric2 = Bilinear(self.concept_embedding_dim).to(self.device)
        self.output_projection = torch.nn.Linear(in_features=self.max_len*self.n_classes, out_features=self.n_classes)
        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
        p1 = self.input_projection(embeddings)
        p2 = self.input_projection2(self.concepts)

        metric_scores = torch.softmax(self.beta*self.metric(p1,p2),-1)

        metric_based_representation = torch.matmul(metric_scores, self.concepts)
        metric_based_representation = metric_based_representation/metric_based_representation.norm(p=2, dim=-1, keepdim=True)

        label_scores = self.metric2(metric_based_representation, self.labels)

        output = self.output_projection(label_scores.flatten(1,2))

        if return_scores:
            return output, metric_scores
        return output

    # def regularize(self):
    #     return self.metric.regularize()

    def additional_concepts(self, x, k=5):
        self.eval()
        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}

        label_vocabulary_rev = {v: k for k, v in self.label_vocabulary.items()}

        prediction = self(self.transform(x).to(self.device), return_scores=True)
        label = [self.classes_rev[x.item()] for x in torch.where(self.threshold(prediction[0], 0.5,"hard")==1)[1]]
        tk = prediction[1].sum(-2).topk(k)[1][0]
        concepts = [label_vocabulary_rev[x.item()] for x in tk]

        print("Labels:\t", label)
        print("Concepts:\t", concepts)