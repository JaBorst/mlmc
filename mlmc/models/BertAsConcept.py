"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
from ..layers import Bilinear, AttentionWeightedAggregation

class BertAsConcept(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation="roberta", label_freeze=True, max_len=300, **kwargs):
        super(BertAsConcept, self).__init__(**kwargs)
        #My Stuff
        assert is_transformer(representation), "This model only works with transformers"


        self.classes = classes
        self.max_len = max_len
        self.n_layers = 2
        self.representation = representation
        self._init_input_representations()
        # Original
        self.n_classes = len(classes)
        self.label_freeze = label_freeze

        self.labels = torch.nn.Parameter(self.embedding(self.transform(self.classes.keys()))[1])
        self.labels.requires_grad=False
        self.label_embedding_dim =self.labels.shape[-1]

        self.input_projection = torch.nn.Linear(self.embedding_dim, 1024)
        self.input_projection2 = torch.nn.Linear(self.label_embedding_dim, 1024)


        self.output_projection = torch.nn.Linear(in_features=self.max_len*self.n_classes, out_features=self.n_classes)
        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)

        p1 = self.input_projection(embeddings)
        p2 = self.input_projection2(self.labels)

        p1 = p1 / p1.norm(p=2,dim=-1,keepdim=True)
        p2 = p2 / p2.norm(p=2,dim=-1,keepdim=True)

        metric_scores = torch.matmul(p1,p2.t()) #self.metric(p1,p2)#+ torch.softmax(self.beta*,-1)

        # metric_based_representation = torch.matmul(metric_scores, self.concepts)
        # metric_based_representation = metric_based_representation/metric_based_representation.norm(p=2, dim=-1, keepdim=True)

        # label_scores = self.metric2(metric_based_representation, self.labels)

        # output = self.output_projection(label_scores.flatten(1,2))

        # output = torch.matmul(metric_scores, self.score_filter).sum(-2)
        output = metric_scores.sum(-2)
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

        prediction, scores = self(self.transform(x).to(self.device), return_scores=True)
        label = [self.classes_rev[x.item()] for x in torch.where(self.threshold(prediction, 0.5,"hard")==1)[1]]
        tk = (scores.sum(-2)).topk(k)[1][0]
        concepts = [label_vocabulary_rev[x.item()] for x in tk]

        print("Labels:\t", label)
        print("Concepts:\t", concepts)


class BertAsConcept2(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """

    def __init__(self, classes, representation="roberta", label_freeze=True, max_len=300, **kwargs):
        super(BertAsConcept2, self).__init__(**kwargs)
        # My Stuff
        assert is_transformer(representation), "This model only works with transformers"

        self.classes = classes
        self.max_len = max_len
        self.n_layers = 2
        self.representation = representation
        self._init_input_representations()
        # Original
        self.n_classes = len(classes)
        self.label_freeze = label_freeze

        self.labels = torch.nn.Parameter(self.embedding(self.transform(self.classes.keys()))[1])
        self.labels.requires_grad = False
        self.label_embedding_dim = self.labels.shape[-1]

        self.input_projection2 = torch.nn.Linear(self.label_embedding_dim, self.embedding_dim)
        self.metric = Bilinear(self.embedding_dim).to(self.device)
        self.output_projection = torch.nn.Linear(in_features=self.max_len , out_features=1)
        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)

        p2 = self.input_projection2(self.labels)

        output = self.output_projection(torch.matmul(embeddings,p2.t()).permute(0,2,1)).squeeze()
        # output = self.metric(embeddings,p2).sum(-2)
        if return_scores:
            return output#, metric_scores
        return output

    def additional_concepts(self, x, k=5):
        self.eval()
        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}

        label_vocabulary_rev = {v: k for k, v in self.label_vocabulary.items()}

        prediction, scores = self(self.transform(x).to(self.device), return_scores=True)
        label = [self.classes_rev[x.item()] for x in torch.where(self.threshold(prediction, 0.5, "hard") == 1)[1]]
        tk = (scores.sum(-2)).topk(k)[1][0]
        concepts = [label_vocabulary_rev[x.item()] for x in tk]

        print("Labels:\t", label)
        print("Concepts:\t", concepts)


class BertAsConcept3(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation="roberta", label_freeze=True, max_len=300, **kwargs):
        super(BertAsConcept3, self).__init__(**kwargs)
        # My Stuff
        assert is_transformer(representation), "This model only works with transformers"

        self.max_len = max_len
        self.n_layers = 2
        self.representation = representation
        self._init_input_representations()
        # Original
        self.n_classes = len(classes)
        self.label_freeze = label_freeze
        self.d_a = 1024

        self.classes = classes
        self.labels = torch.nn.Parameter(self.embedding(self.transform(self.classes.keys()))[1])
        self.labels.requires_grad = False
        self.label_embedding_dim = self.labels.shape[-1]


        self.input_projection2 = torch.nn.Linear(self.label_embedding_dim, self.embedding_dim)
        self.metric = Bilinear(self.embedding_dim).to(self.device)
        self.output_projection = torch.nn.Linear(in_features=self.max_len , out_features=1)

        self.att = AttentionWeightedAggregation(in_features = self.embedding_dim, d_a=self.d_a)

        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)

        p2 = self.input_projection2(self.labels)
        label_scores = torch.matmul(embeddings,p2.t())

        output, att = self.att(embeddings, label_scores, return_att=True)
        if return_scores:
            return output, label_scores, att
        return output

    def create_labels(self, classes):
        if hasattr(self, "labels"):
            del self.labels
        self.classes = classes
        self.labels = torch.nn.Parameter(self.embedding(self.transform(self.classes.keys()).to(self.device))[1])
        self.labels.requires_grad = False
        self.label_embedding_dim = self.labels.shape[-1]
