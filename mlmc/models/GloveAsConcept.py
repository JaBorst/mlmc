"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
import re
from ..representation.labels import makemultilabels
from ..layers import Bilinear
from ignite.metrics import Average
from tqdm import tqdm

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

        self.input_projection = torch.nn.Linear(self.embedding_dim, 1024)
        self.input_projection2 = torch.nn.Linear(self.concept_embedding_dim, 1024)

        # self.beta = torch.nn.Parameter(torch.sqrt(torch.Tensor([self.n_concepts])))
        # self.beta.requires_grad = False
        # self.projection_importance_key = torch.nn.Linear(self.embedding_dim, self.compare_space_dim)
        # self.projection_importance_query = torch.nn.Linear(self.embedding_dim, self.compare_space_dim)
        # self.projection_importance = torch.nn.Linear(self.max_len*self.max_len, self.max_len)

        # self.att = torch.nn.MultiheadAttention(1024, 1).to(self.device)
        # self.metric = Bilinear(1024)
        # self.metric2 = Bilinear(self.concept_embedding_dim)
        # self.label_concept_onehot[self.label_concept_onehot == 0] = -1

        # self.score_filter = torch.nn.Parameter((1/self.n_concepts)*self.label_concept_onehot.t())
        # self.score_filter.requires_grad=True

        # self.scores_trafo = torch.nn.Linear(in_features=self.n_concepts, out_features=self.n_concepts)
        self.output_projection = torch.nn.Linear(in_features=self.max_len*self.n_classes, out_features=self.n_classes)
        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
            # embeddings = self.embedding(x)[1]

        p1 = self.input_projection(embeddings)
        p2 = self.input_projection2(self.labels)
        # p1, att = self.att(p1, p2[:,None,:].repeat(1,x.shape[0],1),p2[:,None,:].repeat(1,x.shape[0],1))
        # p1 = p1.permute(1, 0, 2)

        p1 = p1 / p1.norm(p=2,dim=-1,keepdim=True)
        p2 = p2 / p2.norm(p=2,dim=-1,keepdim=True)

        metric_scores = torch.matmul(p1,p2.t()) #self.metric(p1,p2)#+ torch.softmax(self.beta*,-1)
        # metric_scores = self.scores_trafo(metric_scores)

        # metric_based_representation = torch.matmul(metric_scores, self.concepts)
        # metric_based_representation = metric_based_representation/metric_based_representation.norm(p=2, dim=-1, keepdim=True)

        # label_scores = self.metric2(metric_based_representation, self.labels)

        # output = self.output_projection(label_scores.flatten(1,2))

        # output = torch.matmul(metric_scores, self.score_filter).sum(-2)
        output = metric_scores
        if return_scores:
            return output.sum(-2), metric_scores
        return output.sum(-2)

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

    # def fit(self, train, valid = None, epochs=1, batch_size=16, valid_batch_size=50, classes_subset=None):
    #     validation=[]
    #     train_history = {"loss": []}
    #     for e in range(epochs):
    #         losses = {"loss": str(0.)}
    #         average = Average()
    #         train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    #
    #         with tqdm(train_loader,
    #                   postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs)) as pbar:
    #             for i, b in enumerate(train_loader):
    #                 self.optimizer.zero_grad()
    #                 y = b["labels"].to(self.device)
    #                 y[y!=0] = 1
    #                 x = self.transform(b["text"]).to(self.device)
    #                 output, scores = self(x)
    #                 if hasattr(self, "regularize"):
    #                     l = self.loss(output, torch._cast_Float(y)) + self.regularize() + 0.3*self.loss(scores,self.label_concept_onehot[torch.where(y==1)[1]][:,None,:].repeat([1,self.max_len,1]).to(self.device))
    #                 else:
    #                     l = self.loss(output, torch._cast_Float(y)) + 0.3* self.loss(scores,self.label_concept_onehot[torch.where(y==1)[1]][:,None,:].repeat([1,self.max_len,1]).to(self.device))
    #                 l.backward()
    #                 self.optimizer.step()
    #                 average.update(l.item())
    #                 pbar.postfix[0]["loss"] = round(average.compute().item(),2*self.PRECISION_DIGITS)
    #                 pbar.update()
    #             # torch.cuda.empty_cache()
    #             if valid is not None:
    #                 validation.append(self.evaluate_classes(classes_subset=classes_subset,
    #                                                        data=valid,
    #                                                        batch_size=valid_batch_size,
    #                                                        return_report=False,
    #                                                        return_roc=False))
    #                 pbar.postfix[0].update(validation[-1])
    #                 pbar.update()
    #             # torch.cuda.empty_cache()
    #         train_history["loss"].append(average.compute().item())
    #     return{"train":train_history, "valid": validation }