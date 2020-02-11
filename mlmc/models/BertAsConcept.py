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