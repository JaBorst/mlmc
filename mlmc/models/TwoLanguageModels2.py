"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
import re
from ..representation.labels import makemultilabels
from ..layers import Bilinear, AttentionWeightedAggregation
from ignite.metrics import Average
from tqdm import tqdm
from apex import amp

class LMVSLM_Classifier2(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation="roberta", label_freeze=True, max_len=300, **kwargs):
        super(LMVSLM_Classifier2, self).__init__(**kwargs)
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
        self.train_input=True

        self.classes = classes
        self.labels = self.tokenizer_label(self.classes.keys(), maxlen=10)#torch.nn.Parameter(self.embedding()[1])
        with torch.no_grad():
            self.label_embeddings=torch.nn.Parameter(self.embedding_label(self.labels)[1])
        self.label_embeddings.requires_grad = False

        self.input_projection2 = torch.nn.Linear(self.label_embedding_dim, self.embedding_dim)
        self.metric = Bilinear(self.embedding_dim)
        self.output_projection = torch.nn.Linear(in_features=self.max_len , out_features=1)

        self.att = AttentionWeightedAggregation(in_features = self.embedding_dim, d_a=self.d_a)

        self.build()


    def forward(self, x, return_scores=False):
        if self.train_input:
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
        else:
            with torch.no_grad():
                embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
        p2 = self.input_projection2(self.label_embeddings)
        label_scores = torch.matmul(embeddings,p2.t())
        output, att = self.att(embeddings, label_scores, return_att=True)
        if return_scores:
            return output, label_scores, att
        return output

    def create_labels(self, classes):
        if hasattr(self, "labels"):
            del self.labels
        self.classes = classes
        self.n_classes = len(classes)
        self.labels = self.tokenizer_label(self.classes.keys(), maxlen=10).to(self.device)
        with torch.no_grad():
            self.label_embeddings=torch.nn.Parameter(self.embedding_label(self.labels)[1])
        self.label_embeddings.requires_grad = False
        self.label_embeddings.to(self.device)

    def build(self):
        if isinstance(self.loss, type) and self.loss is not None:
            self.loss = self.loss().to(self.device)
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)
        self.to(self.device)
        self.labels = self.labels.to(self.device)

    def _init_input_representations(self):
        if not hasattr(self, "n_layers"): self.n_layers=4
        self.embedding, self.tokenizer = get(self.representation, output_hidden_states=True)
        with torch.no_grad():
            self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers

        self.embedding_label, self.tokenizer_label = get("albert", output_hidden_states=True)
        with torch.no_grad():
            self.label_embedding_dim = self.embedding_label(torch.LongTensor([[0]]))[0].shape[-1]
        for param in self.embedding_label.parameters(): param.requires_grad = False

    def fit(self, train, valid = None, epochs=1, batch_size=2, valid_batch_size=50, classes_subset=None):
        validation=[]
        train_history = {"loss": []}
        reset_labels=10
        self.labels_distance=[]
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs)) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    y = b["labels"].to(self.device)
                    y[y!=0] = 1
                    x = self.transform(b["text"]).to(self.device)
                    output = self(x)
                    if hasattr(self, "regularize"):
                        l = self.loss(output, torch._cast_Float(y)) + self.regularize()
                    else:
                        l = self.loss(output, torch._cast_Float(y))
                    with amp.scale_loss(l, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),2*self.PRECISION_DIGITS)
                    pbar.update()

                if valid is not None:
                    validation.append(self.evaluate_classes(classes_subset=classes_subset,
                                                           data=valid,
                                                           batch_size=valid_batch_size,
                                                           return_report=False,
                                                           return_roc=False))
                    pbar.postfix[0].update(validation[-1])
                    pbar.update()
                # torch.cuda.empty_cache()
            train_history["loss"].append(average.compute().item())
        return{"train":train_history, "valid": validation }

