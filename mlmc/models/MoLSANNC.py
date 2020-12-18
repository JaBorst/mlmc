"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
from mlmc.models.abstracts.abstracts_mo import TextClassificationAbstractMultiOutput
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..representation import is_transformer
import re
from ..modules import *



class MoLSANNC(TextClassificationAbstractMultiOutput, TextClassificationAbstractZeroShot):

    def __init__(self, classes, scale="mean", share_weighting=False, weight_norm="norm", branch_noise=0., dropout=0.3,
                 hidden_representations=400, representation="roberta", d_a=200, max_len=400, n_layers=1, **kwargs):
        super(MoLSANNC, self).__init__(**kwargs)
        # My Stuff
        self.classes = classes
        self.n_classes = [len(x) for x in classes]
        self.n_outputs = len(classes)

        self.max_len = max_len
        self.representation = representation
        self.scale = scale
        self.n_layers = n_layers
        self.d_a = d_a
        self.hidden_representations = hidden_representations
        self.share_weighting = share_weighting
        self.weight_norm = weight_norm
        self.branch_noise = branch_noise
        self.dropout = dropout
        self.log_bw = False
        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()
        self.create_labels(classes)

        if is_transformer(self.representation):
            self.projection_input = torch.nn.Linear(self.embeddings_dim,
                                                    self.hidden_representations * 2)
        else:
            self.projection_input = torch.nn.LSTM(self.embeddings_dim,
                                                  hidden_size=self.hidden_representations,
                                                  num_layers=1,
                                                  batch_first=True,
                                                  bidirectional=True)

        # self.projection_labels = torch.nn.Linear(self.label_embedding_dim, self.hidden_representations)
        from ..modules import LSANNCModule
        self.lsannc = LSANNCModule(self.hidden_representations*2, self.label_embedding_dim,)

        self.output_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in range(self.n_outputs)])
        self.dropout_layer = torch.nn.Dropout(0.3)
        self.build()

    def forward(self, x, return_weights=False):
        outputs = self.projection_input(self.embed_input(x) / self.label_embedding_dim)
        label_embed = torch.cat([x for x in self.label_embedding],0)

        if not is_transformer(self.representation):
            outputs = outputs[0]
        outputs = self.dropout_layer(outputs)
        doc, weights = self.lsannc(outputs,label_embed,return_weights=True)
        # if self.log_bw:
        #     self.bw.append(weights.cpu())
        doc = self.dropout_layer(doc)
        n = 0
        labels = []
        for i in self.n_classes:
            labels.append(doc[:,n:(n+i)])
            n = n+i

        pred = [l(o / self.label_embedding_dim).squeeze() for o,l in zip(labels, self.output_layer)]
        if return_weights:
            return pred, weights
        return pred

    def log_branch_weights(self, s=True):
        self.log_bw = s

    def reset_branch_weights(self):
        self.bw = []

    def get_branch_weights(self):
        return torch.cat(self.bw).cpu()

    def create_label_dict(self):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        from ..representation import get_word_embedding_mean
        with torch.no_grad():
            l = [get_word_embedding_mean(
                [(" ".join(re.split("[/ _-]", re.sub("[0-9]", "", x.lower())))).strip() for x in classes.keys()],
                "glove300") for classes in self.classes]

        self.label_embedding_dim = l[0].shape[-1]
        return [{w: e for w, e in zip(classes.keys(), emb)} for classes, emb in zip(self.classes, l)]

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = [len(x) for x in classes]

        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict()
        self.label_embedding = [torch.stack([dic[cls] for cls in cls.keys()]) for cls, dic in
                                zip(self.classes, self.label_dict)]
        self.label_embedding = torch.nn.ParameterList([torch.nn.Parameter(x) for x in self.label_embedding]).to(
            self.device)
        for x in self.label_embedding: x.requires_grad = True
