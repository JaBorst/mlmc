"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
from mlmc.models.abstracts.abstracts_mo import TextClassificationAbstractMultiOutput
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..representation import is_transformer
import re
from ..layers import *

class MoLSANNC(TextClassificationAbstractMultiOutput,TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, scale="mean", share_weighting=False, weight_norm ="norm", branch_noise = 0., dropout=0.3,
                 hidden_representations= 400, representation="roberta",  d_a=200, max_len=400, n_layers=1, **kwargs):
        super(MoLSANNC, self).__init__(**kwargs)
        #My Stuff
        self.n_classes = [len(x) for x in classes]
        self.max_len = max_len
        self.representation=representation
        self.scale = scale
        self.n_layers=n_layers
        self.d_a=d_a
        self.hidden_representations = hidden_representations
        self.share_weighting = share_weighting
        self.weight_norm = weight_norm
        self.branch_noise = branch_noise
        self.dropout = dropout
        self.log_bw=False
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

        self.projection_labels = torch.nn.Linear(self.label_embedding_dim, self.hidden_representations)

        self.lsatt = NC_LabelSpecificSelfAttention(in_features=self.hidden_representations * 2,
                                                   in_features2=self.hidden_representations, hidden_features=self.d_a)
        self.latt = SplitWrapper(self.hidden_representations,
                                 NC_LabelSelfAttention(hidden_features=self.hidden_representations))

        self.dynamic_fusion = DynamicWeightedFusion(in_features=self.hidden_representations*2, n_inputs=2,
                                                    share_weights=self.share_weighting, noise=branch_noise,
                                                    norm=self.weight_norm)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.output_layer = torch.nn.ModuleList([torch.nn.Linear(self.hidden_representations * 2, 1) for _ in range(self.n_outputs)])
        self.build()

    def forward(self, x, return_weights=False):
        outputs = self.projection_input(self.embed_input(x) / self.label_embedding_dim)
        if not is_transformer(self.representation):
            outputs = outputs[0]
        outputs = self.dropout_layer(outputs)
        label_embed = self.dropout_layer(self.projection_labels(self.label_embedding))
        self_att = self.lsatt(outputs, label_embed)
        label_att = self.latt(outputs, label_embed)
        doc, weights = self.dynamic_fusion([self_att,  label_att])
        if self.log_bw:
            self.bw.append(weights.cpu())
        doc = self.dropout_layer(doc)
        pred = [l(doc / self.label_embedding_dim).squeeze() for l in self.output_layer]
        if return_weights:
            return pred, weights
        return pred
    def log_branch_weights(self, s=True):
        self.log_bw=s
    def reset_branch_weights(self):
        self.bw=[]
    def get_branch_weights(self):
        return torch.cat(self.bw).cpu()

    def create_label_dict(self):
        from ..representation import get_word_embedding_mean
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                "glove300")
        self.label_embedding_dim = l.shape[-1]
        return {w: e for w, e in zip(self.classes, l)}

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        if not hasattr(self, "label_dict"):
            try:
                self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
            except:
                self.label_dict = self.create_label_dict()
                self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        self.label_embedding = self.label_embedding.to(self.device)

