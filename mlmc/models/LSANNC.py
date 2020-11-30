"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
from mlmc.models.abstracts.abstracts import TextClassificationAbstract
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..representation import is_transformer
import re
from ..modules import *

class LSANNC(TextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, scale="mean", share_weighting=False, weight_norm ="norm", branch_noise = 0., dropout=0.3,
                 hidden_representations= 400, representation="google/bert_uncased_L-2_H-768_A-12" ,  d_a=200, max_len=400, n_layers=1, **kwargs):
        super(LSANNC, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
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

        # self.projection_labels = torch.nn.Linear(self.label_embedding_dim, self.hidden_representations)

        from ..modules import LSANNCModule
        self.lsannc = LSANNCModule(self.hidden_representations*2, self.label_embedding_dim )
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.build()

    def forward(self, x, return_weights=False):
        outputs = self.projection_input(self.embed_input(x) / self.embeddings_dim)
        if not is_transformer(self.representation):
            outputs = outputs[0]
        # outputs = self.dropout_layer(outputs)
        # label_embed = self.dropout_layer(self.label_embedding)
        doc, weights = self.lsannc(outputs, self.label_embedding, return_weights=True)

        pred = self.output_layer(doc).squeeze(-1)
        if return_weights:
            return pred, weights
        return pred
    def log_branch_weights(self, s=True):
        self.log_bw=s
    def reset_branch_weights(self):
        self.bw=[]
    def get_branch_weights(self):
        return torch.cat(self.bw).cpu()

    def label_embed(self, classes):
        from ..representation import get_word_embedding_mean
        import re
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                "glove300")
        return l


    # def create_label_dict(self):
    #     from ..representation import get_word_embedding_mean
    #     with torch.no_grad():
    #         l = get_word_embedding_mean(
    #             [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
    #             "glove300")
    #     self.label_embedding_dim = l.shape[-1]
    #     return {w: e for w, e in zip(self.classes, l)}
    #
    # def create_labels(self, classes):
    #     self.classes = classes
    #     self.n_classes = len(classes)
    #     if not hasattr(self, "label_dict"):
    #         self.label_dict = self.create_label_dict()
    #     self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
    #     self.label_embedding = self.label_embedding.to(self.device)
    #
