"""
Multi-Label Zero-Shot Learning with Structured Knowledge Graphs Lee, Fang, Yeh (2018)
"""

import torch
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.layers import GatedGraphConv, LabelEmbeddingScoring
from mlmc.helpers import embed, get_embedder
import torch_geometric as torch_g

class SKG(TextClassificationAbstract):
    def __init__(self, adjacency, label_embed, classes, lm="bert",**kwargs):
        super(SKG,self).__init__(**kwargs)
        self.classes=classes
        self.n_classes = len(classes)
        self.adjacency = adjacency
        self.edge_list = torch.stack(torch.where(torch.from_numpy(adjacency) == 1), dim=0).long()-1
        self.max_len = 500

        self.embedder = get_embedder(lm)


        self.scorer = LabelEmbeddingScoring(self.n_classes,
                                            self.embedder.embedding_length,
                                            label_embed,
                                            label_freeze=True)

        self.ggc = torch_g.nn.GatedGraphConv(self.n_classes, 5)
        self.build()
    def forward(self, x):
        initial_belief = self.scorer(x).sum(-2)
        updated_belief= self.ggc(initial_belief, self.edge_list.transpose(0,1))
        return updated_belief

    def transform(self, x):
        return torch.from_numpy(embed(x, model=self.embedder, maxlen=self.max_len)).float().to(self.device)

import numpy as np
import mlmc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cpu")#"cuda:1" if torch.cuda.is_available() else "cpu")
import flair
flair.device=device
weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-10k.vec")
data = mlmc.data.get_dataset("blurbgenrecollection", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
le = mlmc.graph.get_nmf(data["adjacency"],dim=200)

skg = SKG(data["adjacency"], le, data["classes"],lm="roberta",
    optimizer=torch.optim.Adam,
             optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
             loss=torch.nn.BCEWithLogitsLoss,
             device=device)

skg.fit(data["train"], data["test"])