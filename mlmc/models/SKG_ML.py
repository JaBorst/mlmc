"""
Multi-Label Zero-Shot Learning with Structured Knowledge Graphs Lee, Fang, Yeh (2018)
"""

import torch
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.layers import GatedGraphConv, LabelEmbeddingScoring
import torch_geometric as torch_g


class SKG(TextClassificationAbstract):
    def __init__(self, adjacency, label_embed, classes, static=None, transformer="bert", **kwargs):
        super(SKG,self).__init__(**kwargs)
        self.classes=classes
        self.n_classes = len(classes)
        self.adjacency = torch.nn.Parameter(torch.from_numpy(adjacency).long(),requires_grad=False)
        self.edge_list = torch.nn.Parameter(torch.stack(torch.where(self.adjacency == 1), dim=0).long(),requires_grad=False)
        self.max_len = 300

        self.embedder, self.tokenizer= mlmc.embeddings.get(static, transformer, output_hidden_states=True)
        self.embedder.eval()
        self.embedding_dim = torch.cat(self.embedder(self.embedder.dummy_inputs["input_ids"])[2][-5:-1],-1).shape[-1]
        self.scorer = LabelEmbeddingScoring(self.n_classes,
                                            self.embedding_dim,
                                            label_embed,
                                            label_freeze=True)

        self.ggc = torch_g.nn.GatedGraphConv(self.n_classes, 10)
        self.build()

    def forward(self, x):
        with torch.no_grad():
            e = torch.cat(self.embedder(x)[2][-5:-1],-1)
        initial_belief = self.scorer(e).sum(-2)
        updated_belief = self.ggc(initial_belief, edge_index=self.adjacency)
        return updated_belief


import mlmc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
def clean(x):
    import string
    return "".join([c for c in x if c in string.ascii_letters + string.punctuation + " "])
data["train"].transform(clean)
# data["test"].transform(clean)

le = mlmc.graph.get_nmf(data["adjacency"], dim=200)

skg = SKG(data["adjacency"], le, data["classes"], transformer="bert",  #weights=weights, vocabulary=vocabulary,
          optimizer=torch.optim.Adam,
          optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
          loss=torch.nn.BCEWithLogitsLoss,
          device=device)
skg.fit(data["train"], mlmc.data.sample(data["test"],absolute=5000),64, batch_size=16)
