"""
Multi-Label Zero-Shot Learning with Structured Knowledge Graphs Lee, Fang, Yeh (2018)
"""

import torch
from ..models.abstracts import TextClassificationAbstract
from ..layers import GatedGraphConv, LabelEmbeddingScoring
from ..representation.representations import get
import torch_geometric as torch_g


class SKG(TextClassificationAbstract):
    def __init__(self, adjacency, label_embed, classes, static=None, transformer="bert", max_len=300, **kwargs):
        super(SKG,self).__init__(**kwargs)
        self.classes=classes
        self.n_classes = len(classes)
        self.adjacency = torch.nn.Parameter(torch.from_numpy(adjacency).long(),requires_grad=False)
        self.edge_list = torch.nn.Parameter(torch.stack(torch.where(self.adjacency == 1), dim=0).long(),requires_grad=False)
        self.max_len = max_len
        self.static=static
        self.transformer=transformer

        self.embedder, self.tokenizer = get(static, transformer, output_hidden_states=True)
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
