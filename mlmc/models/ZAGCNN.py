"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.representations import get

class ZAGCNN(TextClassificationAbstract):
    def __init__(self, classes,   adjacency, label_embedding=None, static=None, transformer=None, max_len=600, dropout = 0.5, **kwargs):
        super(ZAGCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 300
        self.kernel_sizes = [3,5,10]
        self.hidden_dim = label_embedding.shape[-1]
        self.adjacency = torch.nn.Parameter(torch.from_numpy(adjacency))
        self.adjacency.requires_grad = False


        self.embedding, self.tokenizer = get(static, transformer)
        self.embedding_dim = self.embedding.weight.shape[-1]



        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        from mlmc.layers import LabelAttention
        self.label_attention = LabelAttention(self.n_classes, self.filters, hidden_dim=self.hidden_dim,
                                              label_repr=label_embedding, freeze=True)
        self.dropout= torch.nn.Dropout(0.4)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=label_embedding.shape[-1], out_channels=self.hidden_dim)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        torch.nn.MaxPool1d(3, stride=3)
        self.projection = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim+label_embedding.shape[-1])
        self.build()

    def forward(self, x):
        embedded = self.embedding(x.to(self.device))
        embedded = self.dropout(embedded)
        c = torch.cat([self.pool(torch.nn.functional.relu(conv(embedded.permute(0,2,1)))) for conv in self.convs], dim=-1).permute(0,2,1)
        c, _ = self.label_attention(c)
        c = self.dropout(c)
        labelgcn = self.gcn1(self.label_attention.label_repr, torch.stack(torch.where(self.adjacency==1),dim=0))
        labelgcn = self.dropout(labelgcn)
        labelgcn = self.gcn2(labelgcn, torch.stack(torch.where(self.adjacency==1),dim=0))
        labelvectors = torch.cat([self.label_attention.label_repr, labelgcn], dim=-1)
        return (torch.relu(self.projection(c))*labelvectors).sum(-1)

    def transform(self, x):
        return self.tokenizer(x,self.max_len)

import mlmc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)

le = mlmc.graph.get_nmf(data["adjacency"],dim=50) + 1e-100
model=ZAGCNN(
    static="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-10k.vec",
    classes = data["classes"],
    label_embedding=le,
    adjacency=data["adjacency"],
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
    loss=torch.nn.BCEWithLogitsLoss,
    device="cuda:0")
model.fit(data["train"], mlmc.data.sample(data["test"],absolute=1000), epochs=50,batch_size=50)
model.evaluate(data["test"], return_report=True)