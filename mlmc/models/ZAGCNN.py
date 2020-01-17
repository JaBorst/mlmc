"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from mlmc.models.abstracts import TextClassificationAbstract

class ZAGCNN(TextClassificationAbstract):
    def __init__(self, weights, classes, vocabulary,  adjacency, label_embedding=None, max_len=600, dropout = 0.5, **kwargs):
        super(ZAGCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.embedding_dim = weights.shape[-1]
        self.use_dropout = dropout
        self.filters = 300
        self.kernel_sizes = [3]
        self.hidden_dim = label_embedding.shape[-1]
        self.adjacency = torch.nn.Parameter(torch.from_numpy(adjacency))
        self.adjacency.requires_grad = False

        self.embedding_untrainable = torch.nn.Embedding(weights.shape[0], self.embedding_dim)
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])

        from mlmc.layers import LabelAttention
        self.label_attention = LabelAttention(self.n_classes, self.filters, hidden_dim=self.hidden_dim,
                                              label_repr=label_embedding)
        self.dropout= torch.nn.Dropout(0.4)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=label_embedding.shape[-1], out_channels=self.hidden_dim)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)

        self.projection = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim+label_embedding.shape[-1])
        self.build()

    def forward(self, x):
        embedded = self.embedding_untrainable(x.to(self.device))
        embedded = self.dropout(embedded)
        c = torch.cat([torch.nn.functional.relu(conv(embedded.permute(0,2,1))) for conv in self.convs]).permute(0,2,1)
        c, _ = self.label_attention(c)

        labelgcn = self.gcn1(self.label_attention.label_repr, torch.stack(torch.where(self.adjacency==1),dim=0))
        labelgcn = self.dropout(labelgcn)
        labelgcn = self.gcn2(labelgcn, torch.stack(torch.where(self.adjacency==1),dim=0))

        labelvectors = torch.cat([self.label_attention.label_repr, labelgcn], dim=-1)

        return (torch.relu(self.projection(c))*labelvectors).sum(-1)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
                                                               for token in sentence.split(" ")]) for sentence in x],
                                                             batch_first=True, padding_value=0)
import numpy as np
import mlmc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-10k.vec")
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
with open("/disk1/users/jborst/Data/Test/MultiLabel/reuters/corpus-reuters-corpus-vol1/topic_codes.txt","r") as f:
    topics=[x.replace("\n","").split("\t") for x in f.readlines() if len(x) > 1][2:]
topicmap={x[0]:x[1] for x in topics}
le = np.stack([np.mean([weights[vocabulary.get(y.lower(),0)] for y in topicmap[x].split(" ") ],0) for x in data["classes"].keys() if len(x)>1])


#le = mlmc.graph.get_nmf(data["adjacency"],dim=200)
model=ZAGCNN(weights=weights,
             vocabulary=vocabulary,
             classes = data["classes"],
             label_embedding=le,
             adjacency=data["adjacency"],
             optimizer=torch.optim.Adam,
             optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
             loss=torch.nn.BCEWithLogitsLoss,
             device="cuda:0")
# model(model.transform(x=["Sentence 1 of 2","This sentence carries way more informaiton"]))
model.fit(data["train"], mlmc.data.sample(data["test"],absolute=1000), epochs=10,batch_size=50)
model.evaluate(data["test"], return_report=True)