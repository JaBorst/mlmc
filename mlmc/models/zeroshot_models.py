import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import  uniform


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            h = self.rnn(m, h)
        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return torch.tanh(edge_weight.view(-1, 1) * x_j)
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)


# n_classes = 5
# import numpy as np
# i = np.random.rand(16,n_classes)
# adj = np.random.randint(2,size=(n_classes,n_classes))
# adj=np.identity(n_classes)
# ggc = GatedGraphConv(num_layers=2, out_channels=n_classes)
# o = ggc(torch.from_numpy(i).float(),torch.from_numpy(adj).long())

from tqdm import tqdm
import torch
import ignite
from sklearn import metrics as skmetrics
from ..metrics.multilabel import MultiLabelReport



class ZeroShotTextClassificationAbstract(torch.nn.Module):
    def __init__(self, loss, optimizer, optimizer_params = {"lr": 1.0}, device="cpu", **kwargs):
        super(ZeroShotTextClassificationAbstract,self).__init__(**kwargs)

        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    def build(self):
        self.to(self.device)
        self.loss = self.loss().to(self.device)
        self.optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

    def evaluate(self, data, return_report=False):
        """
        Evaluation, return accuracy and loss
        """
        self.eval()  # set mode to evaluation to disable dropout
        p_1 = ignite.metrics.Precision(is_multilabel=True,average=True)
        p_3 = ignite.metrics.Precision(is_multilabel=True,average=True)
        p_5 = ignite.metrics.Precision(is_multilabel=True,average=True)
        subset_65 = ignite.metrics.Accuracy(is_multilabel=True)
        subset_mcut = ignite.metrics.Accuracy(is_multilabel=True)
        report = MultiLabelReport(self.classes)
        average = ignite.metrics.Average()
        data_loader = torch.utils.data.DataLoader(data, batch_size=50)
        for i, b in enumerate(data_loader):
            y = b["labels"]
            y[y!=0] = 1
            x = self.transform(b["text"])
            output = self(x.to(self.device)).cpu()
            l = self.loss(output, torch._cast_Float(y))

            output = torch.sigmoid(output)
            average.update(l.item())
            # accuracy.update((prediction, y))
            p_1.update((torch.zeros_like(output).scatter(1,torch.topk(output, k=1)[1],1), y))
            p_3.update((torch.zeros_like(output).scatter(1,torch.topk(output, k=3)[1],1), y))
            p_5.update((torch.zeros_like(output).scatter(1,torch.topk(output, k=5)[1],1), y))
            subset_65.update((self.threshold(output,tr=0.65,method="hard"), y))
            subset_mcut.update((self.threshold(output,tr=0.65,method="mcut"), y))
            report.update((self.threshold(output,tr=0.65,method="mcut"), y))
            # auc_roc.update((torch.sigmoid(output),y))
        self.train()
        return {
            # "accuracy": accuracy.compute(),
            "valid_loss": round(average.compute().item(),6),
            "p@1": round(p_1.compute(),4),
            "p@3": round(p_3.compute(),4),
            "p@5": round(p_5.compute(),4),
            "a@0.65": round(subset_65.compute(),4),
            "a@mcut": round(subset_mcut.compute(),4),
            "report": report.compute() if return_report else None
            # "auc": round(auc_roc.compute(),4),
        }

    def fit(self, train, valid = None, epochs=1, batch_size=16):
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = ignite.metrics.Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs)) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    y = b["labels"].to(self.device)
                    x = self.transform(b["text"]).to(self.device)
                    output = self(x)
                    l = self.loss(output, y)
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),6)
                    l.backward()
                    self.optimizer.step()
                    pbar.update()
                if valid is not None:

                    pbar.postfix[0].update(self.evaluate(valid))
                    pbar.update()

    def threshold(self, x, tr=0.5, method="hard"):
        if method=="hard":
            return (torch.sigmoid(x)>tr).int()
        if method=="mcut":
            x_sorted = torch.sort(x,-1)[0]
            thresholds = (x_sorted[:,1:] - x_sorted[:,:-1]).max(-1)[0]
            return (x > thresholds[:, None]).float()
from mlmc.layers.label_layers import LabelEmbeddingScoring

class LabelScoringGraphModel(ZeroShotTextClassificationAbstract):
    """
    Reimplementation of https://github.com/EMNLP2019LSAN/LSAN
    """
    def __init__(self, classes, weights, vocabulary, label_embedding=None, adjacency=None, max_len=600,dropout = 0.5, **kwargs):
        super(LabelScoringGraphModel, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.embedding_dim = weights.shape[-1]
        self.lstm_units = self.embedding_dim
        self.use_dropout = dropout
        self.adjacency = torch.from_numpy(adjacency).long().to(self.device)
        #self.adjacency.requires_grad = False

        self.embedding_untrainable = torch.nn.Embedding(weights.shape[0], self.embedding_dim)
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)

        self.lstm = torch.nn.LSTM(self.embedding_dim,
                                  self.lstm_units,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)

        self.label_similarity = LabelEmbeddingScoring(self.n_classes,
                                                      2*self.lstm_units,
                                                      label_repr=label_embedding,
                                                      similarity="cosine",
                                                      label_freeze=True)
        self.ggc = GatedGraphConv(num_layers=2, out_channels=self.n_classes)
        self.projection = torch.nn.Linear(in_features=self.n_classes, out_features= self.n_classes)
        if self.use_dropout > 0.0: self.dropout = torch.nn.Dropout(self.use_dropout)
        self.build()

    def forward(self, x):
        embedded_1 = self.embedding_untrainable(x)
        if self.use_dropout > 0.0: embedded_1 = self.dropout(embedded_1)
        c,_ = self.lstm(embedded_1)
        if self.use_dropout > 0.0: c = self.dropout(c)
        ls = self.label_similarity(c).sum(1)
        ls = self.ggc(ls, self.adjacency)
        return self.projection(ls)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0)