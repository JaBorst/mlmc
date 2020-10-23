import torch
from tqdm import tqdm

from mlmc.representation import is_transformer

from .abstract_labelgraph import LabelEmbeddingAbstract

class LRDG(LabelEmbeddingAbstract):
    def __init__(self,dim, representation,trainable=True,composition="attention",embedding_activation=None, expert_activation= torch.relu,target=["label"],lrd_weight=1e-10,n_neighbours=5, sim="cos",**kwargs):
        super(LRDG, self).__init__(**kwargs)

        self.representation = representation
        self._init_input_representations()

        self.dim = dim
        self.K = 10
        self.dropout = 0.5
        self.sim=sim
        self.trainable = trainable
        self.lrd_weight=lrd_weight
        self.n_neighbours = n_neighbours
        self.composition=composition
        self.target = target if isinstance(target, list) else [target]
        self.expert_activation = getattr(torch,expert_activation) if isinstance(expert_activation, str) else expert_activation
        self.embedding_activation =  getattr(torch,embedding_activation) if isinstance(embedding_activation, str) else embedding_activation

        if self.composition=="attention":
            self.label_linear_first = torch.nn.Linear(in_features=self.embeddings_dim, out_features=200)
            self.label_linear_second = torch.nn.Linear(in_features=self.embeddings_dim, out_features=200)
            self.label_linear_third = torch.nn.Linear(in_features=self.max_len, out_features=1)


        self.training_projection = torch.nn.Linear(in_features=self.dim*2, out_features=self.embeddings_dim)
        self.training_projection2 = torch.nn.Linear(in_features=self.embeddings_dim, out_features=1)

        from mlmc.layers import ExpertCoRep
        self.label_ECR = ExpertCoRep(in_features=self.embeddings_dim,
                                     out_features=self.dim,
                                     centers=torch.rand(self.K, self.embeddings_dim),
                                     trainable=trainable,
                                     activation=self.expert_activation)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.build()

    def forward(self, x):
        if is_transformer(self.representation):
            if self.finetune:
                embedded = self.embedding(x)[0]
            else:
                with torch.no_grad():
                    embedded = self.embedding(x)[0]
        else:
            if self.finetune:
                embedded = self.embedding(x)
            else:
                with torch.no_grad():
                    embedded = self.embedding(x)

        label = self._compose(embedded)
        label = self.embedding_activation(self.label_ECR(label)) if self.embedding_activation is not None else self.label_ECR(label)
        return label

    def _compose(self, x):
        if self.composition=="attention":
            l1 = self.label_linear_second(x)
            l2 = torch.tanh(self.label_linear_first(x)).transpose(1, 2)
            labelselfatt = torch.softmax(self.label_linear_third(torch.bmm(l1, l2)), -2)
            return (x * labelselfatt).sum(1)
        if self.composition == "sum":
            return x.sum(-2)
        if self.composition == "mean":
            return x.mean(-2)

    def _lrd(self,x,y):
        v_closest_seen_to_unseen = torch.sqrt(((x[:,None]-y)**2/(y.std(0)**2)).sum(-1)).topk(1,dim=-1,sorted=True, largest=False)
        v_closest_seen_to_seen = torch.sqrt(((y[:,None] - y[:,:, None]) ** 2).sum(-1)).topk(2,sorted=True, largest=False)[0][:,:,1]
        r = v_closest_seen_to_unseen[0] / (1+torch.cat([k[v] for k, v in zip(v_closest_seen_to_seen, v_closest_seen_to_unseen[1].long())])[:,None])
        return r

    def _lrd_min_mean(self, x, y):
        return torch.nn.functional.mse_loss(x[:,None].repeat(1, y.shape[1],1), y)#(r * mask[None]).mean(-1)

    def _sim(self, x, y):
        if self.sim == "cosine":
            return ((x[:,None]*y).sum(-1) / x.norm(p=2,dim=-1, keepdim=True) / y.norm(p=2,dim=-1, keepdim=False))
        if self.sim == "euclidean":
            return 1-((y-x[:,None])**2).sum(-1)
        if self.sim == "classify":
            r = torch.cat([y, x[:, None].repeat(1, y.shape[1], 1)],-1)
            return self.training_projection2(torch.relu(self.training_projection(r))).squeeze()

    def _lrd_loss(self,x,y):
        # x.shape
        return ((self._lrd_min_mean(x,y)-1)**2)

    def _relation(self, x):
        return self.training_projection2(torch.relu(self.training_projection(x))).squeeze()
        # return self.training_projection(x).squeeze()

    def _loss(self, x, y, z):
        e1 = self(self.transform(x).to(self.device))
        e2 = self.dropout_layer(torch.stack([self(self.transform(w).to(self.device)) for w in y]).transpose(0,1))
        e3 = self.dropout_layer(torch.stack([self(self.transform(w).to(self.device)) for w in z]).transpose(0,1))

        pos = self._sim(e1, e2)
        neg = self._sim(e1, e3)

        result = torch.cat([pos, neg],-1)
        target = torch.cat([torch.ones_like(pos),torch.zeros_like(neg)],-1)

        return self.loss(result,target) +self.lrd_weight*self._lrd_loss(e1,e2)

    def fit(self, graph, valid=None, epochs=1, batch_size=16,_run=None):
        from ..data import GraphDataset
        gd = GraphDataset(graph, n=self.n_neighbours, target=self.target)
        vd = GraphDataset(valid, n=self.n_neighbours, target=self.target)

        from ignite.metrics import  Average
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(gd, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs), ncols=100) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    l = self._loss(b["input"],b["neighbours"],b["negatives"])
                    l.backward()

                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),2*self.PRECISION_DIGITS)
                    pbar.update()
                if _run is not None:
                    _run.log_scalar("loss", average.compute().item(), e)
                if valid is not None:
                    with torch.no_grad():
                        valid_loader = torch.utils.data.DataLoader(vd, batch_size=batch_size, shuffle=False)
                        valid_average = Average()
                        for b in valid_loader:
                            valid_average.update(self._loss(b["input"], b["neighbours"], b["negatives"]))
                        pbar.postfix[0]["valid_loss"]= round(valid_average.compute().item(),2*self.PRECISION_DIGITS)
                        pbar.update()
                    if _run is not None:
                        _run.log_scalar("valid_loss", valid_average.compute().item(), e)

    def embed(self, x):
        return self(self.transform(x))

    def cosine(self, x, y):
        x = self(self.transform(x))
        y = self(self.transform(y))
        return ((x/x.norm(p=2, dim=-1 , keepdim=True)) * (y/y.norm(p=2, dim=-1 , keepdim=True)) ).sum(-1)
    def eclidean(self, x, y):
        x = self(self.transform(x))
        y = self(self.transform(y))
        return torch.sqrt(((x  - y)**2 ).sum(-1))


