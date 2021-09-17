import torch
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FNetTransformer(torch.nn.Module):
    def __init__(self, dim, dropout=0.5,*args, **kwargs):
        super(FNetTransformer, self).__init__()
        self.embedding_dim = dim
        self.normalize = False
        self.norm = torch.nn.LayerNorm(self.embedding_dim)
        self.ff = FeedForward(self.embedding_dim,self.embedding_dim, dropout=dropout)
    def forward(self, x, mask=None):
        t = torch.stack([x, torch.zeros_like(x)],-1)
        ft = torch.fft(torch.fft(t, signal_ndim=1, normalized=self.normalize),signal_ndim=2,normalized=self.normalize)[...,0]
        t = self.norm(x + ft)
        t = self.ff(t)
        if mask is not None:
            added = t * mask
        return added

class FNetEncoder(torch.nn.Module):
    def __init__(self, dim, num_layers):
        super(FNetEncoder, self).__init__()
        self.encoder = torch.nn.ModuleList([FNetTransformer(dim) for _ in range(num_layers)])

    def forward(self, x, **kwargs):
        for l in self.encoder:
            x = l(x, **kwargs)
        return x

class EmbeddingBasedEntailmentFFT(SentenceTextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, mode="vanilla",  *args, **kwargs):
        """
         Zeroshot model based on cosine distance of embedding vectors.
        This changes the default activation to identity function (lambda x:x)
        Args:
            mode: one of ("vanilla", "max", "mean", "max_mean", "attention", "attention_max_mean"). determines how the sequence are weighted to build the input representation
            entailment_output: the format of the entailment output if NLI pretraining is used. (experimental)
            *args:
            **kwargs:
        """
        if "act" not in kwargs:
            kwargs["activation"] = lambda x: x
        super(EmbeddingBasedEntailmentFFT, self).__init__(*args, **kwargs)
        self.modes = ("vanilla","max","mean","max_mean", "attention","attention_max_mean")
        assert mode in self.modes, f"Unknown mode: '{mode}'!"
        self.set_mode(mode=mode)

        self.create_labels(self.classes)
        self.bottle_neck = 1024

        self.parameter = torch.nn.Linear(self.embeddings_dim,self.bottle_neck)
        self.parameter2 = torch.nn.Linear(self.bottle_neck,self.embeddings_dim)

        self.entailment_projection = torch.nn.Linear(5*self.embeddings_dim, self.bottle_neck)
        self.entailment_projection2 = torch.nn.Linear(self.bottle_neck, 3)
        self.hypothesis = torch.nn.Parameter(torch.tensor(torch.rand((1,self.bottle_neck))))

        self.encoder=FNetEncoder(self.bottle_neck,num_layers=8)

        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, *args, **kwargs):
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]

        le_proj = self.parameter(label_embedding)
        ie_proj = self.parameter(input_embedding)

        if self._config["target"] == "entailment":
            comb = torch.cat([le_proj, ie_proj],1)
            comb_mask = torch.cat([self.label_dict["attention_mask"], x["attention_mask"]],1)
            comb = self.encoder(comb, mask=comb_mask[...,None])
            comb_pool = (comb*comb_mask[...,None]).sum(1)/comb_mask.sum(1,keepdims=True)

            comb_pool = self.parameter2(comb_pool)
            input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
            e = torch.cat([input_embedding,
                           label_embedding,
                           torch.abs(input_embedding - label_embedding),
                           input_embedding + label_embedding,
                comb_pool],-1)
            # e=comb_pool
        else:
            comb = torch.cat([le_proj[None].repeat(input_embedding.shape[0],1,1,1),
                              ie_proj[:,None].repeat(1,label_embedding.shape[0],1,1)], 2)

            comb_mask = torch.cat([self.label_dict["attention_mask"][None].repeat(input_embedding.shape[0],1,1),
                                   x["attention_mask"][:,None].repeat(1,label_embedding.shape[0],1)], 2)
            b,l,m,e = comb.shape
            comb = self.encoder(comb.reshape((b*l,m,e)), mask=comb_mask.reshape((b*l,m))[...,None])
            comb = comb.reshape((b, l, m, e))
            comb_pool = (comb*comb_mask[...,None]).sum(2)/comb_mask.sum(2,keepdims=True)

            comb_pool = self.parameter2(comb_pool)

            input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])

            e = torch.cat([input_embedding[:,None].repeat((1,label_embedding.shape[0],1)),
                           label_embedding[None].repeat((input_embedding.shape[0],1,1)),
                            torch.abs(input_embedding[:, None] - label_embedding[None]),
                            input_embedding[:, None] + label_embedding[None],
                           comb_pool],-1)
            # e = comb_pool

        logits = self.entailment_projection2(torch.nn.GELU()(self.entailment_projection(e)))

        if self._config["target"] == "entailment":
            pass
        elif self._config["target"] == "single":
            logits = torch.log(logits[...,-1].softmax(-1))
        elif self._config["target"] == "multi":
            logits = torch.log(e[..., [0, 2]].softmax(-1)[..., -1])
        else:
            assert not self._config["target"], f"Target {self._config['target']} not defined"
        return logits

    def scores(self, x):
        """
        Returns 2D tensor with length of x and number of labels as shape: (N, L)
        Args:
            x:

        Returns:

        """
        self.eval()
        assert not (self._config["target"] == "single" and   self._config["threshold"] != "max"), \
            "You are running single target mode and predicting not in max mode."

        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
            if self._loss_name == "ranking":
                output = 0.5*(output+1)
        self.train()
        return output

    def single(self, loss="ranking"):
        """Helper function to set the model into single label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "single"
        self.set_threshold("max")
        self.set_activation(lambda x: x)
        self._loss_name = loss
        if loss == "ranking":
            self.set_loss(RelativeRankingLoss(0.5))
        else:
            self.set_loss(torch.nn.CrossEntropyLoss())
        self._all_compare=True

    def multi(self, loss="ranking"):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)
        self._loss_name = loss
        if loss == "ranking":
            self.set_loss(RelativeRankingLoss(0.5))
        else:
            self.set_loss(torch.nn.BCELoss)
        self._all_compare=True

    def sts(self):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self._loss_name="ranking"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)
        self.set_loss(RelativeRankingLoss(0.5))

    def entailment(self):
        self._config["target"] = "entailment"
        self.target = "entailment"
        self.set_sformatter(lambda x: x)
        self.set_threshold("max")
        self.set_activation(torch.softmax)
        self.set_loss = torch.nn.CrossEntropyLoss()
        self._all_compare = False
