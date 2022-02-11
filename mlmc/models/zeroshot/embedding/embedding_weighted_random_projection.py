import torch
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract
from ....modules.dropout import VerticalDropout


class NonLinearRandomProjectionAttention(torch.nn.Module):
    def __init__(self, in_features, inner_dim=32, n_proj=24, trainable=False):
        super(NonLinearRandomProjectionAttention, self).__init__()
        self.in_features=in_features
        self.inner_dim = inner_dim
        self.n_proj = n_proj
        self.set_trainable(trainable)
        self.random_projections = torch.nn.ParameterList([torch.nn.Parameter(self._random_gaussorthonormal(), requires_grad=trainable) for _ in range(self.n_proj)])

    def set_trainable(self, trainable= False):
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def _random_projection_matrix(self):
        ##ä### maybe do this sparse?
        random = torch.rand(128, 32)  # .round().float()
        z = torch.zeros_like(random)
        z[random < 0.3] = -1
        z[random > 0.66] = 1
        z = z / z.norm(p = 2, dim=-1, keepdim=True)
        return z


    def _random_gaussorthonormal(self):
        z = torch.nn.init.orthogonal_(torch.rand(self.in_features,self.inner_dim))
        z = z / z.norm(p = 2, dim=0, keepdim=True)
        return z

    def _projection(self, x):
        return  torch.stack([torch.mm(x, z) for z in self.random_projections]).mean(0)

    def forward(self, x, y, v=None):
        xp = self._projection(x)
        yp = self._projection(y)
        attention = torch.mm(xp, yp.t())
        if v is not None:
            return  attention, torch.mm(attention.softmax(-1), v)
        else:
            return attention

class EmbeddingRandom(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, mode="vanilla", inner_dim=32, n_proj=10, dropout=0.5,trainable=False, vertical_dropout=0., *args, **kwargs):
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
        super(EmbeddingRandom, self).__init__(*args, **kwargs)
        self.modes = ("vanilla","max","mean","max_mean", "attention","attention_max_mean")
        assert mode in self.modes, f"Unknown mode: '{mode}'!"
        self.set_mode(mode=mode)
        self._config["inner_dim"] = inner_dim
        self._config["n_proj"] = n_proj
        self._config["trainable"] = trainable

        self.rp = NonLinearRandomProjectionAttention(self.embeddings_dim, inner_dim=self._config["inner_dim"], n_proj=self._config["n_proj"], trainable=trainable)

        self.create_labels(self.classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.vdropout = VerticalDropout(vertical_dropout)

        self._config["dropout"] = dropout
        self._config["vertical_dropout"]= vertical_dropout
        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, emb=False, *args, **kwargs):
        input_embedding = self.dropout(self.embedding(**x)[0])
        label_embedding = self.dropout(self.embedding(**self.label_dict)[0])

        input_embedding = self.vdropout(input_embedding)
        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])

        r = self.rp(input_embedding, label_embedding)
        return r



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
            #output = 0.5*(output+1)
        self.train()
        return output
