import torch

import mlmc.modules
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract

class Caps(torch.nn.Module):
    def __init__(self, dim =128, n = 128):
        super(Caps, self).__init__()
        self.linear = torch.nn.Linear(dim,dim)
        self.linear2 = torch.nn.Linear(dim,dim)

        self.class1 = torch.nn.Linear(n,n)
        self.class2 = torch.nn.Linear(n,n)

        self.act = torch.nn.GELU()
    def forward(self, x, y, m, mask_x, mask_y):
        # x_ = mlmc.modules.squash(self.linear(x))
        # y_ = mlmc.modules.squash(self.linear(y))
        # m_ = mlmc.modules.squash(self.linear(m))
        x_ = self.linear(x)*mask_x[...,None]
        y_ = self.linear(y)*mask_y[...,None]
        m_ = self.linear(m)

        pij = torch.sigmoid(self.class2(self.act(self.class1(torch.matmul(x_, m_.t())))))
        qij = torch.sigmoid(self.class2(self.act(self.class1(torch.matmul(y_, m_.t())))))

        pij = mlmc.modules.norm(pij)
        qij = mlmc.modules.norm(qij)

        x__ = (x + self.linear2(torch.matmul(pij,m))*mask_x[...,None])/2
        y__ = (y + self.linear2(torch.matmul(qij,m))*mask_y[...,None])/2

        return x__, y__


class CapsMemory(torch.nn.Module):
    def __init__(self, l=16, m=128, d=768, dim=128):
        super(CapsMemory, self).__init__()
        self.d = d
        self.dim = dim
        self.l = l
        self.m = m
        self.input_projection = torch.nn.Linear(self.d,self.dim)
        self.input_projection2 = torch.nn.Linear(self.d,self.dim)
        self.memory = torch.nn.Parameter(torch.rand((self.m,self.dim)))
        self.cap = torch.nn.ModuleList([Caps(dim = self.dim) for _ in range(self.l)])

    def forward(self,x, y, mask_x, mask_y):
        x_ = x + self.input_projection(x)*mask_x[...,None]
        y_ = y + self.input_projection2(y)*mask_y[...,None]
        # for cap in self.cap:
        #     x_, y_ = cap(x_, y_, self.memory,mask_x, mask_y)
        return x_, y_


class EmbeddingBasedEntailmentCPS(SentenceTextClassificationAbstract,TextClassificationAbstractZeroShot):
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
        super(EmbeddingBasedEntailmentCPS, self).__init__(*args, **kwargs)
        self.modes = ("vanilla","max","mean","max_mean", "attention","attention_max_mean")
        assert mode in self.modes, f"Unknown mode: '{mode}'!"
        self.set_mode(mode=mode)

        self.create_labels(self.classes)
        self.bottle_neck = 768

        self.parameter = torch.nn.Linear(self.embeddings_dim, self.bottle_neck)
        self.parameter2 = torch.nn.Linear(self.bottle_neck, self.embeddings_dim)

        self.entailment_projection = torch.nn.Linear(3*self.bottle_neck, self.bottle_neck)
        self.entailment_projection2 = torch.nn.Linear(self.bottle_neck, 3)

        self.encoder = CapsMemory(dim=self.bottle_neck)

        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, *args, **kwargs):
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]


        # input_embedding,label_embedding = self.encoder(input_embedding, label_embedding, x["attention_mask"],  self.label_dict["attention_mask"])

        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])

        if self._config["target"] == "entailment":
            e = torch.cat([input_embedding,label_embedding, torch.abs(input_embedding-label_embedding)],-1)
        else:
            e = torch.cat([
                input_embedding[:,None].repeat((1,label_embedding.shape[0],1)),
                label_embedding[None].repeat((input_embedding.shape[0],1,1)),
                torch.abs(input_embedding[:, None]-label_embedding[None])
            ],-1)


        logits = self.entailment_projection2(torch.relu(self.entailment_projection(e)))

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
