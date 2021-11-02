import torch
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract

class EmbeddingBasedML(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, mode="vanilla", similarity="cosine", entailment_output=3, *args, **kwargs):
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
        super(EmbeddingBasedML, self).__init__(*args, **kwargs)
        self.modes = ("vanilla","mean",)
        assert mode in self.modes, f"Unknown mode: '{mode}'!"
        self.set_mode(mode=mode)
        self.set_similarity(similarity=similarity)
        self.create_labels(self.classes)
        self.parameter = torch.nn.Linear(self.embeddings_dim,256)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim,  self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear( self.embeddings_dim, 1)
        self.embedding.embeddings.requires_grad_(False)
        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def set_similarity(self, similarity):
        """Set weighting mode"""
        self._config["similarity"] = similarity

    def _sim(self, x, y):
        if self._config["similarity"] == "cosine":
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            y = y / (y.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            r = (x[:, None] * y[None]).sum(-1)
            r = torch.log(0.5 * (r +1))
        elif self._config["similarity"] == "manhattan":
            r = - (x[:,None] * y[None]).abs().sum(-1)
        elif self._config["similarity"] == "entailment":
            e = self.entailment_projection(torch.cat([
                x[:, None].repeat(1, y.shape[0], 1),
                y[None].repeat(x.shape[0], 1, 1),
                (x[:, None] - y[None]).abs()
            ], -1))
            r = self.entailment_projection2(e.relu()).squeeze(-1)
            if self._config["target"] == "entailment":
                r = r.diag()
            # elif self._config["target"] == "single":
            #     r = torch.log(e[..., -1].softmax(-1))
            # elif self._config["target"] == "multi":
            #    r = torch.log(e[..., [0, 2]].softmax(-1)[..., -1])
            # else:
            #     assert not self._config["target"], f"Target {self._config['target']} not defined"
        return r

    def forward(self, x, *args, **kwargs):
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]

        if "mean" in self.mode:
            label_embedding = label_embedding - label_embedding.mean(0)
            # print("mean added")

        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
        r = self._sim(input_embedding, label_embedding)
        return r

    def embed(self, x):
        """
        Method to return input embeddings.
        ToDo: Modularize the forward to avoid code copying.
        Args:
            x: list of input texts

        Returns: a tuple of:
            A tensor of embeddings shape (b, e), where b is the number of input texts and e the embedding dimension
            A tensor of embeddings shape (l, e), where l is the number of labels and e the embedding dimension

        """
        x = self.transform(x)
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]

        if "mean" in self.mode:
            label_embedding = label_embedding - label_embedding.mean(0)

        if "attention" in self.mode or "max" in self.mode:
            input_embedding2 = input_embedding / input_embedding.norm(p=2, dim=-1, keepdim=True)
            label_embedding2 = label_embedding / label_embedding.norm(p=2, dim=-1, keepdim=True)
            word_scores = torch.einsum("ijk,lnk->iljn", input_embedding2, label_embedding2)

        if "attention" in self.mode:
            attentions = torch.relu(word_scores.mean(-1))
            input_embedding = self._mean_pooling((attentions[..., None] * input_embedding[:, None]).transpose(1, 2),
                                                 x["attention_mask"][:, :, None])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
            input_embedding = input_embedding / (input_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            label_embedding = label_embedding / (label_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
        else:
            input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
            input_embedding = input_embedding / (input_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            label_embedding = label_embedding / (label_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
        return input_embedding, label_embedding

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
