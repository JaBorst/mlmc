import torch
from ...abstracts.abstract_embedding import LabelEmbeddingAbstract

class Siamese(LabelEmbeddingAbstract):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, dropout=0.5, vertical_dropout=0, word_noise=0, score ="cosine", *args, **kwargs):
        """
         Zeroshot model based on cosine distance of embedding vectors.
        This changes the default activation to identity function (lambda x:x)
        Args:
            mode: one of ("vanilla", "max", "mean", "max_mean", "attention", "attention_max_mean"). determines how the sequence are weighted to build the input representation
            entailment_output: the format of the entailment output if NLI pretraining is used. (experimental)
            *args:
            **kwargs:
        """
        super(Siamese, self).__init__(*args, **kwargs)
        self._config["dropout"] = dropout
        self._config["vertical_dropout"] = vertical_dropout
        self._config["word_noise"] = word_noise
        if  self._config["target"] == "entailment" and score != "entailment":
            print("Setting scoring to entailment automatically")
            self.entailment()
            self._config["score"] = "entailment"
        else:
            self._config["score"] = score
        self.create_labels(self.classes)
        self.dropout = torch.nn.Dropout(dropout)


        if self._config["score"] == "entailment":
            self.entailment_projection = torch.nn.Linear(self.embeddings_dim*3, self.embeddings_dim)
            self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 3)

        self.build()

    def forward(self, x, embedding=False, *args, **kwargs):
        input_embedding = self.embed_input(x[0])
        label_embedding = self.dropout(self.embedding(**x[1])[0])
        input_embedding = self._mean_pooling(input_embedding, x[0]["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, x[1]["attention_mask"])

        r = self._score(input_embedding, label_embedding)
        if embedding:
            return r, input_embedding
        return r

    def _score(self, x, y):
        if self._config["score"] == "scalar":
            return self._scalar_score(x,y)
        if self._config["score"] == "cosine":
            return self._cosine_score(x,y)
        if self._config["score"] == "entailment":
            return self._entailment_score(x, y)

    def _cosine_score(self, x, y):
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        y = y / y.norm(p=2, dim=-1, keepdim=True)
        if self._config["target"] in ["entailment"]:
            return torch.matmul(x, y.t()).log_softmax(-1)
        elif self._config["target"] in ["abc"]:
            y = y.reshape((x.shape[0], len(self.classes), x.shape[-1]))
            return (x[:,None] * y).sum(-1)
        else:
            return (torch.matmul(x,y.t())).log_softmax(-1)

    def _scalar_score(self, x, y):
        if self._config["target"] in ["entailment"]:
            return torch.matmul(x, y.t()).log_softmax(-1)
        if self._config["target"] in ["abc"]:
            y = y.reshape((x.shape[0], len(self.classes), x.shape[-1]))
            return (x[:,None] * y).sum(-1).log_softmax(-1)
        else:
            return (torch.matmul(x,y.t())).log_softmax(-1)

    def _entailment_score(self, x, y):
        if self._config["target"] == "entailment":
            e = torch.cat([x, y, (x - y).abs()], -1)
            r = self.entailment_projection2(self.entailment_projection(e).relu())
        if self._config["target"] == "abc":
            y = y.reshape((x.shape[0], len(self.classes), x.shape[-1]))
            e = torch.cat([x[:, None].repeat(1, y.shape[1], 1), y, (x[:, None] - y).abs()], -1)
            r = self.entailment_projection2(self.entailment_projection(e).relu())[..., -1]
        elif self._config["target"] == "single":
            e = torch.cat([
                x[:, None].repeat(1, y.shape[0], 1),
                y[None].repeat(x.shape[0], 1, 1),
                (x[:, None] - y[None]).abs()
            ], -1)
            r = self.entailment_projection2(self.entailment_projection(e).relu())
            r = r[..., -1]
        elif self._config["target"] == "multi":
            e = torch.cat([
                x[:, None].repeat(1, y.shape[0], 1),
                y[None].repeat(x.shape[0], 1, 1),
                (x[:, None] - y[None]).abs()
            ], -1)
            r = self.entailment_projection2(self.entailment_projection(e).relu())
            r = r[:, [0, 2]].log_softmax(-1)[..., -1]
        else:
            assert not self._config["target"], f"Target {self._config['target']} not defined"
        return r

    def _loss(self, x, y):
        return self.loss(x, y)

    def embed(self, x, batch_size = 64):
        data_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
        from tqdm import tqdm
        scores = []
        embeddings = []
        with torch.no_grad():
            with tqdm(data_loader, desc="Embedding", ncols=100) as pbar:
                for i, b in enumerate(data_loader):
                    score, emb = (self.forward(x = self.transform(b["text"]), embedding=True))
                    scores.append(score.cpu())
                    embeddings.append(emb.cpu())
                    pbar.update()

        return torch.cat(scores), torch.cat(embeddings)
