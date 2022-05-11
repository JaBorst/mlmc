import torch
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract
import functools



class LabelTuning(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, dropout=0.5, *args, **kwargs):
        """
         Zeroshot model based on cosine distance of embedding vectors.
        This changes the default activation to identity function (lambda x:x)
        Args:
            mode: one of ("vanilla", "max", "mean", "max_mean", "attention", "attention_max_mean"). determines how the sequence are weighted to build the input representation
            entailment_output: the format of the entailment output if NLI pretraining is used. (experimental)
            *args:
            **kwargs:
        """
        super(LabelTuning, self).__init__(*args, **kwargs)
        self._config["dropout"] = dropout
        self.create_labels(self.classes)
        self.dropout = torch.nn.Dropout(dropout)

        self.projection = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim, bias=True)
        self.projection2 = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim, bias=True)
        # torch.nn.init.eye_(self.projection.weight)
        # torch.nn.init.eye_(self.projection2.weight)
        self.build()
        with torch.no_grad():
            self.label_start = self.embedding(**self.label_dict)[1].detach()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, embedding=False, *args, **kwargs):
        input_embedding = self.dropout(self.embedding(**x)[0])
        self.curr_label_embedding = self.dropout(self.embedding(**self.label_dict)[0])
        if self.training:
            input_embedding = input_embedding + 0.01 * torch.rand_like(input_embedding)[:, 0, None, 0, None].round() * torch.rand_like(input_embedding)  #
            # input_embedding = input_embedding * ((torch.rand_like(input_embedding[:, :, 0]) > 0.01).float() * 2 - 1)[..., None]
            input_embedding = input_embedding * ((torch.rand_like(input_embedding[:, :, 0]) > 0.01).float())[..., None]

        # self.curr_label_embedding = (0.5*self.projection2(self.projection(self.curr_label_embedding).relu()) + 0.5*self.curr_label_embedding)

        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        self.curr_label_embedding = self._mean_pooling(self.curr_label_embedding, self.label_dict["attention_mask"])

        input_embedding = input_embedding / input_embedding.norm(p=2, dim=-1, keepdim=True)
        self.curr_label_embedding = self.curr_label_embedding / self.curr_label_embedding.norm(p=2, dim=-1, keepdim=True)
        r = torch.matmul((input_embedding), (self.curr_label_embedding).t())
        if embedding:
            return r, input_embedding
        return r

    def _loss(self, x, y):
        return self.loss(x, y)

    def regularize(self):
        return 0.001*((self.projection.weight**2).sum()+(self.projection2.weight**2).sum()) #+ ((self.label_start - self.curr_label_embedding)**2).mean())

    def embed(self, x, batch_size = 64):
        data_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)
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