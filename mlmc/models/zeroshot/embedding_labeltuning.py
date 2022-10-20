import torch
from mlmc.models.abstracts.abstract_embedding import LabelEmbeddingAbstract
from mlmc.modules import Augment

class LabelTuning(LabelEmbeddingAbstract):
    """
     https://arxiv.org/pdf/2203.14655.pdf
    """
    def __init__(self, dropout=0.5, *args, **kwargs):
        """
         Zeroshot model based on cosine distance of embedding vectors.
        This changes the default activation to identity function (lambda x:x)
        Args:
            entailment_output: the format of the entailment output if NLI pretraining is used. (experimental)
            *args:
            **kwargs:
        """
        super(LabelTuning, self).__init__(*args, **kwargs)
        self._config["dropout"] = dropout
        self.create_labels(self.classes)
        self.dropout = torch.nn.Dropout(dropout)

        self.projection = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim, bias=False)
        torch.nn.init.eye_(self.projection.weight)
        self.build()
        with torch.no_grad():
            self.label_start = self.embedding(**self.label_dict)[1].detach()

    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, embedding=False, *args, **kwargs):
        input_embedding = self.embed_input(x[0])
        self.curr_label_embedding = self.embed_input(x[1])

        input_embedding = self._mean_pooling(input_embedding, x[0]["attention_mask"])
        self.curr_label_embedding = self.projection(self._mean_pooling(self.curr_label_embedding, x[1]["attention_mask"]))

        r = torch.matmul((input_embedding), (self.curr_label_embedding).t())
        if embedding:
            return r, input_embedding
        return r

    def regularize(self):
        return 0.001*((self.label_start - self.curr_label_embedding)**2).mean().sqrt()

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

    def _contrastive_embedding(self, x, y):
        x = self.transform(x)
        input_embedding = self.embed_input(x[0])
        input_embedding = self._mean_pooling(input_embedding, x[0]["attention_mask"])
        y = self.transform(y)
        y_embedding = self.embed_input(y[0])
        y_embedding = self._mean_pooling(y_embedding, y[0]["attention_mask"])
        return input_embedding, y_embedding