import torch
import mlmc
import numpy as np
from mlmc.representation.label_embeddings import get_wikidata_desc

class TSIDS(mlmc.models.abstracts.SentenceTextClassificationAbstract, mlmc.models.abstracts.TextClassificationAbstractZeroShot):
    def __init__(self, descriptor = get_wikidata_desc, similarity="cosine",  *args, **kwargs):
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
        super(TSIDS, self).__init__(*args, **kwargs)
        self.modes = ("vanilla", "mean",)
        self.set_similarity(similarity=similarity)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 1)
        self.descriptor = descriptor
        self.descriptions = descriptor(list(self.classes.keys()))
        self.targets = [self.transform(list(v)) for v in self.descriptions.values()]
        self.graph = torch.zeros((self.n_classes, len(sum(self.descriptions.values(),[]))))
        s = 0
        for k, v in self.classes.items():
            self.graph[v, s:(s + len(self.descriptions[k]))] = 1
            s += len(self.descriptions[k])
        self.dropout=torch.nn.Dropout(p=0.5)

        self.create_labels(self.classes)

        self.graph = torch.nn.Parameter(self.graph/self.graph.sum(dim=-1, keepdim=True))
        self.graph.requires_grad=False
        self.build()


    def set_similarity(self, similarity):
        """Set weighting mode"""
        self._config["similarity"] = similarity

    def _sim(self, x, y):
        x = x.unsqueeze(-2)
        y = y.unsqueeze(-3)
        if self._config["similarity"] == "cosine":
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            y = y / (y.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            r = (x* y).sum(-1)
            # r = torch.log(0.5 * (r + 1))
        elif self._config["similarity"] == "scalar":
            r = (x * y).sum(-1)
        elif self._config["similarity"] == "manhattan":
            r = - (x - y).abs().sum(-1)
        elif self._config["similarity"] == "entailment":
            x_tup = [1]*len(x.shape)
            y_tup = [1]*len(y.shape)
            x_tup[-2] = y.shape[-2]
            y_tup[-3] = x.shape[-3]

            e = self.entailment_projection(self.dropout(torch.cat([
                x.repeat(*x_tup),
                y.repeat(*y_tup),
                # (x - y).abs()
            ], -1)))
            r = self.entailment_projection2(e).squeeze(-1)
            if self._config["target"] == "entailment":
                r = r.diag()
        return r

    def forward(self, x,return_keywords=False, *args, **kwargs):
        input_embedding = self.embedding(**{k:x[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if k in x})[0]
        target_embedding = [self.embedding(**{k:t[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']if k in t})[0] for t in self.targets]
        label_embedding = self.embedding(**{k:self.label_dict[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']if k in self.label_dict})[0]

        input_embedding = self.dropout(input_embedding)
        target_embedding = [self.dropout(t) for t in target_embedding]
        sdg_embedding = self.dropout(label_embedding)

        if self.training:
            input_embedding = input_embedding + 0.01 * torch.rand_like(input_embedding)[:, 0, None, 0,
                                                           None].round() * torch.rand_like(input_embedding)  #
            input_embedding = input_embedding * \
                                ((torch.rand_like(input_embedding[:, :, 0]) > 0.01).float() * 2 - 1)[..., None]
            input_embedding = input_embedding * ((torch.rand_like(input_embedding[:, :, 0]) > 0.01).float())[
                ..., None]
        target_embedding = [self._mean_pooling(e, x["attention_mask"]) for e, x in zip(target_embedding, self.targets)]

        words = [self._sim(input_embedding, te[None] ) for te, r in zip(target_embedding, self.targets)]
        idf = (1./sum([w.sum([-1]).softmax(-1) for w in words]) )
        tfidf = torch.stack([w.max(2)[0] * idf for w in words],1).softmax(-1)*x["attention_mask"][:, None]

        keyword_embedding = torch.einsum("btw,bwe->bte", tfidf, input_embedding)
        input_embedding = torch.einsum("bwe,bw->be",input_embedding, tfidf.mean(1)*x["attention_mask"])
        sdg_embedding = self._mean_pooling(sdg_embedding, self.label_dict["attention_mask"])

        te = torch.cat(target_embedding, 0)
        te = te - te.mean(0)
        te[0] = te.mean(0)
        te = te / te.norm(p=2, dim=-1, keepdim=True)

        se = sdg_embedding
        se = se - se.mean(0, keepdim=True)
        se[0] = se.mean(0)
        se = se / se.norm(p=2, dim=-1, keepdim=True)

        keyword_embedding = keyword_embedding / keyword_embedding.norm(p=2, dim=-1, keepdim=True)

        keyword_scores = torch.einsum("bse,se->bs", keyword_embedding, se)
        target_scores = (self._sim(input_embedding, te)[:,None] * self.graph[None].to(te.device)).sum(-1)
        sdg_scores = self._sim(input_embedding, se)


        scores = keyword_scores + target_scores + sdg_scores
        if return_keywords:
            return scores, tfidf
        return scores

    def keywords(self, x, y, k=10):
        self.eval()
        with torch.no_grad():
            tok = self.transform(x)
            scores, keywords = self.forward(tok, return_keywords=True)
        import matplotlib.pyplot as plt
        sorted_scores, prediction  = scores.sort(-1)
        idx = scores.argmax(-1)

        label_specific_scores = torch.stack([k[i] for k, i in zip(keywords, idx)])
        keywords = [list(zip(t,l[1:(1+len(t))].tolist())) for t,l in zip(tok["text"], label_specific_scores)]
        keywords_new = []
        for l in keywords:
            new_list = []
            new_tuple = [[], 0]
            for i in range(1, 1+len(l)):
                new_tuple[0].append(l[-i][0])
                new_tuple[1] += l[-i][1]

                if not l[-i][0].startswith("##"):
                    new_tuple[0] = "".join(new_tuple[0][::-1]).replace("##", "")
                    new_list.append(tuple(new_tuple))
                    new_tuple = [[], 0]
            keywords_new.append(new_list)

        prediction_array = self._threshold_fct(scores).cpu().detach()
        prediction = [[self.classes_rev[x] for x in y] for y in prediction.detach().cpu().tolist()]
        binary = np.array([[p in c for p in pred] for c, pred in zip(y, prediction)])
        return [(p[-n:], t, sorted(x, key=lambda x: -x[1])[:k]) for p,n, t,x in zip(prediction, prediction_array.sum(-1).long().tolist(), y, keywords_new)]



    def scores(self, x):
        """
        Returns 2D tensor with length of x and number of labels as shape: (N, L)
        Args:
            x:

        Returns:

        """
        self.eval()
        assert not (self._config["target"] == "single" and self._config["threshold"] != "max"), \
            "You are running single target mode and predicting not in max mode."

        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
        self.train()
        return output

    def transform(self, x, max_length=None) -> dict:
        if max_length is None:
            max_length = self._config["max_len"]
        r = {k: v.to(self.device) for k, v in
                self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt').items()}
        r["text"] = [self.tokenizer.tokenize(s) for s in x]
        return r

    def _init_input_representations(self):
        # TODO: Documentation
        from transformers import AutoModel, AutoTokenizer
        model_class, tokenizer_class, pretrained_weights = AutoModel, AutoTokenizer, self.representation

        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.embedding = model_class.from_pretrained(pretrained_weights, output_attentions=True)
        self.embeddings_dim = self.embedding(torch.tensor([[0]]))[0].shape[-1]
        for param in self.embedding.parameters(): param.requires_grad = self.finetune
        self.embedding.requires_grad = self.finetune

import mlmc
d = mlmc.data.get("rcv1")
model = TSIDS(classes=d["classes"], target="multi", device="cuda:0")
model.evaluate(data=mlmc.data.sampler(d["test"], absolute=1000))