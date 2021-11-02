from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
from ...abstracts.abstract_encoder import EncoderAbstract
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
import torch

class SimpleEncoderClassifier(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(SentenceTextClassificationAbstract, self).__init__(*args, **kwargs)
        self._all_compare = True
        self.p = torch.nn.Linear(self.embeddings_dim,  int(0.5*self.embeddings_dim))
        self.p2 = torch.nn.Linear(  int(0.5*self.embeddings_dim), 1)
        self.build()


    def transform(self,x, max_length=400, reshape=False, device=None):
        x = [x] if isinstance(x, str) else x
        if device is None:
            device=self._config["device"]
        if self._config["target"] == "single" or self._config["target"] == "multi":
            label = list([self._config["sformatter"](x) for x in self._config["classes"]]) * len(x)
            text = [s for s in x for _ in range(len(self._config["classes"]))]
        else:
            label = list([self._config["sformatter"](x) for x in self._config["classes"]])
            text = x
        tok = self.tokenizer(label, text, return_tensors="pt", add_special_tokens=True, padding=True,
                                       truncation=True,
                                       max_length=self.max_len)

        if reshape:
            tok = {k:v.reshape((len(x), len(self._config["classes"]), -1)).to(device) for k,v in tok.items()}
        else:
            tok = {k: v.to(device) for k, v in tok.items()}

        return tok
    def forward(self, x):
        e = self.embedding(**x)[0]
        e = self._mean_pooling(e, x["attention_mask"])
        r = self.p2(self.p(e).relu()).squeeze(-1)
        r = r.reshape((int(x["input_ids"].shape[0] / self._config["n_classes"]), self._config["n_classes"]))
        return r