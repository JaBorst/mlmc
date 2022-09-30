from mlmc.models.abstracts.abstract_embedding import LabelEmbeddingAbstract
import torch
from transformers.tokenization_utils import TruncationStrategy


class AspectEncoder(LabelEmbeddingAbstract):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(AspectEncoder, self).__init__(*args, **kwargs)
        self.entailment_id = self._entailment_classes["entailment"]
        self.contradiction_id = self._entailment_classes["contradiction"]

        self.dropout = torch.nn.Dropout(0.5)
        self.hypo_projection = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim).to(self.device)
        self.hypo_projection2 = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim).to(self.device)


        self.repro = torch.nn.Linear(2*self.embeddings_dim, self.embeddings_dim).to(self.device)
        self.entailment_projection = torch.nn.Linear(self.embeddings_dim, 1).to(self.device)

        self.build()


    def forward(self, x):
        e = self.embedding(**x[0])[1]
        h = self.embedding(**x[1])[1]
        h = self.hypo_projection(self.dropout(self.hypo_projection2(h).relu()))
        ee = torch.cat([e,h],-1)
        e = self.entailment_projection(self.repro(self.dropout(ee)))

        if self._config["target"] in ["single", "multi", "abc"]:
            e = e.reshape((int(x[0]["input_ids"].shape[0] / self._config["n_classes"]), self._config["n_classes"]))
        else:
            assert not self._config["target"], f"Target {self._config['target']} not defined"
        return e

    def _loss(self, x, y):
        """
        Calculating the loss getting  of two tensors using the initiated loss function
        When implementing new models with more complex loss functions, you can reimplement this method in the
        child class to apply them.
        Args:
            x: ouput tensor of the forward pass
            y: true labels

        Returns:
            loss tensor
        """
        # if self.training:
        #     if self._config["target"] in ["abc", "single"]:
        #         cls = torch.nn.functional.one_hot(y, len(self.classes)).flatten()
        #         label = torch.full_like(cls, self.contradiction_id)
        #         label[cls==1] = self.entailment_id
        #     torch.zeros((x.shape[0], len(self._entailment_classes)))
        #     if self._config["target"] == "multi":
        #         label = torch.zeros_like(x)
        #         label[..., self.entailment_id] = torch.nn.functional.one_hot(y, len(self.classes)).flatten()
        #         label[..., self.contradiction_id] = 1 - torch.nn.functional.one_hot(y, len(self.classes)).flatten()
        # else: label=y
        return self.loss(x,y)

    def transform(self,x, h=None, max_length=400, reshape=False, device=None):
        x = [x] if isinstance(x, str) else x
        if device is None:
            device=self.device
        if self._config["target"] == "single" or self._config["target"] == "multi":
            label = list([self._config["sformatter"](x) for x in self._config["classes"]]) * len(x)
            hs = [self._config["sformatter"](cls) for _ in x for cls in self.classes]
            text = [s for s in x for _ in range(len(self._config["classes"]))]
        elif self._config["target"] == "entailment":
            label = h
            text = x
        elif self._config["target"] == "abc":
            label =[self._config["sformatter"](hypo, cls) for hypo in h for cls in self.classes]
            hs =[hypo for hypo in h for cls in self.classes]
            text = [s for s in x for _ in range(len(self._config["classes"]))]

        else:
            # ToDo: Implement
            print(f"{self._config['target']} not yet implemented")
        tok = self.tokenizer( text,label, return_tensors="pt",
                              add_special_tokens=True, padding=True,
                                       truncation=TruncationStrategy.ONLY_FIRST,
                                       max_length=self.max_len)
        hyp = self.tokenizer(hs, return_tensors="pt",
                             add_special_tokens=True, padding=True,
                             truncation=TruncationStrategy.ONLY_FIRST,
                             max_length=self.max_len)

        if reshape:
            tok = {k:v.reshape((len(x), len(self._config["classes"]), -1)).to(device) for k,v in tok.items()}
        else:
            tok = {k: v.to(device) for k, v in tok.items()}
            hyp = {k: v.to(device) for k, v in hyp.items()}

        return tok,hyp
