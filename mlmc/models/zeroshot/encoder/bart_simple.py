from mlmc.models.abstracts.abstract_encoder import EncoderAbstract
import torch
import transformers
from transformers import BartTokenizer, BartForSequenceClassification

class BartEncoder(EncoderAbstract):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(EncoderAbstract, self).__init__(*args, **kwargs)
        # self.decision = torch.nn.Linear(self.embeddings_dim, 1)
        self._all_compare = True
        self.build()


    def transform(self,x, max_length=400, reshape=False, device=None):
        if device is None:
            device=self.device

        if self._all_compare:
            label = list([self._config["sformatter"](x) for x in self.classes]) * len(x)
            text = [s for s in x for _ in range(len(self.classes))]
        else:
            label = list([self._config["sformatter"](x) for x in self.classes])
            text = x
        tok = self.tokenizer(text, label, return_tensors="pt", add_special_tokens=True, padding=True,
                                       truncation=True,
                                       max_length=self.max_len)

        if reshape:
            tok["input_ids"]= tok["input_ids"].reshape((len(x), len(self.classes), -1)).to(device)
            tok["attention_mask"] = tok["attention_mask"].reshape((len(x), len(self.classes), -1)).to(device)
        else:
            tok["input_ids"]= tok["input_ids"].to(device)
            tok["attention_mask"] = tok["attention_mask"].to(device)
        return tok

    def _init_input_representations(self):
        # TODO: Documentation
        if not hasattr(self, "n_layers"): self.n_layers = 1
        try:
            self.tokenizer = BartTokenizer.from_pretrained(self.representation)
            self.embedding = BartForSequenceClassification.from_pretrained(self.representation)
            # self.embeddings_dim = self.embedding(**self.tokenizer("Hello, my dog is cute", return_tensors="pt"), labels=torch.tensor([[1]]))[0].shape[-1]

        except TypeError:
            print("If your using a model that does not support returning hiddenstates, set n_layers=1")
            import sys
            sys.exit()
        for param in self.embedding.parameters(): param.requires_grad = self.finetune
        if self.finetune:
            self.embedding.requires_grad = True


    def forward(self, x, y=None):
        e = self.embedding(**x).logits[:,[0,2]]
        e = e.softmax(-1)[:,-1]
        if self._all_compare:
            return e.reshape((int(e.shape[0]/self.n_classes), self.n_classes))
        else:
            return e