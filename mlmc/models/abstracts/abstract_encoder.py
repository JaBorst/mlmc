import torch
from .abstract_label import LabelEmbeddingAbstract
from transformers.tokenization_utils import TruncationStrategy


class EncoderAbstract(LabelEmbeddingAbstract):
    def __init__(self, *args, **kwargs):
        super(EncoderAbstract, self).__init__(*args, **kwargs)
        self._all_compare = True


    def transform(self,x, max_length=400, reshape=False, device=None):
        x = [x] if isinstance(x, str) else x
        if device is None:
            device=self._config["device"]
        if self._config["target"] == "single" or self._config["target"] == "multi":
            label = list([self._config["sformatter"](x) for x in self._config["classes"]]) * len(x)
            text = [s for s in x for _ in range(len(self._config["classes"]))]
        else:
            label = self._config["classes"]
            text = x
        tok = self.tokenizer( list(zip(text,label)), return_tensors="pt", add_special_tokens=True, padding=True,
                                       truncation=TruncationStrategy.ONLY_FIRST,
                                       max_length=self.max_len)
        if reshape:
            tok = {k:v.reshape((len(x), len(self._config["classes"]), -1)).to(device) for k,v in tok.items()}
        else:
            tok = {k: v.to(device) for k, v in tok.items()}

        return tok

    def _init_input_representations(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.embedding = AutoModelForSequenceClassification.from_pretrained(self.representation, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.representation)
        self.embeddings_dim = self.embedding(**self.tokenizer(["test"], return_tensors="pt"))[0].shape[-1]
        for param in self.embedding.parameters(): param.requires_grad = self.finetune
