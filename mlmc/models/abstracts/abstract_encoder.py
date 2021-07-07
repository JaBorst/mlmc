import torch
from .abstracts import TextClassificationAbstract



class EncoderAbstract(TextClassificationAbstract):
    def __init__(self, sformatter = lambda x: f"This is about {x}", label_length=15, *args, **kwargs):
        super(EncoderAbstract, self).__init__(*args, **kwargs)
        self._config["sformatter"] = sformatter
        self._config["label_length"] = label_length
        self._all_compare = True

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        self._config["classes"] = classes
        self._config["n_classes"] = len(classes)
        if hasattr(self, "classes_rev"):
            del self.classes_rev

    def set_sformatter(self, c):
        assert callable(c)
        self._config["sformatter"] = c

    def transform(self,x, max_length=400, reshape=False, device=None):
        x = [x] if isinstance(x, str) else x
        if device is None:
            device=self.device

        if self._config["target"] == "single" or self._config["target"] == "multi":
            label = list([self._config["sformatter"](x) for x in self._config["classes"]]) * len(x)
            text = [s for s in x for _ in range(len(self._config["classes"]))]
        else:
            label = list([self._config["sformatter"](x) for x in self._config["classes"]])
            text = x
        tok = self.tokenizer(text, label, return_tensors="pt", add_special_tokens=True, padding=True,
                                       truncation=True,
                                       max_length=self.max_len)

        if reshape:
            tok = {k:v.reshape((len(x), len(self._config["classes"]), -1)).to(device) for k,v in tok.items()}
        else:
            tok = {k: v.to(device) for k, v in tok.items()}

        return tok

    def single(self):
        self._config["target"] = "single"
        self.target = "single"
        self.set_threshold("max")
        self.activation = lambda x: x
        self.loss = torch.nn.CrossEntropyLoss()
        # self.build()

    def multi(self):
        self._config["target"] = "multi"
        self.target = "multi"
        self.set_threshold("mcut")
        self.activation = lambda x: x
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.build()

    def sts(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "sts"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)

    def entailment(self):
        """Helper function to set model into default multi label mode"""
        self._config["target"] = "entailment"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)
        self.loss = torch.nn.CrossEntropyLoss()

    def _init_input_representations(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.embedding = AutoModelForSequenceClassification.from_pretrained(self.representation, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.representation)
        for param in self.embedding.parameters(): param.requires_grad = self.finetune
