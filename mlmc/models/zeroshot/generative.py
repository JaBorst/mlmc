from ..abstracts.abstract_textclassification import TextClassificationAbstract
import torch
class Generator(TextClassificationAbstract):
    def __init__(self, *args, **kwargs):
        super(Generator,self).__init__(*args, **kwargs)
        self.create_labels(self.classes)
        self.build()

    def forward(self, x):
        p = self.embedding(**x).logits.mean(1)#[:,-1,:]
        # r = ((p[:,self.labels.to(self.device)["input_ids"]]*self.labels["attention_mask"][None]) / self.labels["attention_mask"][None].sum(-1, keepdims=True)).sum(-1)
        r = ((p[:, self.labels.to(self.device)["input_ids"]] * self.labels["attention_mask"][None])).sum(-1)
        return r


    def transform(self, x, h=None, max_length=None) -> dict:
        """
        A standard transformation function from text to network input format

        The function looks for the tokenizer attribute. If it doesn't exist the transform function has to
        be implemented in the child class

        Args:
            x: A list of text

        Returns:
            A tensor in the network input format.

        """
        assert hasattr(self, 'tokenizer'), "If the model does not have a tokenizer attribute, please implement the" \
                                           "transform(self, x)  method yourself. TOkenizer can be allocated with " \
                                           "embedder, tokenizer = mlmc.representation.get_embedding() or " \
                                           "embedder, tokenizer = mlmc.representation.get_transformer()"
        if max_length is None:
            max_length = self._config["max_len"]
        return {k: v.to(self.device) for k, v in
                self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt', pad_to_max_length = True).items()}


    def create_labels(self, classes):
        self.classes = classes
        self._config["classes"] = classes
        self._config["n_classes"] = len(classes)
        self.n_classes = len(classes)
        self.labels = self.tokenizer(list(self.classes.keys()), add_special_tokens=False, padding=True,
                                     return_tensors="pt")

    def _init_input_representations(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.embedding = AutoModelForCausalLM.from_pretrained(self.representation, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.representation, padding_side="left", pad_token='¥')
        # if self.tokenizer.pad_token is None: self.tokenizer.pad_token = '[PAD]'
        with torch.no_grad(): self.embeddings_dim = self.embedding(**self.tokenizer(["test"], return_tensors="pt"))[0].shape[-1]




