from ..abstracts.abstract_textclassification import TextClassificationAbstract
import torch
class Generator(TextClassificationAbstract):
    def __init__(self, *args, **kwargs):
        super(Generator,self).__init__(*args, **kwargs)
        self.create_labels(self.classes)
        self.build()

    def forward(self, x):
        p = (self.embedding(**x).logits[:,-1,:]/ x["attention_mask"].sum(-1, keepdims=True)).log_softmax(-1)
        # p = self.embedding(**x).logits[:,-self.labels["input_ids"].shape[-1]:,:]
        r = ((p[:,self.labels["input_ids"]]*self.labels["attention_mask"][None]) / self.labels["attention_mask"][None].sum(-1, keepdims=True))
        # i = torch.nn.functional.one_hot( self.labels["input_ids"].long(), num_classes=self.tokenizer.vocab_size)
        # probs = (p[:,None] * i [None]).sum(-1) *  self.labels["attention_mask"][None] /  self.labels["attention_mask"][None].sum(-1, keepdims=True)
        return r.sum(-1)


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
        h = ", ".join(self.classes.keys())
        tok = self.tokenizer(x,[h] * len(x),  padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt', pad_to_max_length = True)
        # new_shape = tuple(tok["input_ids"].shape)[:-1] + (self.labels["input_ids"].shape[-1],)
        # tok["input_ids"] = torch.cat([tok["input_ids"], torch.full(new_shape, self.tokenizer.pad_token_id)],-1)
        # tok["attention_mask"] = torch.cat([tok["attention_mask"], torch.full(new_shape, 1)],-1)
        # if "token_type_ids" in tok.keys() : tok["token_type_ids"] = torch.cat([tok["token_type_ids"], torch.full(new_shape, 0)],-1)
        return {k: v.to(self.device) for k, v in tok.items()}


    def create_labels(self, classes):
        self.classes = classes
        self._config["classes"] = classes
        self._config["n_classes"] = len(classes)
        self.n_classes = len(classes)
        self.labels = {k:v.to(self.device) for k,v in self.tokenizer(list(self.classes.keys()), add_special_tokens=False, padding=True,
                                     return_tensors="pt").items()}

    def _init_input_representations(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.embedding = AutoModelForCausalLM.from_pretrained(self.representation, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.representation, padding_side="left")
        # if self.tokenizer.pad_token is None: self.tokenizer.pad_token = '[PAD]'
        with torch.no_grad(): self.embeddings_dim = self.embedding(**self.tokenizer(["test"], return_tensors="pt"))[0].shape[-1]




