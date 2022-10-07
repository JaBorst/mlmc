from .abstract_textclassification import TextClassificationAbstract

class LabelEmbeddingAbstract(TextClassificationAbstract):
    """
    This extends the basic classification functionality with methods for embedding and transforming labels
    """
    def __init__(self, label_length=15, *args, **kwargs):
        super(LabelEmbeddingAbstract, self).__init__(*args, **kwargs)

        self._config["label_length"] = label_length

    def label_embed(self, x):
        """
        Label embedder in this instance uses the same transformation as the input
        Args:
            x:

        Returns:

        """
        return self.transform([self._config["sformatter"](l) for l in list(x)],max_length=self._config["label_len"] )

    def create_labels(self, classes: dict):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        self._config["classes"] = classes
        self.classes = classes
        self._config["n_classes"] = len(self._config["classes"])

        if isinstance(self._config["classes"], dict):
            self.classes_rev = {v: k for k, v in  self._config["classes"].items()}

        if self._config["n_classes"] != 0: # To ensure we can initialize the model without specifying classes
            self.label_dict = {k: v.to(self.device) for k, v in
                    self.tokenizer(list(self._config["classes"].keys()), padding=True, max_length=self._config["label_len"], truncation=True,
                                   add_special_tokens=True, return_tensors='pt').items()}

    def transform(self, x, h=None, max_length=None):
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
                                           "transform(self, x)  method yourself. Tokenizer can be allocated with " \
                                           "embedder, tokenizer = mlmc.representation.get_embedding() or " \
                                           "embedder, tokenizer = mlmc.representation.get_transformer()"
        if max_length is None:
            max_length = self._config["max_len"]
        if self._config["target"] == "single" or self._config["target"] == "multi":
            return ({k: v.to(self.device) for k, v in
                    self.tokenizer(x, h, padding=True, max_length=max_length, truncation=True,
                                   add_special_tokens=True, return_tensors='pt').items()}, self.label_dict)
        elif self._config["target"] == "entailment":
            assert h is not None, "not hypothesis in Dataset"
            return ({k: v.to(self.device) for k, v in
                     self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                             add_special_tokens=True, return_tensors='pt').items()},
            {k: v.to(self.device) for k, v in
                     self.tokenizer(h, padding=True, max_length=max_length, truncation=True,
                             add_special_tokens=True, return_tensors='pt').items()})
        elif self._config["target"] == "abc":
            assert h is not None, "not hypothesis in Dataset"


            hypo_t = self.tokenizer([self._config["sformatter"](hypo,cls) for hypo in h for cls in self.classes], padding=True, max_length=max_length, truncation=True,
                           add_special_tokens=True, return_tensors='pt')


            return ({k: v.to(self.device) for k, v in
                     self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                                    add_special_tokens=True, return_tensors='pt').items()},
                    {k: v.to(self.device) for k, v in hypo_t.items()}
                    )
        else:
            assert False, f"No Transformation Rule for {self._config['target'] }"
