from .abstract_textclassification import TextClassificationAbstract

class LabelEmbeddingAbstract(TextClassificationAbstract):
    """
    This extends the basic classification functionality with methods for embedding and transforming labels
    """
    def __init__(self, sformatter = lambda x: f"This is about {x}", label_length=15, *args, **kwargs):
        super(LabelEmbeddingAbstract, self).__init__(*args, **kwargs)
        self._config["sformatter"] = sformatter
        self._config["label_length"] = label_length

    def set_sformatter(self, c):
        """
        Setter for the label sformatter
        Args:
            c: callable that takes and returns a string

        Returns:

        """
        assert callable(c)
        self._config["sformatter"] = c

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
            self.label_dict = self.label_embed( self._config["classes"])

