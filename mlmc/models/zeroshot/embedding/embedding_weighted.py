import torch
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract

class EmbeddingBasedWeighted(SentenceTextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, mode="vanilla", entailment_output=3, *args, **kwargs):
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
        super(EmbeddingBasedWeighted, self).__init__(*args, **kwargs)
        self.modes = ("vanilla","max","mean","max_mean", "attention","attention_max_mean")
        assert mode in self.modes, f"Unknown mode: '{self.mode}'!"
        self.set_mode(mode=mode)

        self.create_labels(self.classes)
        self.parameter = torch.nn.Linear(self.embeddings_dim,256)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, entailment_output)
        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, *args, **kwargs):
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]

        if "mean" in self.mode:
            label_embedding = label_embedding - label_embedding.mean(0)
            # print("mean added")


        if "attention" in self.mode or "max" in self.mode:
            input_embedding2 = input_embedding / input_embedding.norm(p=2, dim=-1, keepdim=True)
            label_embedding2 = label_embedding / label_embedding.norm(p=2, dim=-1, keepdim=True)
            word_scores = torch.einsum("ijk,lnk->iljn", input_embedding2, label_embedding2)
            # print("attnetion or max added")

        if "attention" in self.mode:
            attentions = torch.relu(word_scores.mean(-1))
            input_embedding = self._mean_pooling((attentions[..., None] * input_embedding[:, None]).transpose(1, 2),
                                                 x["attention_mask"][:, :, None])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
            input_embedding = input_embedding / (input_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            label_embedding = label_embedding / (label_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            r = (input_embedding * label_embedding[None]).sum(-1)
            # print("attnetion added")
        else:
            input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
            input_embedding = input_embedding / (input_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            label_embedding = label_embedding / (label_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            r = torch.matmul((input_embedding), (label_embedding).t())

        if "max" in self.mode:
            word_maxs = word_scores.reshape((input_embedding.shape[0], label_embedding.shape[0],-1)).max(-1)[0]
            r = r * word_maxs
        return 0.5*(r+1)

    def single(self):
        """Helper function to set the model into single label mode"""
        self._config["target"] = "single"
        self.set_threshold("max")
        self.set_activation(lambda x: x)
        self.set_loss(torch.nn.CrossEntropyLoss())
        self._all_compare=True
        self.build()

    def multi(self):
        """Helper function to set the model into multi label mode"""
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)
        self.set_loss(torch.nn.BCELoss())
        self._all_compare=True
        self.build()

    def sts(self):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)
        self.set_loss(RelativeRankingLoss(0.5))
        self._all_compare = True
        self.build()

    def entailment(self):
        self._config["target"] = "single"
        self._config["entailment_output"] = 3
        self.target = "single"
        self.set_sformatter(lambda x: x)
        self.set_threshold("max")
        self.set_activation(torch.softmax)
        self.set_loss = torch.nn.CrossEntropyLoss()
        self._all_compare = False
        self.build()
