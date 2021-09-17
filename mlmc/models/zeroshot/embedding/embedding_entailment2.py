import torch
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...abstracts.abstract_sentence import SentenceTextClassificationAbstract

class EmbeddingBasedEntailment(SentenceTextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
     Zeroshot model based on cosine distance of embedding vectors.
    """
    def __init__(self, mode="vanilla",  *args, **kwargs):
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
        super(EmbeddingBasedEntailment, self).__init__(*args, **kwargs)
        self.modes = ("vanilla","max","mean","max_mean", "attention","attention_max_mean")
        assert mode in self.modes, f"Unknown mode: '{mode}'!"
        self.set_mode(mode=mode)

        self.create_labels(self.classes)
        self.bottle_neck = 384
        self.parameter = torch.nn.Linear(self.embeddings_dim,self.bottle_neck)
        self.parameter2 = torch.nn.Linear(self.bottle_neck,self.embeddings_dim)
        self.att = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.bottle_neck, nhead=8),num_layers=8)
        # self.att = torch.nn.MultiheadAttention(self.bottle_neck+self.embeddings_dim, num_heads=8)
        self.entailment_projection = torch.nn.Linear(3*self.embeddings_dim, self.bottle_neck)
        self.entailment_projection2 = torch.nn.Linear(self.bottle_neck, 3)
        self.hypothesis = torch.nn.Parameter(torch.tensor(torch.rand((1,self.bottle_neck))))
        self.build()


    def set_mode(self, mode):
        """Set weighting mode"""
        self.mode = mode.split("_")
        self._config["mode"] = mode

    def forward(self, x, *args, **kwargs):
        input_embedding = self.embedding(**x)[0]
        label_embedding = self.embedding(**self.label_dict)[0]



        if self._config["target"] == "entailment":
            le = self.parameter(label_embedding)
            word_scores = torch.softmax(torch.einsum("ijk,ink->ijn", input_embedding, label_embedding),-1)
            interaction = torch.einsum("bmn,bne->bme", word_scores, le)
            # interaction = self.parameter(input_embedding)+ interaction
            interaction = self.att(interaction)
            # word_scores = torch.softmax(torch.einsum("ijk,ink->ijn", interaction, le),-1)
            # interaction = torch.einsum("bmn,bne->bme", word_scores, le)
            interaction = self._mean_pooling(interaction, x["attention_mask"])
            interaction = self.parameter2(interaction)


            input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
            e = torch.cat([#input_embedding,
                           #label_embedding,
                           torch.abs(input_embedding - label_embedding),
                           torch.abs(input_embedding + label_embedding),
                interaction],-1)
            # e = interaction
        else:
            le = self.parameter(label_embedding)
            word_scores = torch.softmax(torch.einsum("ijk,lnk->iljn", input_embedding, label_embedding),-1)
            interaction = torch.einsum("blmn,lne->blme", word_scores, le)
            # interaction = self.parameter(input_embedding[:,None].repeat(1,interaction.shape[1],1,1))+interaction
            b,l,m,e = interaction.shape
            interaction = self.att(interaction.reshape((b*l,m,e))).reshape((b,l,m,e))
            # interaction,_ = self.att(interaction.reshape((b*l,m,e)),interaction.reshape((b*l,m,e)),interaction.reshape((b*l,m,e)))
            interaction = interaction.reshape((b,l,m,e))
            # word_scores = torch.softmax(torch.einsum("ijk,lnk->iljn", interaction, le), -1)
            # interaction = torch.einsum("blmn,lne->blme", word_scores, le)
            interaction = self.parameter2(interaction)


            input_mask_expanded = x["attention_mask"].unsqueeze(1).unsqueeze(-1).float()
            sum_mask = torch.clamp(input_mask_expanded.sum(2), min=1e-9)
            interaction = torch.sum(interaction * input_mask_expanded, 2)/sum_mask
            input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
            label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])

            e = torch.cat([#input_embedding[:,None].repeat((1,label_embedding.shape[0],1)),
                           # label_embedding[None].repeat((input_embedding.shape[0],1,1)),
                            torch.abs(input_embedding[:, None] - label_embedding[None]),
                            torch.abs(input_embedding[:, None] + label_embedding[None]),
                           interaction],-1)
            #
            # e = interaction

        logits = self.entailment_projection2(torch.tanh(self.entailment_projection(e)))

        if self._config["target"] == "entailment":
            pass
        elif self._config["target"] == "single":
            logits = torch.log(logits[...,-1].softmax(-1))
        elif self._config["target"] == "multi":
            logits = torch.log(e[..., [0, 2]].softmax(-1)[..., -1])
        else:
            assert not self._config["target"], f"Target {self._config['target']} not defined"
        return logits

    def scores(self, x):
        """
        Returns 2D tensor with length of x and number of labels as shape: (N, L)
        Args:
            x:

        Returns:

        """
        self.eval()
        assert not (self._config["target"] == "single" and   self._config["threshold"] != "max"), \
            "You are running single target mode and predicting not in max mode."

        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
            if self._loss_name == "ranking":
                output = 0.5*(output+1)
        self.train()
        return output

    def single(self, loss="ranking"):
        """Helper function to set the model into single label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "single"
        self.set_threshold("max")
        self.set_activation(lambda x: x)
        self._loss_name = loss
        if loss == "ranking":
            self.set_loss(RelativeRankingLoss(0.5))
        else:
            self.set_loss(torch.nn.CrossEntropyLoss())
        self._all_compare=True

    def multi(self, loss="ranking"):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self.set_threshold("mcut")
        self.set_activation(lambda x: x)
        self._loss_name = loss
        if loss == "ranking":
            self.set_loss(RelativeRankingLoss(0.5))
        else:
            self.set_loss(torch.nn.BCELoss)
        self._all_compare=True

    def sts(self):
        """Helper function to set the model into multi label mode"""
        from ....loss import RelativeRankingLoss
        self._config["target"] = "multi"
        self._loss_name="ranking"
        self.set_threshold("hard")
        self.set_activation(lambda x: x)
        self.set_loss(RelativeRankingLoss(0.5))

    def entailment(self):
        self._config["target"] = "entailment"
        self.target = "entailment"
        self.set_sformatter(lambda x: x)
        self.set_threshold("max")
        self.set_activation(torch.softmax)
        self.set_loss = torch.nn.CrossEntropyLoss()
        self._all_compare = False
