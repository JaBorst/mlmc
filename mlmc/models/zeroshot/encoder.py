from mlmc.models.abstracts.abstract_encoder import EncoderAbstract
import torch

class Encoder(EncoderAbstract):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(Encoder, self).__init__(*args, **kwargs)
        self.entailment_id = self._entailment_classes["entailment"]
        self.contradiction_id = self._entailment_classes["contradiction"]
        self.build()


    def forward(self, x):
        e = self.embedding(**x).logits
        if self.training:
            return e
        if self._config["target"] == "single":
            e = e[:, self.entailment_id]
            e = e.reshape((int(x["input_ids"].shape[0] / self._config["n_classes"]), self._config["n_classes"]))
        elif self._config["target"] == "multi":
            e = e[:, [self.contradiction_id, self.entailment_id]].log_softmax(-1)[:, -1]
            e = e.reshape(
                (int(x["input_ids"].shape[0] / len(self._config["classes"])), len(self._config["classes"])))
        elif self._config["target"] == "entailment":
            pass
        elif self._config["target"] == "abc":
            e = e[:, self.entailment_id]
            e = e.reshape((int(x["input_ids"].shape[0] / self._config["n_classes"]), self._config["n_classes"]))
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
        if self.training:
            if self._config["target"] in ["abc", "single"]:
                cls = torch.nn.functional.one_hot(y, len(self.classes)).flatten()
                label = torch.full_like(cls, self.contradiction_id)
                label[cls==1] = self.entailment_id

            elif self._config["target"] == "multi":
                label = torch.zeros_like(x)
                label[..., self.entailment_id] = y.flatten()
                label[..., self.contradiction_id] = 1 - y.flatten()
            else:
                label = y
        else: label=y
        return self.loss(x,label)

    def _contrastive_step(self, b):
        if not hasattr(self, "_contrastive_loss" ):
            self._contrastive_loss = torch.nn.CrossEntropyLoss()
        classification_target = self._config["target"]
        self._config["target"] = "entailment"
        x = self._contrastive_embedding(list(b[0]), list(b[1]))
        target = torch.tensor([self._entailment_classes[{0:"contradiction", 1:"entailment"}[l.int().item()]] for l in b[2]])
        l = self._contrastive_loss(x, target.to(self.device))
        self._config["target"] = classification_target
        return l

    def _sample(self, C, n=10000):
        import random
        labels = torch.rand((n,)).round()
        triplets = []
        for label in  labels:
            if label == 1:
                cls = random.choice(list(self.classes.keys()))
                sp = random.choices(C[cls],k=2)
            else:
                cls1 = random.choice(list(self.classes.keys()))
                cls2 = random.choice(list(set(self.classes.keys()) - set([cls1])))
                sp = random.choice(C[cls1]), random.choice(C[cls2])
            triplets.append((*sp, label,))
        return triplets

    def _contrastive_embedding(self, x, y):
        # activation = {}
        # def get_activation(name):
        #     def hook(model, input, output):
        #         activation[name] = output
        #     return hook
        # # self.embedding.model.decoder.layernorm_embedding.register_forward_hook(get_activation('embedding'))
        # self.embedding.base_model.pooler.register_forward_hook(get_activation('embedding'))
        h = self.transform(x,y)
        h = self.embedding(**h).logits
        return h


