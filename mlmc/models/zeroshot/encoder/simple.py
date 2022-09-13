from mlmc.models.abstracts.abstract_encoder import EncoderAbstract
import torch

class SimpleEncoder(EncoderAbstract):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(SimpleEncoder, self).__init__(*args, **kwargs)
        self.entailment_id = self._entailment_classes["entailment"]
        self.contradiction_id = self._entailment_classes["contradiction"]
        self.build()


    def forward(self, x):
        e = self.embedding(**x)[0]
        if self._config["target"] == "single":
            e = e.log_softmax(-1)[:, self.entailment_id]
            e = e.reshape((int(x["input_ids"].shape[0] / self._config["n_classes"]), self._config["n_classes"]))
        elif self._config["target"] == "multi":
            e = e[:, [self.contradiction_id, self.entailment_id]].log_softmax(-1)[:, -1]
            e = e.reshape(
                (int(x["input_ids"].shape[0] / len(self._config["classes"])), len(self._config["classes"])))
        elif self._config["target"] == "entailment":
            pass
        elif self._config["target"] == "abc":
            e = e.log_softmax(-1)[:, self.entailment_id]
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
        return self.loss(x,y)