from mlmc.models.abstracts.abstract_encoder import EncoderAbstract
from ...abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
import torch

class SimpleEncoder(EncoderAbstract, TextClassificationAbstractZeroShot):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(SimpleEncoder, self).__init__(*args, **kwargs)
        self._all_compare = True
        self.build()

    def forward(self, x):
        e = self.embedding(**x)[0]

        if self._config["target"] == "entailment":
            pass
        elif self._config["target"] == "single":
            e = torch.log(e[:,-1].softmax(-1))
            e = e.reshape((int(x["input_ids"].shape[0] / self._config["n_classes"]), self._config["n_classes"]))
        elif self._config["target"] == "multi":
            e = torch.log(e[:, [0, 2]].softmax(-1)[:, -1])
            e = e.reshape((int(x["input_ids"].shape[0] / len(self._config["classes"])), len(self._config["classes"])))
        else:
            assert not self._config["target"], f"Target {self._config['target']} not defined"

        return e