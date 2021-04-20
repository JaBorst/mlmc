from ..abstracts.abstract_encoder import EncoderAbstract
import torch

class SimpleEncoder(EncoderAbstract):
    """
    Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
    on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
    """
    def __init__(self, *args, **kwargs):
        """Only there to initialize a projection for binary classification"""
        super(EncoderAbstract, self).__init__(*args, **kwargs)
        self.decision = torch.nn.Linear(self.embeddings_dim, 1)
        self.build()

    def forward(self, x):
        e = self.embedding(x["input_ids"])[1]
        # if isinstance(e[1], tuple):
        #     e = e[1][0][:,-1,:]
        # else:
        #     e = e
        if self._all_compare:
            return self.decision(e).squeeze(-1).reshape((int(x["input_ids"].shape[0]/len(self.classes)), len(self.classes)))
        else:
            return self.decision(e).squeeze(-1)