import torch
from ..data.transformer import label_smoothing_random


class NoiseSmoothBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Torch like loss function but with smoothed labels. [Confident Learning]
    ToDo:
     - Make the smoothing function an argument
    """
    def __init__(self):
        super(NoiseSmoothBCEWithLogitsLoss, self).__init__()
        self.smoothing_noise = 0.2

    def forward(self, inputs, targets):
        loss = super(NoiseSmoothBCEWithLogitsLoss,self).forward(label_smoothing_random(inputs, self.smoothing_noise), targets)
        return loss
