import torch
from mlmc.data.transformer import label_smoothing_random
class NoiseSmoothBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super(NoiseSmoothBCEWithLogitsLoss, self).__init__()
        self.smoothing_noise = 0.2

    def forward(self, inputs, targets):
        loss = super(NoiseSmoothBCEWithLogitsLoss,self).forward(label_smoothing_random(inputs, self.smoothing_noise), targets)
        return loss


nsc = NoiseSmoothBCEWithLogitsLoss()
bsc = torch.nn.BCEWithLogitsLoss()
p = torch.abs(torch.rand((4,64))-0.3)
y = p.round()
p = torch.log(p)
print(nsc(p,y),bsc(p,y))