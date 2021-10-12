import torch

class MaskNoiseWrapper(torch.nn.Module):
    def __init__(self, loss, warm_up=0):
        super(MaskNoiseWrapper, self).__init__()
        self.loss = loss
        self.average = 0
        self.warm_up = warm_up
        self.step = 0

    def forward(self, x, y):
        l = self.loss(x, y)
        with torch.no_grad():
            mask =  (l < l.mean() + 1*l.std()) #& (l > l.mean() - 1*l.std())
            weight = (1 -l).softmax(-1)
        return (l * mask * weight * l.shape[0]).mean()
