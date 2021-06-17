import torch

class RelativeRankingLoss(torch.nn.Module):
    """
    This considers only relative similarities, y = 1, x1 should be large than x2.
    """
    def __init__(self, margin=0.5, act=None):
        """
        Initialize RelativeRankingLoss
        :param margin:  margin of the loss function
        :param act:  If necessary an additional activation function before applying the loss.
        """
        super(RelativeRankingLoss, self).__init__()
        self.margin = margin
        self.act = act

    def forward(self, inputs, targets):
        if self.act is not None:
            inputs = self.act(inputs)
        if targets.shape != inputs.shape:
            targets = torch.nn.functional.one_hot(targets, inputs.shape[-1])
        inputs_rel = (inputs[..., None,:] - inputs[..., None])
        target_rel = (targets[..., None,:] - targets[..., None])
        l1 = torch.relu(-inputs_rel[target_rel > 0] + self.margin)
        l2 = torch.relu(inputs_rel[target_rel < 0] + self.margin)
        return l1.mean() + l2.mean()
