import torch


class LDAM(torch.nn.Module):
    """Torch like loss function but with smoothed labels. [Confident Learning]
    ToDo:
     - Make the smoothing function an argument
    """
    def __init__(self):
        super(LDAM, self).__init__()
        self.averages = None
        self.c = 1


    def forward(self, inputs, targets):
        s = targets.sum(0)
        if self.averages is None or self.averages.shape != s.shape:
            self.averages = s+1
            self.c += 4
        else:
            self.averages += s
        # self.c += inputs.shape[0]
        inputs = inputs/inputs.norm(p=1,dim=-1, keepdim=True)
        m_is =  torch.exp(inputs-self.c/(self.averages.float()[None]))
        norm = -torch.exp(inputs) + torch.exp(inputs).sum(-1).unsqueeze(-1)
        loss = (-torch.log(m_is / ( m_is + norm))) / inputs.shape[-1]
        return loss.mean()
