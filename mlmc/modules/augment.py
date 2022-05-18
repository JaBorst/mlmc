import torch


class Augment(torch.nn.Module):
    def __init__(self, vertical_dropout=0.0, word_noise=0.0):
        super(Augment, self).__init__()
        self.vertical_dropout = vertical_dropout
        self.word_noise = word_noise


    def forward(self, x, mask, *args, **kwargs):
        if self.training:
            if self.vertical_dropout > 0:
                x = x * ((torch.rand_like(x[:, :, 0]) > self.vertical_dropout).float())[..., None]
            if self.word_noise > 0:
                x = x + self.word_noise * torch.rand_like(x)[:, 0, None, 0, None].round() * torch.rand_like(x)  #
        return x*mask

