import torch


class Augment(torch.nn.Module):
    """
    Taken from
    Dinghan Shen, Mingzhi Zheng, Yelong Shen, Yanru Qu, Weizhu Chen: A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation

    https://arxiv.org/pdf/2009.13818.pdf"""
    def __init__(self, word_cutoff=0.0, feature_cutoff=0.0, span_cutoff=0.0,  word_noise=0.0,):
        super(Augment, self).__init__()
        self.word_cutoff = word_cutoff
        self.word_noise = word_noise
        self.span_cutoff = span_cutoff
        self.feature_cutoff = feature_cutoff
        self.on()

    def forward(self, x, mask, *args, **kwargs):
        if self.training and self._on:
            if self.word_noise > 0:
                x = x + self.word_noise * torch.rand_like(x)[:, 0, None, 0, None].round() * torch.rand_like(x)
            if self.word_cutoff > 0:
                x = x * ((torch.rand_like(x[:, :, 0]) > self.word_cutoff).float())[..., None]
            if self.feature_cutoff > 0:
                x = x * ((torch.rand_like(x[:, 0,  :]) > self.feature_cutoff).float())[..., None, :]
            if self.span_cutoff > 0:
                raise NotImplementedError
        return x*mask

    def off(self):
        self._on = False

    def on(self):
        self._on = True
