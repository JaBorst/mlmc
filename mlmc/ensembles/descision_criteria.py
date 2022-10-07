import torch

class Decision(torch.nn.Module):
    def __init__(self):
        super(Decision, self).__init__()

    def forward(self, scores):
        pass

class MajorityDecision(Decision):
    def forward(self, scores):
        s_max = scores.argmax(-2)
        s = torch.nn.functional.one_hot(s_max, scores.shape[-2])
        who = (s.sum(-2).argmax(-1).unsqueeze(-1) == s_max).int()
        return who.argmax(-1)

class ConfidenceDecision(Decision):
    def forward(self, scores):
        return scores.max(-2)[0].argmax(-1)

class EntropyDecision(Decision):
    def forward(self, scores):
        return (-(scores.softmax(-2)*scores.log_softmax(-2)).sum(-2)).argmin(-1)

