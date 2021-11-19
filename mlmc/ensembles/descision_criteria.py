import torch

class Decision(torch.nn.Module):
    def __init__(self):
        super(Decision, self).__init__()

    def forward(self, scores):
        pass

class MajorityDecision(Decision):
    def forward(self, scores):
        scores_stack = torch.stack(scores, -1)
        return scores_stack.argmax(-2).mode().indices

class ConfidenceDecision(Decision):
    def forward(self, scores):
        scores_stack = torch.stack(scores,-1)
        return scores_stack.max(-2)[0].argmax(-1)

class EntropyDecision(Decision):
    def forward(self, scores):
        scores_stack = torch.stack(scores,-1)
        return (-(scores_stack.softmax(-2)*scores_stack.log_softmax(-2)).sum(-2)).argmin(-1)

