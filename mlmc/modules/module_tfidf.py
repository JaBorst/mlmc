import torch


class TFIDFAggregation(torch.nn.Module):
    def __init__(self, agg=""):
        super(TFIDFAggregation, self).__init__()
        self.agg = agg

    def forward(self, x, y, x_mask=None):
        x_norm = x/x.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            y = [w-w.mean(0)[None] for w in y]
            words = [torch.einsum("ijn,ln->ilj",x_norm , te/te.norm(dim=-1, keepdim=True) ) for te in y]
            words = [(w* x_mask[:,None]) for w in words]

            cidf = (1./(sum([w.sum(1)[0]/ x_mask[:,None].sum(-1) for w in words]) ))
            cidf[cidf.isinf()]=0
            # cidf = cidf.softmax(-1)

            tfidf = torch.stack([(w).max(1)[0] for w in words],-1) * cidf[...,None]

        k = torch.einsum("bwt,bwe->bte", tfidf.softmax(-2), x)
        return words, k, tfidf