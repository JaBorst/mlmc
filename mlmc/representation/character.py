import string

import numpy as np
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
import torch

def makemultilabels(query, maxlen, tagset=None):
    if tagset is not None:
        tagsetmap = dict(zip(tagset, range(len(tagset))))
        labels = [[tagsetmap[token] for token in lseq] for lseq in query]
    else:
        labels = query
    return torch.stack([torch.sum(torch.nn.functional.one_hot(torch.LongTensor(x), maxlen), 0) for x in labels], dim=0)
