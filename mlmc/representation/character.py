import string

import numpy as np
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
import torch

def charindex(query, maxlen, maxchar, alphabet=string.ascii_letters + string.digits):
    """
    Mapping a query string onto a sequence of integers using the alphabet.
    The Alphabet is a string containing the valid characters in the order they will be assigned to the
    indices (starting at 1).
    Since query can be a list of sentences, which is a list of words,
    maxlen will determine the maximum number of words per sentence (every thing will be padded to this maximum)
    Args:
        query: list of sentences containing lists of words
        maxlen: maximum number of words per sentence
        maxchar: maximum number of characters per
        alphabet: A string of valid characters defaults to string.ascii_letters+string.digits

    Returns: A tensor of indices

    """
    # Creating the dictionary
    alphabet_map = dict(zip(alphabet, range(1, len(alphabet) + 1)))
    # Allocating the array
    result = np.full((len(query), maxlen, maxchar), 0., dtype="float32")
    # Fill the array
    for i, s in enumerate(query):
        indices = pad_sequence(
            [LongTensor([alphabet_map.get(x, len(alphabet) + 1) for x in word]) for word in s.split()],
            batch_first=True).numpy()
        result[i, :min(indices.shape[-2], maxlen), :min(indices.shape[-1], maxchar)] = indices[:maxlen, :maxchar]
    return result


def makemultilabels(query, maxlen, tagset=None):
    if tagset is not None:
        tagsetmap = dict(zip(tagset, range(len(tagset))))
        labels = [[tagsetmap[token] for token in lseq] for lseq in query]
    else:
        labels = query
    return torch.stack([torch.sum(torch.nn.functional.one_hot(torch.LongTensor(x), maxlen), 0) for x in labels], dim=0)
