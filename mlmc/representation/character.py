import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor

def charindex(query, maxlen, maxchar, alphabet):
    alphabet_map = dict(zip(alphabet, range(1, len(alphabet)+1)))
    result = np.full((len(query),maxlen,maxchar), 0., dtype="float32")
    for i, s in enumerate(query):
        indices=pad_sequence([LongTensor([alphabet_map.get(x, len(alphabet)+1)for x in word ]) for word in s.split()],batch_first=True).numpy()
        result[i, :min(indices.shape[-2],maxlen), :min(indices.shape[-1], maxchar)] = indices[:maxlen, :maxchar]
    return result
