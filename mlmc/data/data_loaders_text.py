from torch.utils.data import Dataset

class RawTextDatasetTokenizer(Dataset):
    """
    Dataset to hold raw text and tokenize batches.
    """
    def __init__(self, x, tokenizer, length=512, bidirectional=False, **kwargs):
        self.__dict__.update(kwargs)
        self.x = x
        self.tokenizer = tokenizer
        self.bidirectional=bidirectional
        self.tokenized=self.tokenizer.encode(x).tokens
        self.length = length

    def __len__(self):
        return len(self.tokenized) - self.length - 2

    def __getitem__(self, idx):
        r = {'input': self.tokenized[(idx+1):(idx+self.length+1)],
             'forward': self.tokenized[(idx+1+1):(idx+self.length+1+1)]}
        if self.bidirectional:
            r['backward'] = self.tokenized[(idx):(idx+self.length)]
        return r

class RawTextDataset(Dataset):
    """
    Dataset to hold raw text and tokenize batches.
    """
    def __init__(self, x, target="words", length=512, bidirectional=False, **kwargs):
        self.__dict__.update(kwargs)
        if callable(target):
            self.x = target(x)
        elif target=="words":
            self.x = x.split()
        elif target =="character":
            self.x = list(x)
        else:
            print("target has to be either 'words' or 'character' or a callable function that splits the input string.")

        self.bidirectional=bidirectional
        self.length = length

    def __len__(self):
        return len(self.x) - self.length - 2

    def __getitem__(self, idx, next=True):
        r = {'input': "".join(self.x[(idx+1):(idx+self.length+1)])}
        if next:
            r['forward'] = self.x[(idx+self.length+1)]
        else:
            r['forward'] = self.x[(idx + 1 + 1):(idx + self.length + 1 + 1)]
            if self.bidirectional:
                r['backward'] = self.x[(idx):(idx+self.length)]
        return r

    def generate_wordlist(self, n=None):
        from collections import Counter
        c = Counter(self.x)
        if n is None:
            n=len(c)
        return [x[0] for x in c.most_common(n)]

from torch import tensor
class RawTextDatasetTensor(Dataset):
    """
    Dataset to hold raw text and tokenize batches.
    """
    def __init__(self, x, vocab, target="words", length=512, bidirectional=False, **kwargs):
        self.__dict__.update(kwargs)
        if callable(target):
            self.x = target(x)
        elif target=="words":
            self.x = x.split()
        elif target =="character":
            self.x = list(x)
        else:
            print("target has to be either 'words' or 'character' or a callable function that splits the input string.")
        self.window = length+1
        self.vocab = dict(zip(vocab, range(len(vocab))))

        sequences = []
        for start in range(0, len(self.x) - self.window):
            sequences.append([x for x in self.x[start:start + self.window]])
        self.x = sequences

        self.tr = tensor([[self.vocab.get(c, len(vocab)+1) for c in s] for s in self.x]).long()

    def __len__(self):
        return self.tr.shape[0]

    def __getitem__(self, idx, raw=False):
        resource = self.x if raw else self.tr
        return resource[idx,:-1], resource[idx,-1]