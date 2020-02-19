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

    def __getitem__(self, idx):
        r = {'input': self.x[(idx+1):(idx+self.length+1)],
             'forward': self.x[(idx+1+1):(idx+self.length+1+1)]}
        if self.bidirectional:
            r['backward'] = self.x[(idx):(idx+self.length)]
        return r

    def generate_wordlist(self, n=None):
        from collections import Counter
        c = Counter(self.x)
        if n is None:
            n=len(c)
        return [x[0] for x in c.most_common(n)]