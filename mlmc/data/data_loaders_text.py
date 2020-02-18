from torch.utils.data import Dataset

class RawTextDataset(Dataset):
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
        return {'input': self.tokenized[(idx+1):(idx+self.length+1)],
                'forward': self.tokenized[(idx+1+1):(idx+self.length+1+1)],
                'backward':self.tokenized[(idx):(idx+self.length)]}


