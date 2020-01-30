from .data_loaders  import load_eurlex, load_wiki30k, load_huffpost, load_appd, load_rcv1, load_conll2003en, \
    load_moviesummaries,load_blurbgenrecollection, load_blurbgenrecollection_de, load_20newsgroup,export
# register = {fct.split("load_")[1]:dl.__dict__[fct] for fct in dir(dl) if fct.startswith("load_")}
register = {
    "appd": load_appd,
    "rcv1": load_rcv1,
    "huffpost": load_huffpost,
    "wiki30k": load_wiki30k,
    "eurlex": load_eurlex,
    "movies_summaries": load_moviesummaries,
    "conll2003en": load_conll2003en,
    "blurbgenrecollection": load_blurbgenrecollection,
    "blurbgenrecollection_de": load_blurbgenrecollection_de,
    "20newsgroup": load_20newsgroup
}

from torch.utils.data import Dataset, DataLoader
import torch
class MultiLabelDataset(Dataset):
    def __init__(self, x, y, classes, purpose="train", target_dtype=torch.LongTensor):
        self.classes = classes
        self.purpose = purpose
        self.x = x
        self.y = y
        self.target_dtype = target_dtype

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        labels = [self.classes[tag] for tag in self.y[idx]]
        labels = torch.nn.functional.one_hot(torch.LongTensor(labels), len(self.classes)).sum(0)
        return {'text': self.x[idx], 'labels': self.target_dtype(labels)}

    def transform(self, fct):
        self.x = [fct(sen) for sen in self.x]

class SequenceDataset(Dataset):
    def __init__(self, x, y, classes, purpose="train", target_dtype=torch._cast_Long):
        self.classes = classes
        self.purpose = purpose
        self.x = x
        self.y = y
        self.target_dtype = target_dtype

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'text': " ".join(self.x[idx]), 'labels': " ".join([str(x) for x in self.y[idx]])} #self.target_dtype(labels)}

def get_dataset(name, type, ensure_valid=False, valid_split=0.25, target_dtype=torch.LongTensor):
    data, classes = register.get(name, None)()
    if data is None:
        Warning("data not found")
        return None
    else:
        if "valid" not in data.keys():
            print("No Validation data found.")
            if ensure_valid:
                print("Providing random split...")
                from sklearn.model_selection import train_test_split
                splits_from_train = train_test_split(*data["train"], test_size=valid_split)
                train=[splits_from_train[0],splits_from_train[2]]
                valid = [splits_from_train[1],splits_from_train[3]]
                data["train"]=train
                data["valid"]=valid
            else:
                data["valid"]= None
        datasets = {
            split: type(x=data[split][0],
                                     y=data[split][1],
                                     classes=classes,
                                     purpose=split,
                                     target_dtype=target_dtype) if data[split] is not None else None
        for split in ["test","train","valid"]
        }
        for k in data.keys():
            if k not in ["test", "train", "valid"]:
                datasets[k] = data[k]
        datasets["classes"]=classes

        return datasets

def sample(dataset, fraction=None, absolute=None):
    from numpy.random import choice
    n_samples = absolute if absolute is not None else  int(fraction*len(dataset))
    ind = choice(range(len(dataset)), n_samples)
    x = [dataset.x[i] for i in ind]
    y = [dataset.y[i] for i in ind]
    return type(dataset)(x=x, y=y, classes=dataset.classes, target_dtype=dataset.target_dtype)

# d = get_dataset("conll2003en", type=mlmc.data.SequenceDataset, target_dtype=torch._cast_Long)
# for b in torch.utils.data.DataLoader(d["train"], batch_size=15): break