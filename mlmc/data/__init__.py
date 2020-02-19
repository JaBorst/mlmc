from torch.utils.data import Dataset
import torch
from .data_loaders  import load_eurlex, load_wiki30k, load_huffpost, load_aapd, load_rcv1, load_conll2003en, \
    load_moviesummaries,load_blurbgenrecollection, load_blurbgenrecollection_de, load_20newsgroup,export

# String Mappings
register = {
    "aapd": load_aapd,
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



class MultiLabelDataset(Dataset):
    """Dataset to hold text and label combinations. Also on __getitem__ the labels are
    transformed into a multihot representation

    It also inherits torch.utils.data.Dataset so to be able to lates use the Dataloader and iterate
    """
    def __init__(self, x, y, classes, purpose="train", target_dtype=torch.LongTensor, **kwargs):
        self.__dict__.update(kwargs)
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

    def to_json(self):
        """Transform the data set into a json string representation"""
        import json
        json_string = json.dumps(self.to_dict())
        return json_string

    def to_dict(self):
        """Transform the dataset into a dictionary-of-lists representation"""
        return {"x": self.x, "y": self.y, "classes":list(self.classes.keys())}

    def __add__(self, o):
        new_classes = list(set( list(self.classes.keys()) +  list(o.classes.keys()) ))
        new_classes.sort()
        new_classes = dict(zip(new_classes, range(len(new_classes))))

        new_data = list(set(self.x + o.x))
        new_labels = [[] for _ in range(len(new_data))]

        for i, x in enumerate(self.x):
            new_labels[new_data.index(x)].extend(self.y[i])

        for i, x in enumerate(o.x):
            new_labels[new_data.index(x)].extend(o.y[i])

        new_labels = [list(set(x)) for x in new_labels]

        return MultiLabelDataset(x=new_data, y=new_labels, classes=new_classes)


class SequenceDataset(Dataset):
    """Dataset format for Sequence data."""
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

#-------------------------------------------------------------------------------------


def get_dataset(name, type, ensure_valid=False, valid_split=0.25, target_dtype=torch.FloatTensor):
    """
    General data getter
    :param name: name of the dataset in register
    :param type: MultilabelDataset or SequenceDataset defined in mlmc.data
    :param ensure_valid: if True and there's no validation data in the original data a portion of the trainset is split
    :param valid_split: the fraction of the train set to be used as validation if ensure_valid=True
    :param target_dtype: Target Tensortype of the label multihot representation. (default torch.FloatTensor)
    :return: a dictionary with keys: "train", "valid" and "test" and additional information the dataset provides (graphs, maps, classes,..)
    """
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

## Wrapper for multilabel datasets
def get_multilabel_dataset(name, type=MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float):
    return get_dataset(name, type, ensure_valid=ensure_valid, valid_split=valid_split, target_dtype=target_dtype)


## Sampler import
from .sampler import sampler, successive_sampler, class_sampler
from .data_loaders_text import RawTextDatasetTokenizer, RawTextDataset