from .data_loaders  import load_eurlex,load_amazon12k, load_wiki30k, load_huffpost, load_appd, load_rcv1

# register = {fct.split("load_")[1]:dl.__dict__[fct] for fct in dir(dl) if fct.startswith("load_")}
register = {
    "appd": load_appd,
    "rcv1": load_rcv1,
    "huffpost": load_huffpost,
    "wiki30k": load_wiki30k,
    "amazon12k": load_wiki30k,
    "eurlex": load_eurlex
}

from torch.utils.data import Dataset, DataLoader
import torch
class MultiLabelDataset(Dataset):
    def __init__(self, x, y, classes, purpose="train"):
        self.classes = classes
        self.purpose = purpose
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        labels = [self.classes[tag] for tag in self.y[idx]]
        return {'text': self.x[idx], 'labels': torch.nn.functional.one_hot(torch.LongTensor(labels), len(self.classes)).sum(0)}

def get_dataset_only(name, ensure_valid=False, valid_split=0.25):
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

        datasets = {
            split: MultiLabelDataset(x=data[split][0],
                                     y=data[split][1],
                                     classes=classes,
                                     purpose=split) for split in data.keys()
        }
        datasets["classes"]=classes

        return datasets
# mlmc.data.get_dataset_only("appd")


