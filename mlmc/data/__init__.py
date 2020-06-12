"""
Defines dataset class and provides automated load/cache functions for some public multilabel datasets.
"""

from torch.utils.data import Dataset
import torch
from .data_loaders  import load_eurlex, load_wiki30k, load_huffpost, load_aapd, load_rcv1, \
    load_moviesummaries,load_blurbgenrecollection, load_blurbgenrecollection_de, load_20newsgroup,export,\
    load_agnews, load_dbpedia, load_ohsumed

# String Mappings
register = {
    "aapd": load_aapd,
    "rcv1": load_rcv1,
    "huffpost": load_huffpost,
    "wiki30k": load_wiki30k,
    "eurlex": load_eurlex,
    "movies_summaries": load_moviesummaries,
    "blurbgenrecollection": load_blurbgenrecollection,
    "blurbgenrecollection_de": load_blurbgenrecollection_de,
    "20newsgroup": load_20newsgroup,
    "agnews": load_agnews,
    "dbpedia": load_dbpedia,
    "ohsumed": load_ohsumed
}



class MultiLabelDataset(Dataset):
    """
    Dataset to hold text and label combinations.

    Providing a unified interface to Multilabel data, associating textual input with sets of labels. Also
    holding a mapping of labels to indices and transforming the target labelset of an instance to multi-hot representations.
    It also inherits torch.utils.data.Dataset, so it can be used in combination with torch.utils.data.Dataloader
    for fast training loops.
    """
    def __init__(self, x, y, classes, target_dtype=torch._cast_Float, one_hot=True, **kwargs):
        """
        Class constructor

        Creates an instance of MultilabelDataset.

        Args:
            x: A list of the input text
            y:  A list of corresponding label sets
            classes: A class mapping from label strings to successive indices
            target_dtype: The final cast on the label output. (Some of torch's loss functions expect other data types. This argument defines
                a function that is applied to the final output of the label tensors. (default: torch._cast_Float)
            one_hot: (default: True) if True, will transform the multilabel sets into a multi-hot tensor representation for training
                if False: will return the labelset strings as is
            **kwargs:  Any additional information that is given by named keywords will be saved as metadata

        Returns:
            A MultilabelDataset instance.

        Examples:
            ```
            x = ["This is a text about science and philosophy",
                "This is another text about science and politics"]


            y = [['science', 'philosophy'],
                ['science', 'politics']]

            classes = {
                "science": 0,
                "philosophy": 1,
                "politics": 2
            }
            dataset = mlmc.data.MultilabelDataset(x=x, y=y, classes=classes)
            dataset[0]
            ```

        """

        self.__dict__.update(kwargs)
        self.classes = classes
        self.x = x
        self.y = y
        self.one_hot = one_hot
        self.target_dtype = target_dtype

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.one_hot:
            labels = [self.classes[tag] for tag in self.y[idx]]
            labels = torch.nn.functional.one_hot(torch.LongTensor(labels), len(self.classes)).sum(0)
            return {'text': self.x[idx], 'labels': self.target_dtype(labels)}
        else:
            return {'text': self.x[idx], 'labels': self.y[idx]}

    def transform(self, fct):
        """
        Mapping functions that act on strings to every data instance

        Applies fct to every input element of the dataset. (Can be used for cleaning or preprocessing)

        Args:
            fct: A function that takes a string as input and returns the transformed string
        """
        self.x = [fct(sen) for sen in self.x]

    def to_json(self):
        """
        Transform the data set into a json string representation

        Returns:
             String representation of the dataset ( only x, y and the classes)
        """
        import json
        json_string = json.dumps(self.to_dict())
        return json_string

    def to_dict(self):
        """
        Transform the dataset into a dictionary-of-lists representation

         Returns:
              A python dictionary of the training data (only x, y and the classes)
        """
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

    def remove(self, classes):
        """
        Deleting labels from the dataset.

        Removes all occurrences of classes argument (string or list of strings) from the dataset.
        Instances with then empty labelsets will be removed completely

        Args:
            classes: A label or list of label names.
        """
        if isinstance(classes, str):
            classes = [classes]
        assert all([x in self.classes.keys() for x in classes]), "Some of the provided classes are not contained in the dataset"
        self.y = [[label for label in labelset if label not in classes] for labelset in self.y]
        emptylabelsets = [i for i, x in enumerate(self.y) if x == []]
        self.x = [ x for i,x in enumerate(self.x) if i not in emptylabelsets]
        self.y = [ x for i,x in enumerate(self.y) if i not in emptylabelsets]

    def map(self, map: dict):
        """
        Transforming label names

        Apply label mappings to every data instance. Maps every label string in the dataset according to 'map'.

        Args:
            map: Dictionary of map from current label string to new label string
        """
        if any([x not in map.keys() for x in self.classes.keys()]):
            print("Some classes are not present in the map. The will be returned as is.")
        self.classes = {map.get(k,k):v for k,v in self.classes.items()}
        self.y = [[map.get(l,l) for l in labelset] for labelset in self.y]

    def reduce(self, subset: dict):
        """
        Reduces the dataset to a subset of the classes.

        The resulting dataset will only contain instances with at least one label that appears in the subset argument.
        The subset must also provide a new mapping from the new label names to indices.
        All labels not in subset will be removed. Instances with an empty label set will be removed.

        Args:
            subset: A mapping of classes to indices
        """
        assert all([x in self.classes.keys() for x in subset.keys()]), "Subset contains classes not present in dataset"
        ind = [i for i, labelset in enumerate(self.y) if any([l in subset.keys() for l in labelset])]
        self.x = [self.x[i] for i in ind]
        self.y = [[x for x in self.y[i] if x in subset.keys()] for i in ind]
        self.classes = subset

    def count(self, label):
        """
        Count the occurrences of all labels in 'label' in the dataset.

        Args:
            label: Label name or list of label names
        Returns:
             Dictionary of label name and frequency in the dataset.
        """
        if isinstance(label, list):
            result = {   l: sum([l in s for s in self.y])
                for l in label
            }
        else:
            result = {label: sum([label in s for s in self.y])}
        return result

    def density(self):
        """
        Returns the average label set size per instance.

        Returns: The average labelset size per instance
        """
        return sum([len(x) for x in self.y])/len(self.y)


class SingleLabelDataset(MultiLabelDataset):
    def __init__(self, *args, **kwargs):
        super(SingleLabelDataset, self).__init__(*args, **kwargs)
        assert all([len(x)==1 for x in self.y]), "This is not a single label dataset. Some labels contain multiple labels."

    def __getitem__(self, idx):
        return {'text': self.x[idx], 'labels': torch.tensor(self.classes[self.y[idx][0]])}



#-------------------------------------------------------------------------------------


def get_dataset(name, type, ensure_valid=False, valid_split=0.25, target_dtype=torch.FloatTensor):
    """
    General dataset getter for datasets in provided by the package.

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
def get_multilabel_dataset(name, target_dtype=torch._cast_Float):
    """
    Load multilabel training data if available.

    This is the default wrapper function for retrieving multilabel datasets.

    :param name: See: mlmc.data.register.keys()
    :param target_dtype: The target_dtype of the labeldata in training. See MultilabelDataset
    :return:
    """
    return get_dataset(name, type=MultiLabelDataset, ensure_valid=False, target_dtype=torch._cast_Float)


## Wrapper for singlelabel datasets
def get_singlelabel_dataset(name, target_dtype=torch._cast_Float):
    """
    Load singlelabel training data if available.

    This is the default wrapper function for retrieving SingleLabelDataset datasets.
    This is a special case of MultilabelDataset.

    :param name: See: mlmc.data.register.keys()
    :param target_dtype: The target_dtype of the labeldata in training. See MultilabelDataset
    :return:
    """
    return get_dataset(name, type=SingleLabelDataset, ensure_valid=False, target_dtype=torch._cast_Float)




## Sampler import
from .sampler import sampler, successive_sampler, class_sampler, validation_split
from .data_loaders_text import RawTextDatasetTokenizer, RawTextDataset,RawTextDatasetTensor
