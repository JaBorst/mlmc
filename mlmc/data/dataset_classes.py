import torch
from torch.utils.data import Dataset
from warnings import warn
from mlmc.data.sampler.random import subset


class MultiLabelDataset(Dataset):
    """
    Dataset to hold text and label combinations.

    Providing a unified interface to Multilabel data, associating textual input with sets of labels. Also
    holding a mapping of labels to indices and transforming the target labelset of an instance to multi-hot representations.
    It also inherits torch.utils.data.Dataset, so it can be used in combination with torch.utils.data.Dataloader
    for fast training loops.
    """

    def __init__(self, x, y, classes, hypothesis=None, target_dtype=torch._cast_Float, one_hot=True, augmenter=None, **kwargs):
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
        self.classes = {k:i for i,k in enumerate(classes)} if classes is not None else None
        self.x = x
        self.y = y
        self.one_hot = one_hot
        self.target_dtype = target_dtype
        self.augmenter = augmenter
        self.hypothesis = hypothesis

    def __len__(self):
        """
        Returns the length of the dataset. The length is determined by the size
        of the list containing the input text.

        :return: Length of the dataset
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Retrieves a single entry from the dataset.

        :param idx: Index of the entry
        :return: Dictionary containing the text and labels of the entry
        """
        if self.hypothesis is None:
            if self.one_hot:
                labels = [self.classes[tag] for tag in self.y[idx]]
                labels = torch.nn.functional.one_hot(torch.LongTensor(labels), len(self.classes)).sum(0)
                return {'text': self._augment(self.x[idx]),
                        'labels': self.target_dtype(labels)}
            else:
                return {'text': self._augment(self.x[idx]),
                        'labels': torch.tensor(self.classes[self.y[idx][0]])}
        else:
            if self.one_hot:
                labels = [self.classes[tag] for tag in self.y[idx]]
                labels = torch.nn.functional.one_hot(torch.LongTensor(labels), len(self.classes)).sum(0)
                return {'text': self._augment(self.x[idx]),
                        "hypothesis": self._augment(self.hypothesis[idx]),
                        'labels': self.target_dtype(labels)}
            else:
                return {'text': self._augment(self.x[idx]),
                        "hypothesis": self._augment(self.hypothesis[idx]),
                        'labels': torch.tensor(self.classes[self.y[idx][0]])}

    def set_augmenter(self, fct ):
        """
        Use this function to augment on the fly
        """
        self.augmenter = fct

    def _augment(self, x):
        return self.augmenter(x) if self.augmenter is not None else x

    def generate(self, augmenter, n=10):
        """
        Use this function to generate a number of exmamples at once
        """
        if n > 0 :
            self.x = self.x + sum((augmenter.generate(x,n) for x in self.x),[])
            if self.hypothesis is not None:
                self.hypothesis = self.hypothesis + sum((augmenter.generate(x,n) for x in self.hypothesis),[])
            self.y = self.y + sum(([y]*n for y in self.y), [])

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
        return {"x": self.x, "y": self.y, "classes": list(self.classes.keys())}

    def __add__(self, o):
        """
        Merges dataset with another dataset.

        :param o: Another dataset
        :return: MultiLabelDataset containing x, y and classes of both datasets
        """
        from collections import OrderedDict
        if OrderedDict(self.classes) == OrderedDict(o.classes):
            new_classes = self.classes
        else:
            warn("Adding two datasets with unequal class dictionaries alters the sorting order to create a new union mapping")
            new_classes = list(set(list(self.classes.keys()) + list(o.classes.keys())))
            new_classes.sort()
            new_classes = dict(zip(new_classes, range(len(new_classes))))

        if self.hypothesis is None:
            new_data = dict(zip(self.x, self.y))
            for x,y in zip(o.x, o.y):
                if x not in new_data: new_data[x] = []
                new_data[x].extend(y)
            new_data = {k:list(set(v)) for k,v in new_data.items()}
            try:
                return SingleLabelDataset(x=list(new_data.keys()), y=list(new_data.values()), classes=new_classes)
            except AssertionError:
                return MultiLabelDataset(x=list(new_data.keys()), y=list(new_data.values()), classes=new_classes)
        else:
            new_data = dict(zip(list(zip(self.x, self.hypothesis)), self.y))
            for x, y, h in zip(o.x, o.y, o.hypothesis):
                if (x, h) not in new_data:
                    new_data[(x, h)] = []
                new_data[(x, h)].extend(y)
            new_data = {k: list(set(v)) for k, v in new_data.items()}
            return type(self)(x=list(x[0] for x in new_data.keys()),
                                     hypothesis=list(x[1] for x in new_data.keys()), y=list(new_data.values()),
                                     classes=new_classes)

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
        assert all([x in self.classes.keys() for x in
                    classes]), "Some of the provided classes are not contained in the dataset"
        self.y = [[label for label in labelset if label not in classes] for labelset in self.y]
        nonemptylabelsets = [i for i, x in enumerate(self.y) if x != []]
        new_x = [self.x[i] for i in nonemptylabelsets]
        new_y = [self.y[i] for i in nonemptylabelsets]
        self.x = new_x
        self.y = new_y

        new_classes = [x for x in self.classes.keys() if x not in classes]
        self.classes = dict(zip(new_classes, range(len(new_classes))))

    def map(self, map: dict):
        """
        Transforming label names

        Apply label mappings to every data instance. Maps every label string in the dataset according to 'map'.

        Args:
            map: Dictionary of map from current label string to new label string
        """
        if any([x not in map.keys() for x in self.classes.keys()]):
            print("Some classes are not present in the map. They will be returned as is.")
        self.classes = {map.get(k, k): v for k, v in self.classes.items()}
        self.y = [[map.get(l, l) for l in labelset] for labelset in self.y]

    def reduce(self, subset):
        """
        Reduces the dataset to a subset of the classes.

        The resulting dataset will only contain instances with at least one label that appears in the subset argument.
        The subset can also provide a new mapping from the new label names to indices (dict).
        All labels not in subset will be removed. Instances with an empty label set will be removed.

        Args:
            subset: A mapping of classes to indices
        """

        if isinstance(subset, str): subset = [subset]

        if isinstance(subset, dict):
            subset_dict = subset
            subset_list = list(subset.keys())

        elif isinstance(subset, list):
            subset_dict = dict(zip(subset, range(len(subset))))
            subset_list = subset
        else:
            raise ValueError

        remove_classes = [x for x in self.classes.keys() if x not in subset_list]
        self.remove(remove_classes)
        self.classes = subset_dict

    def count(self, label=None):
        """
        Count the occurrences of all labels in 'label' in the dataset.

        Args:
            label: Label name or list of label names
        Returns:
             Dictionary of label name and frequency in the dataset.
        """
        if label is None:
            label = list(self.classes.keys())
        if isinstance(label, list):
            result = {l: sum([l in s for s in self.y])
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
        return sum([len(x) for x in self.y]) / len(self.y)

    @staticmethod
    def from_pandas(df, x, y, sep=" ", classes=None):
        y = y if isinstance(y, str) else y[0]
        label = [l for l in df[y].to_list()]
        if classes is None:
            cls_list = list(set(sum(label,[])))
            classes =  {cls:i for i, cls in enumerate(sorted(cls_list))}
        return MultiLabelDataset(
            x=df[x].applymap(str).agg(sep.join, axis=1).to_list(),
            y=label,
            classes=classes
        )

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame.from_dict({"x": self.x, "y": self.y})

    def subset(self, index):
        return subset(self, index)

class SingleLabelDataset(MultiLabelDataset):
    def __init__(self, *args, **kwargs):
        """
        Class constructor. Creates an instance of SingleLabelDataset.

        :param classes: A class mapping from label strings to successive indices
        :param x: A list of the input text
        :param y: A list of corresponding label sets
        :param target_dtype: The final cast on the label output. (Some of torch's loss functions expect other data types. This argument defines
                a function that is applied to the final output of the label tensors. (default: torch._cast_Float)
        :param kwargs: Any additional information that is given by named keywords will be saved as metadata

        Example:
            ```
            x = ["This is a text about science",
                "This is another text about philosophy"]

            y = [['science'],
                ['politics']]

            classes = {
                "science": 0,
                "philosophy": 1,
            }
            dataset = mlmc.data.SingleLabelDataset(x=x, y=y, classes=classes)
            dataset[0]
            ```
        """
        super(SingleLabelDataset, self).__init__(*args, **kwargs)
        assert all(
            [len(x) == 1 for x in self.y]), "This is not a single label dataset. Some labels contain multiple labels."
        self.one_hot=False


class EntailmentDataset(SingleLabelDataset):
    def __init__(self, *args, **kwargs):
        super(EntailmentDataset, self).__init__(*args, **kwargs)
        assert self.hypothesis is not None

class ABCDataset(SingleLabelDataset):
    def __init__(self, *args, **kwargs):
        super(ABCDataset, self).__init__(*args, **kwargs)
        assert self.hypothesis is not None

class PredictionDataset(MultiLabelDataset):
    def __init__(self, x, hypothesis=None, **kwargs):
        super().__init__(x, y=None, hypothesis=hypothesis, classes=None, target_dtype=torch._cast_Float, one_hot=True, **kwargs)
    def __getitem__(self, idx):
        if self.hypothesis is None:
                return {'text': self._augment(self.x[idx])}
        else:
            return {'text': self._augment(self.x[idx]),
                    "hypothesis": self._augment(self.hypothesis[idx]),}


def is_multilabel(x):
    """
    Checks if input is a multilabel dataset.

    :param x: A dataset
    :return: True if multilabel, else False.
    """
    return type(x) in  (MultiLabelDataset)
