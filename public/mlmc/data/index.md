Module mlmc.data
================
Defines dataset class and provides automated load/cache functions for some public multilabel datasets.

Sub-modules
-----------
* mlmc.data.data_loaders
* mlmc.data.data_loaders_text
* mlmc.data.sampler
* mlmc.data.transformer

Functions
---------

    
`get_dataset(name, type, ensure_valid=False, valid_split=0.25, target_dtype=torch.FloatTensor)`
:   General dataset getter for datasets in provided by the package.
    
    :param name: name of the dataset in register
    :param type: MultilabelDataset or SequenceDataset defined in mlmc.data
    :param ensure_valid: if True and there's no validation data in the original data a portion of the trainset is split
    :param valid_split: the fraction of the train set to be used as validation if ensure_valid=True
    :param target_dtype: Target Tensortype of the label multihot representation. (default torch.FloatTensor)
    :return: a dictionary with keys: "train", "valid" and "test" and additional information the dataset provides (graphs, maps, classes,..)

    
`get_multilabel_dataset(name, target_dtype=<built-in method _cast_Float of type object>)`
:   Load multilabel training data if available.
    
    This is the default wrapper function for retrieving multilabel datasets.
    
    :param name: See: mlmc.data.register.keys()
    :param target_dtype: The target_dtype of the labeldata in training. See MultilabelDataset
    :return:

Classes
-------

`MultiLabelDataset(x, y, classes, target_dtype=<built-in method _cast_Float of type object>, one_hot=True, **kwargs)`
:   Dataset to hold text and label combinations.
    
    Providing a unified interface to Multilabel data, associating textual input with sets of labels. Also
    holding a mapping of labels to indices and transforming the target labelset of an instance to multi-hot representations.
    It also inherits torch.utils.data.Dataset, so it can be used in combination with torch.utils.data.Dataloader
    for fast training loops.
    
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

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset

    ### Methods

    `count(self, label)`
    :   Count the occurrences of all labels in 'label' in the dataset.
        
        Args:
            label: Label name or list of label names
        Returns:
             Dictionary of label name and frequency in the dataset.

    `density(self)`
    :   Returns the average label set size per instance.
        
        Returns: The average labelset size per instance

    `map(self, map)`
    :   Transforming label names
        
        Apply label mappings to every data instance. Maps every label string in the dataset according to 'map'.
        
        Args:
            map: Dictionary of map from current label string to new label string

    `reduce(self, subset)`
    :   Reduces the dataset to a subset of the classes.
        
        The resulting dataset will only contain instances with at least one label that appears in the subset argument.
        The subset must also provide a new mapping from the new label names to indices.
        All labels not in subset will be removed. Instances with an empty label set will be removed.
        
        Args:
            subset: A mapping of classes to indices

    `remove(self, classes)`
    :   Deleting labels from the dataset.
        
        Removes all occurrences of classes argument (string or list of strings) from the dataset.
        Instances with then empty labelsets will be removed completely
        
        Args:
            classes: A label or list of label names.

    `to_dict(self)`
    :   Transform the dataset into a dictionary-of-lists representation
        
         Returns:
              A python dictionary of the training data (only x, y and the classes)

    `to_json(self)`
    :   Transform the data set into a json string representation
        
        Returns:
             String representation of the dataset ( only x, y and the classes)

    `transform(self, fct)`
    :   Mapping functions that act on strings to every data instance
        
        Applies fct to every input element of the dataset. (Can be used for cleaning or preprocessing)
        
        Args:
            fct: A function that takes a string as input and returns the transformed string