Module mlmc.data.sampler
========================
A collection of sampling methods for the MultilabelDataset object

.

Functions
---------

    
`class_sampler(data, classes, samples_size)`
:   Create a subset of the data containing only classes in the classes argument and subsampling each class
    roughly to the number given in sample_size.
    
    Args:
        data: MultilabelDataset to sample from
        classes: subset of classes
        samples_size: Number of examples for each class (roughly)
    Returns:
        A dataset with roughly sample_size number of instances per class in classes.

    
`sampler(dataset, fraction=None, absolute=None)`
:   Sample a Random subsample of fixed size or fixed fraction of a dataset (i.i.d. sample).
    Args:
        dataset: A instance of mlmc.data.MultilabelDataset
        fraction: The fraction of the data that should be returned (0<fraction<1)
        absolute: The absolute size of the sampled dataset
    Returns:
         A randomly subsampled MultilabelDataset of the desired size.

    
`successive_sampler(dataset, classes, separate_dataset, reindex_classes=True)`
:   Return an iterable of datasets sampled from dataset.
    
    Args:
         dataset: The input dataset (MultilabelDataset)
         classes: A classes mapping
         separate_dataset: The number of datasets to generate
         reindex_classes: If True, classes in the subsampled datasets are reindexed to 1:len(classes)
    Returns:
        A list of MultilabelDatasets

    
`validation_split(dataset, fraction=None, absolute=None)`
:   Split a dataset into two separate splits (validation split)
    
    Args:
        dataset: data set to split
        fraction: The fraction of the data that should be returned (0<fraction<1)
        absolute: The absolute size of the sampled dataset
    Returns:
         A tuple of randomly subsampled MultilabelDatasets of the desired size.