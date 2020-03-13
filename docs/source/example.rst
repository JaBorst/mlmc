Examples
=========

This section provides some code examples for typical workflows.

Data
-----

Load Datasets
______________
The package provides methods for a few public datasets. It wraps around downloading, formatting and caching the dataset.
The list of datasets included is probably going to be expanded over time.

For now to see a list of available datasets see::

    mlmc.data.register

Then a dataset can be loaded via::

    data = mlmc.data.get_multilabel_dataset("aapd")

When invoked the first time the function downloads and preprocesses the data from an online source and
saves them to a cache. It returns a dictionary containing at least the keys "train", "test" and "classes".
If theres additional information in the dataset, like graph hierarchies, maps or meta information, it will appear in the
dictionary. ``data["train"]`` and ``data["test"]`` are of type ``mlmc.data.MultilabelDataset``. If there is
an official validation split, it can be found under ``data["valid]"``


Create a Custom Dataset
________________________

If you want to use data that is not part of the package as of yet you can create a custom dataset.

The dataset creation is kept as simple as possible. You need to specify two lists of input and output data, where
``x[i]`` is the text input as a string and ``y[i]`` the corresponding label set as a python list.

An example of two sentences::

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



Classifier
------------

Train a classifier
___________________

When training a classifier you will have to load your data first. The instantiation of a model will depend of the
classes mapping present in the data. A list of available models can be found under ``mlmc.models``::

    import mlmc

    data = mlmc.data.get_multilabel_dataset("20newsgroup")

    tc = mlmc.models.KimCNN(classes=data["train"].classes, device="cuda:1")
    history = tc.fit(train=data["train"],  epochs=15, batch_size=32)

    tc.evaluate(data["test"])


    labels = tc.predict("Predict the labels for this sentence", return_scores=True)



Models can be saved loaded with ``mlmc.save`` and ``mlmc.load``::

    mlmc.save(tc, "test.pt",only_inference=False)
    tc = mlmc.load( "test.pt",only_inference=False)



Other Functionality
---------------------
Embed sentences
________________

For the purpose of calculating embeddings outside if the neural network architectures, there is a class
``mlmc.representation.Embedder``. You can load any of the language models from huggingface or glove and embed text.::

    from mlmc.representation import Embedder
    e = Embedder("bert-base-uncased")
    embeddings = e.embed(["An example sentence"])

You can also pad (or cut) all the elements of the input list to the same length by setting the ``pad=...`` argument.::

    embeddings = e.embed(["An example sentence"], pad=100)

It is also possible to embed large amount of text data by going through the data batchwise.::

    e.embed_batch([...large_list_of_strings...])

I also provides a iterator interface to the batchwise embedding method, so you can process the results of batch
embeddings while iterating, like writing to disk.::

    e.embed_batch_iterator([...large_list_of_strings...])


