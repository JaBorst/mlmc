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
dictionary. ``data["train"]`` and ``data["test"]`` are of type :py:class`MultiLabelDataset <mlmc.data.MultiLabelDataset>`. If there is
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

    dataset = mlmc.data.MultiLabelDataset(x=x, y=y, classes=classes)

If your know your data to be only single labelled you can use :py:mod:`mlmc.data.SingleLabelDataset`.
The process and data format is the same as for MultiLabelDataset, but with every labelset in y must contain only one label. ::

    y = [['philosophy'],
        ['science']]

    all([len(labelset) == 1 for labelset in y]) # must be True!

This will be checked at initialization.
The SingleLabelDataset provides the classification models with specialized output format for using other losses.
The default when using a SingleLabelDataset as input for the fit function is `torch.nn.CrossEntropyLoss`.

Classifier
------------

Train a classifier
___________________

Multilabel Classification
--------------------------
When training a classifier you will have to load your data first. The instantiation of a model will depend of the
classes mapping present in the data. A list of available models can be found under :py:mod:`mlmc.models`.
The device argument determines on which device the model will be trained. Use "cpu" to train on CPU and "cuda:" for
training on the GPU. GPU is zero-indexed, so if there is one GPU it will be "cuda:0"::

    import mlmc

    data = mlmc.data.get_multilabel_dataset("20newsgroup")

    tc = mlmc.models.KimCNN(classes=data["train"].classes, device="cuda:1")
    history = tc.fit(train=data["train"],  epochs=15, batch_size=32)

    tc.evaluate(data["test"])


    labels = tc.predict("Predict the labels for this sentence", return_scores=True)

If return_scores=True the confidence for the predicted labels is returned along with the labels.

Models can be saved loaded with :py:meth:`mlmc.save() <mlmc.save_and_load.save>` and  :py:meth:`mlmc.load() <mlmc.save_and_load.load>`::

    mlmc.save(tc, "test.pt",only_inference=False)
    tc = mlmc.load( "test.pt",only_inference=False)


Single label Classification
----------------------------
The same workflow can be used with single label classification datasets, when using the corresponding functions and keyword arguments: ::


   import mlmc

    data = mlmc.data.get_singlelabel_dataset("agnews")

    tc = mlmc.models.KimCNN(classes=data["train"].classes, device="cuda:1", target="single")
    history = tc.fit(train=data["train"],  epochs=15, batch_size=32)

    tc.evaluate(data["test"])


    labels = tc.predict("Predict the labels for this sentence", return_scores=True)


Other Functionality
---------------------
Embed sentences
________________

For the purpose of calculating embeddings outside if the neural network architectures, there is a class
:py:class:`Embedder <mlmc.representation.embedder.Embedder`. You can load any of the language models from huggingface or glove and embed text.::

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


