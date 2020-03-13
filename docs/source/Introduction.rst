Introduction
============

**M**\ ulti **l**\ abel **M**\ ulti **c**\ lass


A python package for training, and application of neural networks for multilabel textclassification, implementing
SOTA network architectures.

Installation
---------------

The project is currently only hosted on gitlab, but can be directly installed with pip.
There are optional dependencies that are not part of the automatic installation process because of driver dependencies.

So before installing you have to install the pytorch version fitting your setup (CUDA driver version, GPU or cpu only).
See: https://download.pytorch.org/whl/torch_stable.html for a list of options and install the right version.

Optional:
    There are currently models that require the `pytorch_geometric <https://github.com/rusty1s/pytorch_geometric>`_.
    This dependency is optional, it just reminds you if you try to import models that rely on geometric layers.
    To install, follow the instructions on their github page: https://github.com/rusty1s/pytorch_geometric

After that you can install `mlmc <https://git.informatik.uni-leipzig.de/asv-ml/mlmc>`_  by sample::

    pip install git+https://git.informatik.uni-leipzig.de/asv-ml/mlmc

Workflow
--------------

The workflow follows a single instantiation process.
A model can be created and then provides methods to train, evaluate and predict text instances.

A typical usage would look like: sample::

    import mlmc

    # Dataset creation ( see section mlmc.data )
    data = ... # Data for training
    validation_data = ... # Data for validation

    # Model instantiation (with training on a GPU)
    tc = mlmc.models.KimCNN(classes=data.classes, "cuda:0")

    # Training
    history = tc.fit(train=data,  epochs=15, batch_size=32)

    # Prediction
    prediction = tc.predict("This an example sentence  I want to classify")

    # Evaluation
    metrics = tc.evaluate(validation_data)

    # Save and load functions
    mlmc.save(tc,"test.pt")
    tc = mlmc.load("test.pt")



