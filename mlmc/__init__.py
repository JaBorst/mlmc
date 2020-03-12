"""
**M**ulti**l**abel **M**ulti**c**lass

A python package for training, and application of neural networks for multilabel textclassification, implementing
SOTA network architectures.

# Workflow

The workflow follows a single instantiation process.
A model can be created and then provides methods to train, evaluate and predict text instances.

A typical usage would look like:

```python
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
```

## Creating a Dataset
The dataset creation is kept as simple as possible. You need to specify two lists of input and output data, where
``x[i]`` is the text input as a string and ``y[i]`` the corresponding label set as a python list.

An example of two sentences:

```python
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
```


"""

import mlmc.data
import mlmc.models
import mlmc.graph
import mlmc.metrics
import mlmc.representation
# Save and load models for inference without the language models representaiton to save disc space
from .save_and_load import save, load