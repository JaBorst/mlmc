
![](mlmc-logo.png)
---

**M**ulti**l**abel **M**ulti**c**lass

Python package for application of neural networks for single label and multi label text classification.
Specialized for NLI-based and Siamese zero-shot capabilities and finetuning scenarios.


Before installing, make sure the right pytorch package is installed
for your cuda setup. Download the right version from
[https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)

Install with
```
pip install git+https://git.informatik.uni-leipzig.de/asv-ml/mlmc
```
## Workflow

The basic usage of the package is as follows:

```
import mlmc

#Load data
text = ["Scientists found out that breathing keeps you alive!",
        "The Eagles won the Superbowl in 2018"]
labels = [["Science"], ["Sports"]]

classes = {"Science":0, "Sports":1}

data = {"train": mlmc.data.SingleLabelDataset(x=text, y=label, classes=classes),
        "test": mlmc.data.SingleLabelDataset(x=text, y=label, classes=classes)}

#Create model
model = mlmc.models.KimCNN(classes=classes, target="single")

#Train model
history = model.fit(train=data["train"], epochs=10, batch_size=50)

#Evaluate model
results = model.evaluate(data["test"])
```


## Documentation
For more detailed explanations, see the [Documentation](https://mlmc-docs.readthedocs.io/en/latest/contents.html) on readthedocs.org.

## Results

The following results have been achieved by training a KimCNN using a Transformer-Encoder as input. All results have been averaged over 3 runs. Single-label results are measured in Test Accuracy (%).


## Single-label zero-shot classification
