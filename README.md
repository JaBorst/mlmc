
![](mlmc-logo.png)
---

**M**ulti**l**abel **M**ulti**c**lass

Python package for application of neural networks for single label and multi label text classification.
Contains capabilities for zero and few-shot scenarios.


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

### Single-label

| Method      | DBpedia     | AG News     | 20NEWS
| ----------- | ----------- | ----------- | -----------
| BERT-FB     | 99.19       | 92.97       | 84.25
| BERT-Fit    | **99.26**   | 94.01       | 84.69
| RoBERTa-FB  | 99.04       | 92.49       | **85.81**
| RoBERTa-Fit | 99.21       | **94.84**   | 85.59

### Multi-label

|              | RCV1        | Ohsumed    |
|--------------|-------------|------------|
|<table><tr><th>Method</th></tr><tr><td>BERT-FB</td></tr><td>BERT-FiT</td><tr></tr><td>RoBERTa-FB</td><tr></tr><td>RoBERTa-FiT</td><tr></tr></table>|<table><tr><th>Precision</th><th>Recall</th><th>F1</th></tr><tr><td>**87.88**</td><td>77.97</td><td>82.63</td></tr><tr><td>86.12</td><td>86.39</td><td>86.26</td></tr><tr><td>87.62</td><td>81.34</td><td>84.36</td></tr><tr><td>86.93</td><td>**87.3**</td><td>**87.11**</td></tr></table>|<table><tr><th>Precision</th><th>Recall</th><th>F1</th></tr><tr><td>71.25</td><td>61.21</td><td>65.84</td></tr><tr><td>71.74</td><td>**69.75**</td><td>70.72</td></tr><tr><td>69.67</td><td>64.08</td><td>66.76</td></tr><tr><td>**73.78**</td><td>69.26</td><td>**71.74**</td></tr></table>|

<br>

|              | BGC_EN      | BGC_DE     |
|--------------|-------------|------------|
|<table><tr><th>Method</th></tr><tr><td>BERT-FB</td></tr><td>BERT-FiT</td><tr></tr><td>RoBERTa-FB</td><tr></tr><td>RoBERTa-FiT</td><tr></tr></table>|<table><tr><th>Precision</th><th>Recall</th><th>F1</th></tr><tr><td>76.87</td><td>70.18</td><td>73.37</td></tr><tr><td>77.54</td><td>78.64</td><td>78.09</td></tr><tr><td>73.72</td><td>71.35</td><td>72.52</td></tr><tr><td>**78.41**</td><td>**79.36**</td><td>**78.88**</td></tr></table>|<table><tr><th>Precision</th><th>Recall</th><th>F1</th></tr><tr><td>69.5</td><td>50.93</td><td>58.78</td></tr><tr><td>68.5</td><td>**63.47**</td><td>**65.87**</td></tr><tr><td>**71.73**</td><td>48.41</td><td>57.81</td></tr><tr><td>67.89</td><td>62.9</td><td>65.3</td></tr></table>|
