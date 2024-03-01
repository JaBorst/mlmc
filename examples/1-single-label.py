##########################################################################
#
# Example for custom data set format for single label classification.
#
##########################################################################

import mlmc
from mlmc.data import SingleLabelDataset

# Classes have to  be a mapping from strings to ascending indices

classes = {"class1":0, "class2":1, "class3":2}

# All models handle can handle strings as input, no need for preprocessing. The input is just a list of strings.
x = [
    "Text 1 about anything",
    "Text 2 about something specific"
]

# The labels have to be a list of lists. Every datapoint corresponds to the same index in `x`. For single label data
# each element of y has to be of length one. All of these labels must have an entry in the classes dict.
y = [["class1"], ["class3"], ["class2"]]

# With data in this format you can instanciate a single label data set.
dataset = SingleLabelDataset(x=x, y=y, classes=classes)

# .. and a model. make sure in this case to set the target to 'single' to load default parameters for single label
# classification
m = mlmc.models.KimCNN(
    classes=classes,
    target="single",
    finetune="compacter",
    # device="cuda:0" # uncomment if gpu
)

# Use the fit method to train the model
m.fit(dataset, epochs=10)

# Use the predict function to predict other examples
m.predict(["A real life example", "Another one"])
