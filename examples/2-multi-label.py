##########################################################################
#
# Example for custom data set format for multi label classification.
#
##########################################################################

import mlmc

# Classes have to  be a mapping from strings to ascending indices
import mlmc.data.dataset_classes

classes = {"class1":0, "class2":1, "class3":2}

# All models handle can handle strings as input, no need for preprocessing. The input is just a list of strings.
x = [
    "Text 1 about anything",
    "Text 2 about something specific"
]

# The labels have to be a list of lists. Every datapoint corresponds to the same index in `x`. For multi label data
# the elements of y can vary in length. All of these labels must have an entry in the classes dict.
y = [["class1", "class2"], ["class3"], ["class1", "class3"]]

# With data in this format you can instanciate a multi label data set.
dataset = mlmc.data.dataset_classes.MultiLabelDataset(x=x, y=y, classes=classes)

# .. and a model. make sure in this case to set the target to 'multi' to load default parameters for multi label
# classification.
m = mlmc.models.KimCNN(
    classes=classes,
    target="multi",
    finetune=True,
    # device="cuda:0" # uncomment if gpu
)

# Use the fit method to train the model
m.fit(dataset, epochs=10)

# Use the predict function to predict other examples
m.predict(["A real life example", "Another one"])
