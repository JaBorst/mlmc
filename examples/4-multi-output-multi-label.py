##########################################################################
#
# Example for custom data set format for multi output multi label classification.
#
##########################################################################


import mlmc

# The name multi output multi label classification refers to the fact that the models has multiple outputs and
# each output can have a different set of classes. Per output there can be varying number of labels.

# So classes is now a list of class mappings. Every output of the model can have its own class dictionary
# (length and names can vary and every mapping should start at zero.
import mlmc.data.datasets

classes = [
    {"class1":0, "class2":1, "class3":2},
    {"class1":0, "class3":1},
    {"class4":0, "class5":1, "class6":2},
]

# Input is still a list of strings
x = [
    "Text 1 about anything",
    "Text 2 about something specific"
]

# Every element in y corresponds to the element in x with the same index. The length of each element should now be equal
# to the number of outputs (the length of the `classes`. In each element each label per output channel contains the
# list of labels for that output. The length of lists of labels can vary.
y = [
    [["class1", "class3"], ["class3"], ["class4", "class5","class6"]],
    [["class3"], ["class1"], ["class4","class5"]],
]

# Now we can instantiate a multi output multi label dataset.
d = mlmc.data.datasets.MultiOutputMultiLabelDataset(x=x, y=y, classes=classes)

# The target is multi. Models with multiple outputs are prefixed with `Mo-`.
m = mlmc.models.MoKimCNN(classes=classes,target="multi",finetune=True)

# you can fit the data
m.fit(d, epochs=1)

# Use the predict function to predict other examples
m.predict(d.x)
