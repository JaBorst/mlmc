##########################################################################
#
# Example for custom data set format for multi output single label classification.
#
##########################################################################


import mlmc

# The name multi output single label classification refers to the fact that the models has multiple outputs and
# each output can have a different set of classes. But per output there will be exactly one label.

# So classes is now a list of class mappings. Every output of the model can have its own class dictionary
# (length and names can vary and every mapping should start at zero.
import mlmc.data.dataset_classes

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
# to the number of outputs (the length of the `classes`. In each element each label per output channel has to be a list
# of length one.
y = [
    [["class1"], ["class3"], ["class6"]],
    [["class3"], ["class1"], ["class4"]],
]

# Now we can instantiate a multi output single label dataset.
d = mlmc.data.dataset_classes.MultiOutputSingleLabelDataset(x=x, y=y, classes=classes)

# The target is single. Models with multiple outputs are prefixed with `Mo-`.
m = mlmc.models.MoKimCNN(
    classes=classes,
    target="single",
    finetune=True,
    # device="cuda:0" # uncomment if gpu
)

# Train
m.fit(d, epochs=1)

# Evaluate
m.evaluate(d)

# Use the predict function to predict other examples
m.predict(d.x)
