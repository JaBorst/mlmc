##########################################################################
#
# Example for custom data set format for single label classification.
# This example demonstrates how to create a custom dataset from a list of strings and train a model on it.
# It covers the creation of class labels, dataset preparation, model instantiation, training, and prediction.
#
##########################################################################

import mlmc  # Import mlmc library for machine learning tasks.
from mlmc.data import SingleLabelDataset  # Import the SingleLabelDataset class for handling single label classification datasets.

# Define a set of class names for the dataset.
# Here, 'Sport' and 'Sci/Tech' are the two classes our model will learn to classify.
classes = {"Sport", "Sci/Tech"}

# Define the dataset as a list of strings.
# These strings are sample texts that the model will use for training.
# Each string represents an example that belongs to one of the defined classes above.
x = [
    "The Kansas City Chiefs won Superbowl LIV.",
    "The Kansas City Chiefs won Superbowl LVII.",
    "Running back Willis McGahee has asked the Buffalo Bills to trade him",
    "Ibrahim Wins Egypt's First Gold in 56 Years",
    "Freshman Chad Henne, not Matt Gutierrez, leads No. 8-ranked Michigan to a 43-10 triumph over Miami",
    "Giants Look for Way to Stop Owens, Eagles (AP)",
    "NBA, NBRA Approve Collective Bargaining Agreement",
    "Microsoft is stopping people getting hold of a key security update via net-based file- sharing systems.",
    "Firefox ignites demand for alternative browser",
    "Broadband hits new high in the UK",
    "Linksys Teams With Verizon, Best Buy, Intel",
    "Fastest 3G Wireless Network",
    "Worldwide Technology Consumers To Lose in DVD Standards War \n Blu-ray and HD DVD"
]

# Define the labels for each data point in the dataset.
# The labels are structured as a list of lists, where each inner list contains one class label corresponding to the data point in `x`.
# These labels must match the classes defined in the `classes` set.
y = [["Sport"], ["Sport"], ["Sport"], ["Sport"], ["Sport"], ["Sport"], ["Sport"], ["Sci/Tech"], ["Sci/Tech"], ["Sci/Tech"], ["Sci/Tech"], ["Sci/Tech"], ["Sci/Tech"]]

# Instantiate a SingleLabelDataset with the data and labels.
# This dataset object is used to organize our data and labels in a way that is compatible with the mlmc library for training and evaluation.
dataset = SingleLabelDataset(x=x, y=y, classes=classes)

# Instantiate a model for single label classification.
# Here, we use the KimCNN model, a convolutional neural network designed for text classification.
# The 'target' parameter is set to 'single' indicating single label classification, and 'finetune' is set to 'compacter' for model fine-tuning options.
# Uncomment the 'device' parameter to use a GPU for training, if available.
m = mlmc.models.KimCNN(
    classes=classes,
    target="single",
    finetune="all",
    # device="cuda:0"  # Uncomment this line to train the model on a GPU.
)

# Split the dataset into training and validation sets.
# Here, the `validation_split` function splits the dataset in half for training and validation purposes.
train, valid = mlmc.data.validation_split(dataset, fraction=0.6)

cb = [mlmc.callbacks.CallbackSaveAndRestore(metric= "accuracy", mode="max")]
# Train the model using the fit method.
# The training process uses the training dataset, validation dataset, number of epochs, and batch size as parameters.
m.fit(train, valid, epochs=10, batch_size=1, callbacks=cb)


# Use the predict function to make predictions on new examples.
# This demonstrates how the trained model can be applied to classify new, unseen text samples.
m.predict(["The Kansas City Chiefs won Superbowl LVIII.",  "Yahoo pages to get touch-up "])