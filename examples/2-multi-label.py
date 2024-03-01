##########################################################################
#
# This script demonstrates the process of creating a custom dataset for multi-label classification
# and training a model with MLMC (Machine Learning for Multi-Classification). It showcases how to
# handle a dataset consisting of text data with multiple labels per instance, suitable for scenarios
# where each sample might belong to more than one category.
#
##########################################################################

import mlmc
from mlmc.data import MultiLabelDataset

# Define a set of class names that represents the categories in our dataset.
# In this example, classes include sports events, technology news, NFL-related content, and web developments.
classes = {"Sport", "Sci/Tech", "NFL", "Web"}

# Input data (`x`): A list of strings, each representing a document or text snippet.
# No preprocessing is required as MLMC models are designed to handle raw text.

x = [
    "The Kansas City Chiefs won Superbowl LIV.",
    "The Kansas City Chiefs won Superbowl LVII.",
    "Running back Willis McGahee has asked the Buffalo Bills to trade him",
    "Ibrahim Wins Egypt\'s First Gold in 56 Years",
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

# Labels (`y`): A list of lists, where each sublist contains the labels for the corresponding text snippet in `x`.
# Each label must match an entry in the `classes` set. This format supports multiple labels per instance.
y = [["Sport", "NFL"], ["Sport","NFL"], ["Sport", "NFL"],
     ["Sport"], ["Sport"], ["Sport", "NFL"], ["Sport"], ["Sci/Tech"], ["Sci/Tech", "Web"], ["Sci/Tech", "Web"], ["Sci/Tech"], ["Sci/Tech", "Web"], ["Sci/Tech"]]

# Instantiate a MultiLabelDataset with our data and labels. This dataset type is suited for multi-label classification tasks.
dataset = MultiLabelDataset(x=x, y=y, classes=classes)


# Initialize a model for multi-label classification, specifying model parameters and configuration.
# - `representation`: Specifies the model architecture or pretrained model to use as a basis for feature extraction.
# - `target="multi"`: Indicates that this model is configured for multi-label classification.
# - `finetune`: Specifies the finetuning strategy to adapt the pretrained model to our specific task.
# - `optimizer_params`: Additional parameters for the optimizer, such as learning rate.
# Uncomment `device="cuda:0"` to use GPU acceleration if available.

m = mlmc.models.KimCNN(
    representation="google/bert_uncased_L-12_H-256_A-4",
    classes=classes,
    target="multi",
    finetune="LoRA",
    optimizer_params={"lr":1e-3},
    # device="cuda:0" # uncomment if gpu
)


# Train the model on the dataset. Specify the number of epochs and batch size.
m.fit(dataset, epochs=20, batch_size=4)

# Predict labels for new unseen examples using the trained model.
m.predict(["The Kansas City Chiefs won Superbowl LVIII.", "Yahoo pages to get touch-up "])
