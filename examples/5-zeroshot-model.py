##########################################################################
#
# Example for  using a zero shot model (experimental feature)
#
##########################################################################
import torch

import mlmc

# Zeroshot Models can be found in the mlmc.models.zeroshot submodule and can be instantiated without classes.
# Zeroshot classifiy by using information from the label name. So make sure that the label names are semantic.
# You can add a generic formatting function to make sure every label gets translated into a meaningful sentence or
# label name. This help improving larger language models.

formatter = lambda x: f"The topic of this is {x}"

# Instantiation of the smallest possible model. This should work for any computer.
m = mlmc.models.zeroshot.Siamese(
    classes={},
    target="single",
    representation="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    finetune="all",
    optimizer_params={"lr":1e-5},
    device="cuda:0"
)


# If you have a more capable computer and even a GPU, you can use this instantiation to load a larger model
# and put it on the GPU.


# Zeroshot models have a method to switch and create new target classes
classes1 = {"Sports":0, "Science":1}
classes2 = {"Vacation": 0, "Work":1}

m.create_labels(classes1)

print(m.predict("Did you watch the football match last night?"))
print(m.predict("The laws of physics are complex."))

m.create_labels(classes2)
print(m.predict("I think I'll take a trip to Italy this week."))
print(m.predict("I'm gonna have to stay late to finish this project."))

# These models can still be trained to match a specific set of labels
# For example download agnews data. A data set containing news texts assigned to 4 categories
data, classes = mlmc.data.load_agnews()
data = {k: mlmc.data.SingleLabelDataset(x=v[0], y=v[1], classes=classes) for k,v in data.items()}


####
# The larger model benefits from the formatter options. A formatter translates a label name into a short sentence.

# We can switch to the label set of the data and evaluate. The smaller model achieve around 40 % accuracy, the
# larger model should land around 36% accuracy without seeing any training example and without formatter:
m.set_sformatter(lambda x: x)
m.create_labels(classes)
print("Accuracy without training:", m.evaluate(data["test"])[1]["accuracy"])

# Texts can formatter really do make a difference. Using the provided formatter for agnews,
# this should increase to  51% for the smaller model and about 60 % for the larger model.
m.set_sformatter(lambda x: f"This is news about {x}")
m.create_labels(classes)
print("Accuracy without training with formatter:", m.evaluate(data["test"])[1]["accuracy"])



# This Performance can still be improved of course by showing some of the training data.
# We'll only take around 10 exapmles per class.
train = mlmc.data.sampler(data["train"], absolute=40)
print(train.count())


# Use the fit method to train the model
# m.loss = mlmc.loss.RelativeRankingLoss(0.5)
m.single()
m.fit(train, epochs=10)    # RelativeRankingLoss might go to zero. This is not bad thing but you can interrupt
                            # the training at this point or set the number of epochs accordingly


# For the large model this should achieve for the larger model over 80 % accuracy, for the smaller instantiation around 70% accuracy.
# depending on the information contained in the sampled examples
print("Accuracy after seeing few examples:", m.evaluate(data["test"])[1]["accuracy"])
