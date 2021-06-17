##########################################################################
#
# Example for  using a zero shot model (experimental feature)
#
##########################################################################

import mlmc

# Zeroshot Models can be found in the mlmc.models.zeroshot submodule and can be instantiated without classes.
# Zeroshot classifiy by using information from the label name. So make sure that the label names are semantic.
# You can add a generic formatting function to make sure every label gets translated into a meaningful sentence or
# label name. This help improving larger language models.

formatter = lambda x: f"The topic of this is {x}"

# These and the label mappings for datasets in mlmc can also be found in:
mlmc.data.SFORMATTER

# For instance:
print(mlmc.data.SFORMATTER["trec6"]("DESC"))

# Instantiation of the smallest possible model. This should work for any computer.
m = mlmc.models.zeroshot.EmbeddingBasedWeighted(
    classes={},
    target="single",
    mode="vanilla",
    finetune=True,
    optimizer_params={"lr":1e-3},
)
# If you have a more capable computer and even a GPU, you can use this instantiation to load a larger model
# and put it on the GPU.
# m = mlmc.models.zeroshot.EmbeddingBasedWeighted(
#     classes={},
#     target="single",
#     mode="vanilla",
#     sformatter=formatter,
#     finetune=True,
#     optimizer_params={"lr": 1e-5},
#     device="cuda:3",  # If you have a GPU uncomment this
#     representation="google/bert_uncased_L-12_H-768_A-12"
#
# )

# Zeroshot models have a method to switch and create new target classes
classes1 = {"Sports":0, "Politics":1, "Science and Tech":2}
classes2 = {"Vacation": 0, "Work":1}

m.create_labels(classes1)

print(m.predict("Did you watch the football match last night?"))
print(m.predict("The laws of physics are complex."))

m.create_labels(classes2)
m.predict("I think I'll take a trip to Italy this week.")
m.predict("I'm gonna have to stay late to finish this project.")

# These models can still be trained to match a specific set of labels
# For example download agnews data. A data set containing news texts assigned to 4 categories
data = mlmc.data.get("agnews")

# We can switch to the label set of the data and evaluate. The model should have arount 49% accuracy without
# seeing one training example:
m.create_labels(data["classes"])
print("Accuracy without training:", m.evaluate(data["test"])[1]["accuracy"])


# This Performance can still be improved of course by showing some of the training data.
# We'll only take around 10 exapmles per class!
train = mlmc.data.sampler(data["train"], absolute=40)
print(train.count())


# Use the fit method to train the model
m.loss = mlmc.loss.RelativeRankingLoss(0.5)
m.fit(train, epochs=100)    # RelativeRankingLoss might go to zero. This is not bad thing but you can interrupt
                            # the training at this point or set the number of epochs accodingly


# For the large model this should achieve around 85 % accuracy, for the smaller instantiation around 76% accuracy.
print("Accuracy after seeing few examples:", m.evaluate(data["test"])[1]["accuracy"])
