import mlmc
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def assertion_function(model_type, **kwargs):
    classes = [
        {"class1": 0, "class2": 1, "class3": 2},
        {"class1": 0, "class3": 1},
        {"class4": 0, "class5": 1, "class6": 2},
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
    data = mlmc.data.dataset_classes.MultiOutputSingleLabelDataset(x=x, y=y, classes=classes)
    model = model_type(classes=classes, **kwargs, target="single", optimizer_params={"lr": 0.001})
    history = model.fit(train=data, epochs=5, batch_size=3)
    assert len(history["train"]["loss"]) == 5, "Number of Epochs not reached"

def test_MoKimCNN():
    assertion_function(model_type=mlmc.models.MoKimCNN, filters=10, kernel_sizes=[3,4])

def test_MoLSANNC():
    assertion_function(model_type=mlmc.models.MoLSANNC, filters=10, kernel_sizes=[3,4])

def test_MoTransformer():
    assertion_function(model_type=mlmc.models.MoTransformer, label_model="google/bert_uncased_L-2_H-128_A-2", hidden_representations=10, d_a=10)


