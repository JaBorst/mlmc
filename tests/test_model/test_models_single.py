import mlmc
import torch
from pathlib import Path
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

def assertion_function(model_type, **kwargs):

    mlmc.representation.custom_embedding("custom", dir_path / "custom_embedding.txt")
    classes = {"label_%i" % (i,): i for i in range(5)}
    data = mlmc.data.SingleLabelDataset(
        x = ["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
        y = [["label_0"], [ "label_2"], [ "label_3"], ["label_1"], ["label_4"]],
        classes = classes
    )
    model = model_type(classes, **kwargs, target="single", optimizer_params={"lr": 0.001})
    history = model.fit(train=data, epochs=15, batch_size=3)
    assert len(history["train"]["loss"]) == 15, "Number of Epochs not reached"

def test_KimCNN():
    assertion_function(model_type=mlmc.models.KimCNN, mode="untrainable", representation="custom")

def test_XMLCNN():
    assertion_function(model_type=mlmc.models.XMLCNN, representation="custom", mode="untrainable")

def test_LSAN_transformer():
    assertion_function(model_type=mlmc.models.LSANOriginalTransformerNoClasses, representation="roberta", n_layers=1)

