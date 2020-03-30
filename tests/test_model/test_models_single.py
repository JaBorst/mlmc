import mlmc
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def assertion_function(model_type, **kwargs):
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
    assertion_function(model_type=mlmc.models.KimCNN, mode="untrainable", representation="glove50")

def test_XMLCNN():
    assertion_function(model_type=mlmc.models.XMLCNN, representation="glove50", mode="untrainable")

def test_LSAN_transformer():
    assertion_function(model_type=mlmc.models.LSANOriginalTransformerNoClasses, representation="roberta", n_layers=1)

